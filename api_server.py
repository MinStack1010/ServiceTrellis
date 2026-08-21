"""
FastAPI server for Microsoft TRELLIS.2 image-to-3D generation.

- GET  /health   — server status
- POST /generate — base64 image → GLB (uploaded to GCS, returns public URL)

Usage:
    python api_server.py --port 8080
"""
import argparse
import asyncio
import base64
import functools
import io
import logging
import os
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import o_voxel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import storage as gcs
from PIL import Image

from api_models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)
from trellis2.pipelines import Trellis2ImageTo3DPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", os.environ.get("ATTN_BACKEND", "sdpa"))
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ.get("SPARSE_ATTN_BACKEND", "sdpa"))
os.environ.setdefault("SPARSE_CONV_BACKEND", os.environ.get("SPARSE_CONV_BACKEND", "none"))

PIPELINE_TYPE_MAP = {
    "512": "512",
    "1024": "1024",
    "1024_cascade": "1024_cascade",
    "1536": "1536_cascade",
    "1536_cascade": "1536_cascade",
}

# GCS bucket where generated GLB files are stored.
# The bucket must have "allUsers → Storage Object Viewer" for public access.
GCS_BUCKET = os.environ.get("GCS_BUCKET", "synode-trellis-iframe")
GCS_GLB_PREFIX = "generated"   # objects are stored as generated/{job_id}.glb

pipeline: Trellis2ImageTo3DPipeline | None = None
_gcs_client: gcs.Client | None = None


def _get_gcs_client() -> gcs.Client:
    """Return a cached GCS client (uses Workload Identity / ADC automatically)."""
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = gcs.Client()
    return _gcs_client


@dataclass
class Job:
    """Job data structure."""
    job_id: str
    request: GenerateRequest
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    result: Optional[GenerateResponse] = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# In-memory job storage
jobs: dict[str, Job] = {}
job_queue: asyncio.Queue = None
worker_task: asyncio.Task = None
cleanup_task: asyncio.Task = None
processing_lock = asyncio.Lock()

# Jobs are removed from memory this many seconds after they finish (success or failure).
# This prevents unbounded RAM growth from accumulated GLB payloads.
JOB_TTL_SECONDS = 3600  # 1 hour


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, job_queue, worker_task, cleanup_task

    model_id = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    
    # Initialize job queue and start worker
    job_queue = asyncio.Queue()
    worker_task = asyncio.create_task(job_worker())
    cleanup_task = asyncio.create_task(job_cleanup_worker())
    logger.info("Job worker and cleanup worker started")
    
    yield
    
    # Cleanup
    for task in (worker_task, cleanup_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="TRELLIS.2 API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if pipeline is None:
        return HealthResponse(status="loading", weights_loaded=False)
    return HealthResponse(status="ok", weights_loaded=True)


def _export_and_upload_glb(mesh, request: GenerateRequest, job_id: str) -> str:
    """Export mesh to GLB, upload to GCS, return public URL.

    Returns the public HTTPS URL of the uploaded GLB object.
    The temporary local file is always cleaned up before returning.
    """
    logger.info("Starting GLB export...")
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=request.decimation_target,
            texture_size=request.texture_size,
            remesh=False,   # remesh=True costs 1–4 min CPU; decimation alone is sufficient
            verbose=False,
        )
        logger.info("GLB object created successfully")
    except Exception as exc:
        logger.error(f"GLB object creation failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb_path = tmp.name

    try:
        logger.info(f"Exporting GLB to temporary file: {glb_path}")
        glb.export(glb_path, extension_webp=True)
        logger.info("GLB export to file successful")

        # Upload to GCS
        object_name = f"{GCS_GLB_PREFIX}/{job_id}.glb"
        bucket = _get_gcs_client().bucket(GCS_BUCKET)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(glb_path, content_type="model/gltf-binary")
        # The bucket has allUsers → Storage Object Viewer, so every uploaded
        # object is publicly readable without an explicit make_public() call.
        public_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{object_name}"
        logger.info(f"GLB uploaded to GCS: {public_url}")
        return public_url

    except Exception as exc:
        logger.error(f"GLB export/upload failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if os.path.exists(glb_path):
            os.unlink(glb_path)
            logger.info(f"Temporary file cleaned up: {glb_path}")


async def job_worker():
    """Background worker that processes jobs one at a time."""
    while True:
        try:
            job_id = await job_queue.get()
            job = jobs.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found in storage")
                continue
            
            async with processing_lock:
                await process_job(job)
            
            job_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Job worker cancelled")
            break
        except Exception as e:
            logger.error(f"Job worker error: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")


async def job_cleanup_worker():
    """Periodically evict completed/failed jobs older than JOB_TTL_SECONDS.

    Each finished job can hold tens of MB of base64 GLB data in RAM.
    Without cleanup the server will eventually OOM on long-running deployments.
    """
    while True:
        try:
            await asyncio.sleep(300)  # check every 5 minutes
            now = time.time()
            expired = [
                job_id
                for job_id, job in list(jobs.items())
                if job.completed_at is not None
                and now - job.completed_at > JOB_TTL_SECONDS
            ]
            for job_id in expired:
                jobs.pop(job_id, None)
            if expired:
                logger.info(f"Evicted {len(expired)} expired job(s) from memory")
        except asyncio.CancelledError:
            logger.info("Cleanup worker cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup worker error: {e}")


async def process_job(job: Job):
    """Process a single job.

    pipeline.run() and _export_glb() are both CPU/GPU-bound blocking calls.
    Running them directly on the asyncio event loop would freeze the entire
    server (including /health and /jobs/{id} endpoints) for the full
    generation duration (30-120 s).  We offload them to a thread-pool
    executor so the event loop stays responsive.
    """
    global pipeline
    
    job.status = JobStatus.PROCESSING
    job.started_at = time.time()
    job.progress = 0.0
    job.message = "Starting generation..."
    logger.info(f"Processing job {job.job_id}")
    
    loop = asyncio.get_event_loop()

    try:
        # Decode image
        job.message = "Decoding image..."
        job.progress = 5.0
        image_bytes = base64.b64decode(job.request.image)
        image = Image.open(io.BytesIO(image_bytes))
        
        pipeline_type = PIPELINE_TYPE_MAP.get(job.request.pipeline_type, job.request.pipeline_type)
        
        # Run generation — blocking GPU call, offloaded to thread pool
        job.message = "Generating sparse structure..."
        job.progress = 10.0
        
        run_kwargs = dict(
            seed=job.request.seed,
            preprocess_image=True,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={
                "steps": job.request.ss_sampling_steps,
                "guidance_strength": job.request.ss_guidance_strength,
                "guidance_rescale": job.request.ss_guidance_rescale,
                "rescale_t": job.request.ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": job.request.shape_slat_sampling_steps,
                "guidance_strength": job.request.shape_slat_guidance_strength,
                "guidance_rescale": job.request.shape_slat_guidance_rescale,
                "rescale_t": job.request.shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": job.request.tex_slat_sampling_steps,
                "guidance_strength": job.request.tex_slat_guidance_strength,
                "guidance_rescale": job.request.tex_slat_guidance_rescale,
                "rescale_t": job.request.tex_slat_rescale_t,
            },
        )
        meshes = await loop.run_in_executor(
            None, functools.partial(pipeline.run, image, **run_kwargs)
        )
        
        job.message = "Processing mesh..."
        job.progress = 70.0
        mesh = meshes[0]
        await loop.run_in_executor(None, functools.partial(mesh.simplify, 16777216))
        
        job.message = "Exporting & uploading GLB..."
        job.progress = 80.0
        glb_url = await loop.run_in_executor(
            None, functools.partial(_export_and_upload_glb, mesh, job.request, job.job_id)
        )
        
        job.message = "Finalizing..."
        job.progress = 95.0
        
        generation_time = time.time() - job.started_at
        job.result = GenerateResponse(
            glb_url=glb_url,
            vertices=int(mesh.vertices.shape[0]),
            faces=int(mesh.faces.shape[0]),
            generation_time=round(generation_time, 2),
        )
        
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.message = "Generation completed"
        job.completed_at = time.time()
        logger.info(f"Job {job.job_id} completed in {generation_time:.2f}s")
        
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.message = f"Generation failed: {exc}"
        job.completed_at = time.time()
        logger.error(f"Job {job.job_id} failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    finally:
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU cache cleared for job {job.job_id}")


@app.post("/generate", response_model=JobResponse)
async def generate(request: GenerateRequest):
    """Create a new generation job and return job ID immediately."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Validate image
    try:
        image_bytes = base64.b64decode(request.image)
        Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    
    # Create job
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        request=request,
        status=JobStatus.QUEUED,
        message="Job queued",
    )
    jobs[job_id] = job
    
    # Add to queue
    await job_queue.put(job_id)
    logger.info(f"Job {job_id} created and queued")
    
    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Calculate queue position: count how many QUEUED jobs were created before this one.
    # Position 1 means this job is next to run.
    queue_position: int | None = None
    if job.status == JobStatus.QUEUED:
        earlier_queued = sum(
            1 for j in jobs.values()
            if j.status == JobStatus.QUEUED and j.created_at < job.created_at
        )
        queue_position = earlier_queued + 1

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
        queue_position=queue_position,
    )


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
