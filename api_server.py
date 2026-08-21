import argparse
import asyncio
import base64
import functools
import gc
import io
import json
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
    QueueStatusResponse,
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
GCS_BUCKET = os.environ.get("GCS_BUCKET", "synode-trellis-iframe")
GCS_GLB_PREFIX = "generated"

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


JOB_STATE_PATH = os.environ.get("JOB_STATE_PATH", "/app/tmp/jobs.json")
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_JOB_PREFIX = "trellis:job:"
REDIS_JOB_TTL = 3600

try:
    import redis.asyncio as aioredis 
    _redis_available = True
except ImportError:
    _redis_available = False

_redis_client: "aioredis.Redis | None" = None
_redis_init_lock = asyncio.Lock()
_file_store_lock = asyncio.Lock()


async def _get_redis() -> "aioredis.Redis | None":
    """Return a cached async Redis client, or None if Redis is unavailable."""
    global _redis_client
    if not _redis_available or not REDIS_URL:
        return None
    if _redis_client is not None:
        return _redis_client
    async with _redis_init_lock:
        if _redis_client is None:
            try:
                _redis_client = aioredis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                await _redis_client.ping()
                logger.info(f"Redis connected: {REDIS_URL}")
            except Exception as exc:
                logger.warning(f"Redis unavailable ({exc}) — using file-based job store")
                _redis_client = None
    return _redis_client


def _job_to_dict(job: "Job") -> dict:
    return {
        "job_id": job.job_id,
        "request": job.request.model_dump(),
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "result": job.result.model_dump() if job.result else None,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def _dict_to_job(d: dict) -> "Job":
    result = GenerateResponse(**d["result"]) if d.get("result") else None
    return Job(
        job_id=d["job_id"],
        request=GenerateRequest(**d["request"]),
        status=JobStatus(d["status"]),
        progress=d.get("progress", 0.0),
        message=d.get("message", ""),
        result=result,
        error=d.get("error", ""),
        created_at=d.get("created_at", time.time()),
        started_at=d.get("started_at"),
        completed_at=d.get("completed_at"),
    )


def _write_jobs_file_sync(snapshot: dict) -> None:
    os.makedirs(os.path.dirname(JOB_STATE_PATH), exist_ok=True)
    tmp_path = JOB_STATE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(snapshot, f)
    os.replace(tmp_path, JOB_STATE_PATH)


async def _flush_jobs_to_file() -> None:
    async with _file_store_lock:
        snapshot = {jid: _job_to_dict(j) for jid, j in jobs.items()}
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _write_jobs_file_sync, snapshot)
        except Exception as exc:
            logger.warning(f"Job file write failed: {exc}")


def _load_jobs_from_file() -> dict[str, "Job"]:
    if not os.path.exists(JOB_STATE_PATH):
        return {}
    try:
        with open(JOB_STATE_PATH) as f:
            raw = json.load(f)
        restored = {}
        for jid, d in raw.items():
            try:
                restored[jid] = _dict_to_job(d)
            except Exception as exc:
                logger.warning(f"Skipping malformed job {jid}: {exc}")
        logger.info(f"Restored {len(restored)} job(s) from {JOB_STATE_PATH}")
        return restored
    except Exception as exc:
        logger.warning(f"Could not read job file ({exc}) — starting fresh")
        return {}


async def _save_job_redis(job: "Job") -> None:
    r = await _get_redis()
    if r is None:
        return
    key = f"{REDIS_JOB_PREFIX}{job.job_id}"
    try:
        payload = json.dumps(_job_to_dict(job))
        ttl = REDIS_JOB_TTL if job.completed_at is not None else 0
        if ttl:
            await r.set(key, payload, ex=ttl)
        else:
            await r.set(key, payload)
    except Exception as exc:
        logger.warning(f"Redis save failed for job {job.job_id}: {exc}")


async def _load_all_jobs_from_redis() -> dict[str, "Job"]:
    r = await _get_redis()
    if r is None:
        return {}
    restored: dict[str, "Job"] = {}
    try:
        keys = await r.keys(f"{REDIS_JOB_PREFIX}*")
        for key in keys:
            raw = await r.get(key)
            if raw:
                try:
                    job = _dict_to_job(json.loads(raw))
                    restored[job.job_id] = job
                except Exception as exc:
                    logger.warning(f"Could not deserialize job from Redis key {key}: {exc}")
        if restored:
            logger.info(f"Restored {len(restored)} job(s) from Redis")
    except Exception as exc:
        logger.warning(f"Redis restore failed: {exc}")
    return restored


async def _save_job(job: "Job") -> None:
    """Persist job — tries Redis first, falls back to file store."""
    r = await _get_redis()
    if r is not None:
        try:
            await _save_job_redis(job)
            return
        except Exception as exc:
            logger.warning(f"Redis save failed for job {job.job_id}, falling back to file: {exc}")
    await _flush_jobs_to_file()


async def _load_all_jobs() -> dict[str, "Job"]:
    """Load all jobs on startup — Redis takes priority over file store."""
    r = await _get_redis()
    if r is not None:
        result = await _load_all_jobs_from_redis()
        if result:
            return result
    return _load_jobs_from_file()


jobs: dict[str, Job] = {}
job_queue: asyncio.Queue = None
worker_task: asyncio.Task = None
cleanup_task: asyncio.Task = None
processing_lock = asyncio.Lock()

JOB_TTL_SECONDS = 3600

_avg_generation_time: float = 120.0


def _is_busy() -> bool:
    """Return True when a job is processing or queued."""
    return any(
        j.status in (JobStatus.PROCESSING, JobStatus.QUEUED)
        for j in jobs.values()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, job_queue, worker_task, cleanup_task

    model_id = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()

    job_queue = asyncio.Queue()

    restored = await _load_all_jobs()
    jobs.update(restored)

    interrupted = [
        j for j in jobs.values()
        if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
    ]
    for j in interrupted:
        j.status = JobStatus.FAILED
        j.error = "Server restarted — job was interrupted"
        j.message = "Job cancelled due to server restart"
        j.completed_at = time.time()
        await _save_job(j)
    if interrupted:
        logger.info(f"Cancelled {len(interrupted)} interrupted job(s) after restart")

    worker_task = asyncio.create_task(job_worker())
    cleanup_task = asyncio.create_task(job_cleanup_worker())
    logger.info("Job worker and cleanup worker started")

    yield

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
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass


app = FastAPI(title="TRELLIS.2 API", version="1.0.0", lifespan=lifespan)

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if pipeline is None:
        return HealthResponse(status="loading", weights_loaded=False)
    return HealthResponse(status="ok", weights_loaded=True)


@app.get("/queue/status", response_model=QueueStatusResponse)
async def queue_status():
    """Public queue status — used by waiting clients to show busy state."""
    all_jobs = list(jobs.values())
    processing = [j for j in all_jobs if j.status == JobStatus.PROCESSING]
    queued    = [j for j in all_jobs if j.status == JobStatus.QUEUED]

    processing_count = len(processing)
    queued_count     = len(queued)
    total_active     = processing_count + queued_count
    busy             = total_active > 0

    estimated_wait: float | None = None
    if busy:
        remaining = 0.0
        if processing:
            ref = processing[0].started_at
            elapsed = time.time() - (ref if ref is not None else time.time())
            remaining = max(0.0, _avg_generation_time - elapsed)
        estimated_wait = round(remaining + queued_count * _avg_generation_time, 1)

    return QueueStatusResponse(
        busy=busy,
        processing_count=processing_count,
        queued_count=queued_count,
        total_active=total_active,
        estimated_wait_seconds=estimated_wait,
    )


@app.post("/generate", response_model=JobResponse)
async def generate(request: GenerateRequest):
    """Create a new generation job.

    Returns 409 if a job is already running or queued — only one job at a
    time is supported (single GPU). Clients should poll /queue/status and
    disable the generate button while busy.
    """
    if pipeline is None or job_queue is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if _is_busy():
        raise HTTPException(
            status_code=409,
            detail="Server is busy — a generation job is already running. Please wait for it to finish."
        )

    try:
        image_bytes = base64.b64decode(request.image)
        Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        request=request,
        status=JobStatus.QUEUED,
        message="Job queued",
    )
    jobs[job_id] = job
    await _save_job(job)
    await job_queue.put(job_id)
    logger.info(f"Job {job_id} created and queued")

    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status/progress of a job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    queue_position: int | None = None
    if job.status == JobStatus.QUEUED:
        queued_jobs = sorted(
            [j for j in list(jobs.values()) if j.status == JobStatus.QUEUED],
            key=lambda j: j.created_at,
        )
        try:
            queue_position = queued_jobs.index(job) + 1
        except ValueError:
            queue_position = None

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
        queue_position=queue_position,
    )


async def job_worker():
    while True:
        try:
            job_id = await job_queue.get()
            job = jobs.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found in storage")
                job_queue.task_done()
                continue
            async with processing_lock:
                await process_job(job)
            job_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Job worker cancelled")
            break
        except Exception as e:
            logger.error(f"Job worker error: {e}\n{traceback.format_exc()}")


async def job_cleanup_worker():
    while True:
        try:
            await asyncio.sleep(300)
            now = time.time()
            expired = [
                jid for jid, job in list(jobs.items())
                if job.completed_at is not None
                and now - job.completed_at > JOB_TTL_SECONDS
            ]
            for jid in expired:
                jobs.pop(jid, None)
            if expired:
                logger.info(f"Evicted {len(expired)} expired job(s) from memory")
        except asyncio.CancelledError:
            logger.info("Cleanup worker cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup worker error: {e}")


async def process_job(job: Job):
    global pipeline, _avg_generation_time

    job.status = JobStatus.PROCESSING
    job.started_at = time.time()
    job.progress = 0.0
    job.message = "Starting generation..."
    logger.info(f"Processing job {job.job_id}")
    await _save_job(job)

    loop = asyncio.get_event_loop()

    try:
        job.message = "Decoding image..."
        job.progress = 5.0
        image_bytes = base64.b64decode(job.request.image)
        image = Image.open(io.BytesIO(image_bytes))
        del image_bytes  # free base64 buffer early

        pipeline_type = PIPELINE_TYPE_MAP.get(job.request.pipeline_type, job.request.pipeline_type)

        job.message = "Generating sparse structure..."
        job.progress = 10.0
        await _save_job(job)

        sparse_structure_sampler_params = {
            "steps": job.request.ss_sampling_steps,
            "guidance_strength": job.request.ss_guidance_strength,
            "guidance_rescale": job.request.ss_guidance_rescale,
            "rescale_t": job.request.ss_rescale_t,
        }
        shape_slat_sampler_params = {
            "steps": job.request.shape_slat_sampling_steps,
            "guidance_strength": job.request.shape_slat_guidance_strength,
            "guidance_rescale": job.request.shape_slat_guidance_rescale,
            "rescale_t": job.request.shape_slat_rescale_t,
        }
        tex_slat_sampler_params = {
            "steps": job.request.tex_slat_sampling_steps,
            "guidance_strength": job.request.tex_slat_guidance_strength,
            "guidance_rescale": job.request.tex_slat_guidance_rescale,
            "rescale_t": job.request.tex_slat_rescale_t,
        }

        # ── Stage 1: Sparse structure ────────────────────────────────────────
        def _run_sparse():
            proc_img = pipeline.preprocess_image(image)
            torch.manual_seed(job.request.seed)
            cond_512 = pipeline.get_cond([proc_img], 512)
            cond_1024 = pipeline.get_cond([proc_img], 1024) if pipeline_type != "512" else None
            ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
            coords = pipeline.sample_sparse_structure(
                cond_512, ss_res, 1, sparse_structure_sampler_params
            )
            return cond_512, cond_1024, coords

        try:
            cond_512, cond_1024, coords = await loop.run_in_executor(None, _run_sparse)
        except AttributeError:
            # Pipeline doesn't expose step-by-step API — fall back to full run
            run_kwargs = dict(
                seed=job.request.seed,
                preprocess_image=True,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params=sparse_structure_sampler_params,
                shape_slat_sampler_params=shape_slat_sampler_params,
                tex_slat_sampler_params=tex_slat_sampler_params,
            )
            meshes = await loop.run_in_executor(
                None, functools.partial(pipeline.run, image, **run_kwargs)
            )
            if not meshes:
                raise RuntimeError("Pipeline returned no meshes — check input image quality")
        else:
            # ── Stage 2: Shape SLat ──────────────────────────────────────────
            job.message = "Generating 3D shape..."
            job.progress = 25.0
            await _save_job(job)

            def _run_shape():
                if pipeline_type == "512":
                    shape_slat = pipeline.sample_shape_slat(
                        cond_512, pipeline.models["shape_slat_flow_model_512"],
                        coords, shape_slat_sampler_params,
                    )
                    resolution = 512
                elif pipeline_type == "1024":
                    shape_slat = pipeline.sample_shape_slat(
                        cond_1024, pipeline.models["shape_slat_flow_model_1024"],
                        coords, shape_slat_sampler_params,
                    )
                    resolution = 1024
                elif pipeline_type == "1024_cascade":
                    shape_slat, resolution = pipeline.sample_shape_slat_cascade(
                        cond_512, cond_1024,
                        pipeline.models["shape_slat_flow_model_512"],
                        pipeline.models["shape_slat_flow_model_1024"],
                        512, 1024,
                        coords, shape_slat_sampler_params,
                    )
                else:  # 1536_cascade
                    shape_slat, resolution = pipeline.sample_shape_slat_cascade(
                        cond_512, cond_1024,
                        pipeline.models["shape_slat_flow_model_512"],
                        pipeline.models["shape_slat_flow_model_1024"],
                        512, 1536,
                        coords, shape_slat_sampler_params,
                    )
                return shape_slat, resolution

            shape_slat, resolution = await loop.run_in_executor(None, _run_shape)

            # ── Stage 3: Texture SLat ────────────────────────────────────────
            job.message = "Generating texture..."
            job.progress = 50.0
            await _save_job(job)

            def _run_texture():
                if pipeline_type == "512":
                    flow_model = pipeline.models["tex_slat_flow_model_512"]
                else:
                    flow_model = pipeline.models["tex_slat_flow_model_1024"]
                tex_slat = pipeline.sample_tex_slat(
                    cond_1024 if cond_1024 is not None else cond_512,
                    flow_model,
                    shape_slat,
                    tex_slat_sampler_params,
                )
                return tex_slat

            tex_slat = await loop.run_in_executor(None, _run_texture)

            # ── Stage 4: Decode ──────────────────────────────────────────────
            job.message = "Decoding mesh & texture..."
            job.progress = 65.0
            await _save_job(job)

            def _run_decode():
                return pipeline.decode_latent(shape_slat, tex_slat, resolution)

            meshes = await loop.run_in_executor(None, _run_decode)

            if not meshes:
                raise RuntimeError("Pipeline returned no meshes — check input image quality")

        job.message = "Processing mesh..."
        job.progress = 70.0
        await _save_job(job)
        mesh = meshes[0]
        await loop.run_in_executor(None, functools.partial(mesh.simplify, 16777216))

        del meshes
        gc.collect()
        if torch.cuda.is_available():
            # Offload pipeline weights to CPU to free VRAM for GLB export
            pipeline.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Pipeline offloaded to CPU — VRAM freed for GLB export")

        job.message = "Exporting & uploading GLB..."
        job.progress = 80.0
        await _save_job(job)
        try:
            glb_url = await loop.run_in_executor(
                None, functools.partial(_export_and_upload_glb, mesh, job.request, job.job_id)
            )
        finally:
            # Always reload pipeline back to GPU regardless of export success/failure
            if torch.cuda.is_available():
                pipeline.cuda()
                logger.info("Pipeline reloaded to GPU")

        job.message = "Finalizing..."
        job.progress = 95.0

        generation_time = time.time() - job.started_at
        job.result = GenerateResponse(
            glb_url=glb_url,
            vertices=int(mesh.vertices.shape[0]),
            faces=int(mesh.faces.shape[0]),
            generation_time=round(generation_time, 2),
        )

        del mesh
        gc.collect()

        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.message = "Generation completed"
        job.completed_at = time.time()
        await _save_job(job)

        _avg_generation_time = _avg_generation_time * 0.8 + generation_time * 0.2
        logger.info(f"Job {job.job_id} completed in {generation_time:.2f}s (avg: {_avg_generation_time:.1f}s)")

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.message = f"Generation failed: {exc}"
        job.completed_at = time.time()
        await _save_job(job)
        logger.error(f"Job {job.job_id} failed: {exc}\n{traceback.format_exc()}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU cache cleared for job {job.job_id}")



def _export_and_upload_glb(mesh, request: GenerateRequest, job_id: str) -> str:
    """Export mesh to GLB, upload to GCS, return public URL."""
    logger.info("Starting GLB export...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("Memory freed before GLB export")

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
            remesh=False,
            verbose=False,
        )
        logger.info("GLB object created successfully")
    except MemoryError:
        logger.error("OOM during GLB creation")
        raise MemoryError(
            "Not enough RAM to export this model. "
            "Try reducing Texture size (e.g. 1024 px) or GLB decimation target."
        )
    except Exception as exc:
        logger.error(f"GLB object creation failed: {exc}\n{traceback.format_exc()}")
        raise

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb_path = tmp.name

    try:
        logger.info(f"Exporting GLB to temporary file: {glb_path}")
        glb.export(glb_path, extension_webp=True)
        logger.info("GLB export to file successful")

        del glb
        gc.collect()

        object_name = f"{GCS_GLB_PREFIX}/{job_id}.glb"
        bucket = _get_gcs_client().bucket(GCS_BUCKET)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(glb_path, content_type="model/gltf-binary")
        public_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{object_name}"
        logger.info(f"GLB uploaded to GCS: {public_url}")
        return public_url

    except MemoryError:
        logger.error("OOM during GLB file export")
        raise MemoryError(
            "Not enough RAM to write the GLB file. "
            "Try reducing Texture size (e.g. 1024 px) or GLB decimation target."
        )
    except Exception as exc:
        logger.error(f"GLB export/upload failed: {exc}\n{traceback.format_exc()}")
        raise
    finally:
        if os.path.exists(glb_path):
            os.unlink(glb_path)
            logger.info(f"Temporary file cleaned up: {glb_path}")



def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
