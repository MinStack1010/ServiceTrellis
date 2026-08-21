"""
FastAPI server for Microsoft TRELLIS.2 image-to-3D generation.

- GET  /health   — server status
- POST /generate — base64 image → GLB (uploaded to GCS, returns public URL)

Usage:
    python api_server.py --port 8080

Environment variables:
    REDIS_URL   — Redis connection URL (e.g. redis://localhost:6379/0).
                  If not set or Redis is unreachable, falls back to in-memory
                  storage (jobs will be lost on server restart).
"""
import argparse
import asyncio
import base64
import functools
import gc
import io
import json
import logging
import os
import signal
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


# ─── Persistent Job Store ────────────────────────────────────────────────────
# Job state is written to a JSON file on disk so it survives container restarts.
# The file lives in JOB_STATE_PATH (default /app/tmp/jobs.json).
# No external service required — works with the existing Docker volume setup.
#
# Optional Redis: set REDIS_URL env var to use Redis instead (e.g. when
# running multiple replicas). Falls back to file store if Redis is unreachable.

JOB_STATE_PATH = os.environ.get("JOB_STATE_PATH", "/app/tmp/jobs.json")
REDIS_URL = os.environ.get("REDIS_URL", "")
REDIS_JOB_PREFIX = "trellis:job:"
REDIS_JOB_TTL = 3600  # 1 hour — TTL for completed/failed jobs in Redis

try:
    import redis.asyncio as aioredis  # type: ignore
    _redis_available = True
except ImportError:
    _redis_available = False

_redis_client: "aioredis.Redis | None" = None  # type: ignore[name-defined]
_redis_init_lock = asyncio.Lock()  # B-M2: ngăn tạo 2 clients đồng thời
_file_store_lock = asyncio.Lock()  # serialize file writes


async def _get_redis() -> "aioredis.Redis | None":  # type: ignore[name-defined]
    """Return a cached async Redis client, or None if Redis is unavailable."""
    global _redis_client
    if not _redis_available or not REDIS_URL:
        return None
    if _redis_client is not None:
        return _redis_client
    async with _redis_init_lock:  # B-M2: double-checked locking
        if _redis_client is None:
            try:
                _redis_client = aioredis.from_url(  # type: ignore[union-attr]
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
    """Serialize Job to a JSON-safe dict."""
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
    """Deserialize a dict back to a Job."""
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


# ── File-based store ──────────────────────────────────────────────────────────

def _write_jobs_file_sync(snapshot: dict) -> None:
    """Write jobs snapshot to disk (sync — called from asyncio via executor)."""
    os.makedirs(os.path.dirname(JOB_STATE_PATH), exist_ok=True)
    tmp_path = JOB_STATE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(snapshot, f)
    os.replace(tmp_path, JOB_STATE_PATH)  # atomic rename


async def _flush_jobs_to_file() -> None:
    """Persist all current jobs to disk (async, with lock)."""
    async with _file_store_lock:
        snapshot = {jid: _job_to_dict(j) for jid, j in jobs.items()}
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _write_jobs_file_sync, snapshot)
        except Exception as exc:
            logger.warning(f"Job file write failed: {exc}")


def _load_jobs_from_file() -> dict[str, "Job"]:
    """Load jobs from disk file on startup (sync, called before event loop tasks)."""
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


# ── Redis store ───────────────────────────────────────────────────────────────

async def _save_job_redis(job: "Job") -> None:
    """Persist job to Redis (no-op if Redis is unavailable)."""
    r = await _get_redis()
    if r is None:
        return
    key = f"{REDIS_JOB_PREFIX}{job.job_id}"
    try:
        payload = json.dumps(_job_to_dict(job))
        ttl = REDIS_JOB_TTL if job.completed_at is not None else 0
        if ttl:
            await r.setex(key, ttl, payload)
        else:
            await r.set(key, payload)
    except Exception as exc:
        logger.warning(f"Redis save failed for job {job.job_id}: {exc}")


async def _load_all_jobs_from_redis() -> dict[str, "Job"]:
    """Restore jobs from Redis on startup."""
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


# ── Unified save/load ─────────────────────────────────────────────────────────

async def _save_job(job: "Job") -> None:
    """Persist job — tries Redis first, falls back to file store."""
    r = await _get_redis()
    if r is not None:
        try:
            await _save_job_redis(job)
            return  # Redis OK, done
        except Exception as exc:
            logger.warning(f"Redis save failed for job {job.job_id}, falling back to file: {exc}")
    # B-M3: fallback luôn chạy nếu Redis không available hoặc fail
    await _flush_jobs_to_file()


async def _load_all_jobs() -> dict[str, "Job"]:
    """Load all jobs on startup — Redis takes priority over file store."""
    r = await _get_redis()
    if r is not None:
        result = await _load_all_jobs_from_redis()
        if result:
            return result
    # Fall back to file store
    return _load_jobs_from_file()


# In-memory job storage — also used as write-through cache when Redis is active
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

    # Restore jobs từ file/Redis trước khi start worker
    restored = await _load_all_jobs()
    jobs.update(restored)

    # Re-queue các jobs bị gián đoạn (QUEUED hoặc PROCESSING khi server restart).
    # PROCESSING jobs được đặt lại về QUEUED vì pipeline đã bị reset.
    interrupted = sorted(
        [j for j in jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)],
        key=lambda j: j.created_at,
    )
    for j in interrupted:
        if j.status == JobStatus.PROCESSING:
            j.status = JobStatus.QUEUED
            j.progress = 0.0
            j.message = "Re-queued after server restart"
            await _save_job(j)
        await job_queue.put(j.job_id)
    if interrupted:
        logger.info(f"Re-queued {len(interrupted)} interrupted job(s) after restart")

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
    # Đóng Redis connection
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass


app = FastAPI(title="TRELLIS.2 API", version="1.0.0", lifespan=lifespan)

# B-M6: CORS origins từ env var — set CORS_ORIGINS="https://your-domain.com" để restrict
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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


# Average generation time (seconds) used for ETA estimate.
# Seeded with a conservative default; updated after each completed job.
_avg_generation_time: float = 120.0
_completed_job_count: int = 0


@app.get("/queue/status", response_model=QueueStatusResponse)
async def queue_status():
    """Public queue status — no auth required.

    Any client (including machines that don't own a job) can call this to
    find out whether the server is busy before attempting to generate.
    """
    # Snapshot to avoid mutation during iteration
    all_jobs = list(jobs.values())
    processing = [j for j in all_jobs if j.status == JobStatus.PROCESSING]
    queued = [j for j in all_jobs if j.status == JobStatus.QUEUED]

    processing_count = len(processing)
    queued_count = len(queued)
    total_active = processing_count + queued_count
    busy = total_active > 0

    # ETA: remaining time on the current job + queue depth × avg time
    estimated_wait: float | None = None
    if busy:
        remaining_on_current = 0.0
        if processing:
            current = processing[0]
            # B-M4: dùng `is not None` thay vì `or` để tránh falsy 0.0
            ref_time = current.started_at if current.started_at is not None else time.time()
            elapsed = time.time() - ref_time
            remaining_on_current = max(0.0, _avg_generation_time - elapsed)
        estimated_wait = round(remaining_on_current + queued_count * _avg_generation_time, 1)

    return QueueStatusResponse(
        busy=busy,
        processing_count=processing_count,
        queued_count=queued_count,
        total_active=total_active,
        estimated_wait_seconds=estimated_wait,
    )


def _export_and_upload_glb(mesh, request: GenerateRequest, job_id: str) -> str:
    """Export mesh to GLB, upload to GCS, return public URL.

    Returns the public HTTPS URL of the uploaded GLB object.
    The temporary local file is always cleaned up before returning.
    """
    logger.info("Starting GLB export...")

    # Giải phóng GPU cache và Python heap trước khi export để tránh OOM.
    # Pipeline đã xong → CUDA tensors trung gian không cần nữa.
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
            remesh=False,   # remesh=True costs 1–4 min CPU; decimation alone is sufficient
            verbose=False,
        )
        logger.info("GLB object created successfully")
    except MemoryError:
        # OOM trên CPU heap — xảy ra khi texture_size hoặc decimation_target quá lớn
        logger.error("OOM during GLB creation — try reducing texture size or decimation target")
        raise MemoryError(
            "Not enough RAM to export this model. "
            "Try reducing Texture size (e.g. 1024 px) or GLB decimation target."
        )
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

        # Giải phóng GLB object khỏi RAM ngay sau khi đã ghi file
        del glb
        gc.collect()

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

    except MemoryError:
        logger.error("OOM during GLB file export")
        raise MemoryError(
            "Not enough RAM to write the GLB file. "
            "Try reducing Texture size (e.g. 1024 px) or GLB decimation target."
        )
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
                job_queue.task_done()  # B-H1 fix: không để queue block mãi
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
    Redis TTL handles expiry on the Redis side automatically (set via setex in _save_job).
    This worker only cleans up in-memory cache.
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
                # Redis TTL đã được set khi save, không cần xóa thủ công
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
    await _save_job(job)  # Bug #2 fix: persist trạng thái PROCESSING
    
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
        await _save_job(job)  # Bug #2 fix: persist tiến trình trước khi GPU call dài
        
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

        # B-M5: pipeline có thể trả list rỗng nếu ảnh quá tệ
        if not meshes:
            raise RuntimeError("Pipeline returned no meshes — check input image quality")

        job.message = "Processing mesh..."
        job.progress = 70.0
        await _save_job(job)
        mesh = meshes[0]
        await loop.run_in_executor(None, functools.partial(mesh.simplify, 16777216))

        # Giải phóng meshes list (chỉ giữ mesh[0]) để tiết kiệm RAM trước export
        del meshes
        gc.collect()

        job.message = "Exporting & uploading GLB..."
        job.progress = 80.0
        await _save_job(job)
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

        # Mesh không cần nữa sau khi đã lấy thông tin
        del mesh
        gc.collect()
        
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.message = "Generation completed"
        job.completed_at = time.time()
        await _save_job(job)  # Bug #2 fix: persist kết quả cuối cùng với TTL

        # Cập nhật moving average generation time cho ETA estimate
        global _avg_generation_time, _completed_job_count
        _completed_job_count += 1
        # Exponential moving average — trọng số cao hơn cho jobs gần đây
        _avg_generation_time = _avg_generation_time * 0.8 + generation_time * 0.2

        logger.info(f"Job {job.job_id} completed in {generation_time:.2f}s (avg: {_avg_generation_time:.1f}s)")
        
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.message = f"Generation failed: {exc}"
        job.completed_at = time.time()
        await _save_job(job)  # Bug #2 fix: persist trạng thái failed
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
    # B-M1: check cả job_queue — lifespan có thể chưa hoàn thành init
    if pipeline is None or job_queue is None:
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
    await _save_job(job)  # Bug #2 fix: persist ngay khi tạo để không mất nếu crash
    
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

    # Bug #5 fix: Tạo snapshot của jobs dict để tránh race condition khi
    # worker đang cập nhật jobs concurrently trong run_in_executor.
    # Sort theo created_at để queue_position ổn định (không nhảy số).
    queue_position: int | None = None
    if job.status == JobStatus.QUEUED:
        # Snapshot + sort ổn định theo thời gian tạo
        queued_jobs = sorted(
            [j for j in list(jobs.values()) if j.status == JobStatus.QUEUED],
            key=lambda j: j.created_at
        )
        try:
            queue_position = queued_jobs.index(job) + 1  # 1-indexed
        except ValueError:
            # Job vừa chuyển sang PROCESSING trong khoảnh khắc này
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


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
