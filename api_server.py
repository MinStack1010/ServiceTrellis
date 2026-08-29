import argparse
import asyncio
import base64
import functools
import gc
import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import o_voxel
import psutil
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
		if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
			ttl = REDIS_JOB_TTL
		else:
			ttl = REDIS_JOB_TTL * 4
		await r.set(key, payload, ex=ttl)
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
	r = await _get_redis()
	if r is not None:
		try:
			await _save_job_redis(job)
			return
		except Exception as exc:
			logger.warning(f"Redis save failed for job {job.job_id}, falling back to file: {exc}")
	await _flush_jobs_to_file()


async def _load_all_jobs() -> dict[str, "Job"]:
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
	return any(
		j.status in (JobStatus.PROCESSING, JobStatus.QUEUED)
		for j in jobs.values()
	)


def _model_to_gpu(model_key: str) -> None:
	if torch.cuda.is_available() and pipeline is not None:
		m = pipeline.models.get(model_key)
		if m is not None:
			m.cuda()
			logger.debug(f"[stage] {model_key} → GPU")


def _model_to_cpu(model_key: str) -> None:
	if pipeline is not None:
		m = pipeline.models.get(model_key)
		if m is not None:
			m.cpu()
			logger.debug(f"[stage] {model_key} → CPU")
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
	global pipeline, job_queue, worker_task, cleanup_task, _fbx_semaphore

	model_id = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
	
	pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
	pipeline.cpu()
	logger.info("Pipeline loaded into CPU RAM (per-stage GPU loading enabled)")

	_fbx_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BLENDER)
	logger.info(f"FBX concurrency limit: {MAX_CONCURRENT_BLENDER} parallel Blender processes")

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

_BLOCKED_PREFIXES = (
	"/.env", "/.git", "/.aws", "/.docker",
	"/wp-", "/phpinfo", "/info.php",
	"/backup", "/config.", "/secrets.",
	"/terraform.", "/docker-compose",
	"/actuator", "/server.js", "/app.js",
	"/credentials", "/site.zip", "/www.zip",
)

@app.middleware("http")
async def block_scanners(request: Request, call_next):
	path = request.url.path
	if any(path.startswith(p) or path == p.rstrip(".") for p in _BLOCKED_PREFIXES):
		logger.debug(f"Scanner probe blocked: {request.client.host} {path}")
		return JSONResponse(status_code=404, content={"detail": "Not found"})
	return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
	logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
	return JSONResponse(
		status_code=500,
		content={"detail": f"Internal server error: {str(exc)}"}
	)


@app.get("/health", response_model=HealthResponse)
async def health():
	ready = pipeline is not None and bool(getattr(pipeline, "models", None))
	if not ready:
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
			expired = []
			for jid, job in list(jobs.items()):
				if job.completed_at is not None and now - job.completed_at > JOB_TTL_SECONDS:
					expired.append(jid)
				elif job.status == JobStatus.FAILED and job.completed_at is None:
					if job.started_at is not None and now - job.started_at > JOB_TTL_SECONDS:
						expired.append(jid)
					elif job.created_at is not None and now - job.created_at > JOB_TTL_SECONDS:
						expired.append(jid)
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
		del image_bytes

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

		def _run_generation():
			pipeline.cpu()
			for _ in range(5):
				gc.collect()
			if torch.cuda.is_available():
				torch.cuda.synchronize()
				torch.cuda.empty_cache()
				torch.cuda.reset_peak_memory_stats()
				torch.cuda.reset_accumulated_memory_stats()
				torch.cuda.empty_cache()
				free_vram = torch.cuda.mem_get_info()[0] / 1024**3
				logger.info(f"VRAM free before generation: {free_vram:.1f} GiB")

			S = {}

			pipeline._device = torch.device("cuda")
			try:
				S['proc_img'] = pipeline.preprocess_image(image)
				torch.manual_seed(job.request.seed)
				S['cond_512'] = pipeline.get_cond([S['proc_img']], 512)
				S['cond_1024'] = pipeline.get_cond([S['proc_img']], 1024) if pipeline_type != "512" else None
				del S['proc_img']

				ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
				S['coords'] = pipeline.sample_sparse_structure(
					S['cond_512'], ss_res, 1, sparse_structure_sampler_params
				)

				if pipeline_type == "512":
					S['shape_slat'] = pipeline.sample_shape_slat(
						S['cond_512'], pipeline.models["shape_slat_flow_model_512"],
						S['coords'], shape_slat_sampler_params,
					)
					resolution = 512
				elif pipeline_type == "1024":
					S['shape_slat'] = pipeline.sample_shape_slat(
						S['cond_1024'], pipeline.models["shape_slat_flow_model_1024"],
						S['coords'], shape_slat_sampler_params,
					)
					resolution = 1024
				elif pipeline_type == "1024_cascade":
					S['shape_slat'], resolution = pipeline.sample_shape_slat_cascade(
						S['cond_512'], S['cond_1024'],
						pipeline.models["shape_slat_flow_model_512"],
						pipeline.models["shape_slat_flow_model_1024"],
						512, 1024,
						S['coords'], shape_slat_sampler_params,
					)
				else:
					S['shape_slat'], resolution = pipeline.sample_shape_slat_cascade(
						S['cond_512'], S['cond_1024'],
						pipeline.models["shape_slat_flow_model_512"],
						pipeline.models["shape_slat_flow_model_1024"],
						512, 1536,
						S['coords'], shape_slat_sampler_params,
					)

				del S['coords']
				if pipeline_type != "512":
					del S['cond_512']
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()

				if pipeline_type == "512":
					S['tex_cond'] = S.pop('cond_512')
					tex_model_key = "tex_slat_flow_model_512"
				else:
					S['tex_cond'] = S['cond_1024']
					tex_model_key = "tex_slat_flow_model_1024"

				S['tex_slat'] = pipeline.sample_tex_slat(
					S['tex_cond'],
					pipeline.models[tex_model_key],
					S['shape_slat'],
					tex_slat_sampler_params,
				)
				del S['tex_cond']
				if 'cond_1024' in S:
					del S['cond_1024']
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()

				meshes = pipeline.decode_latent(S['shape_slat'], S['tex_slat'], resolution)
				del S['shape_slat'], S['tex_slat']
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()

				return meshes, resolution

			finally:

				S.clear()
				pipeline._device = torch.device("cpu")
				for _ in range(5):
					gc.collect()
				pipeline.cpu()
				if torch.cuda.is_available():
					torch.cuda.synchronize()
					torch.cuda.empty_cache()
					torch.cuda.reset_peak_memory_stats()
					torch.cuda.reset_accumulated_memory_stats()
					torch.cuda.empty_cache()
					free_vram = torch.cuda.mem_get_info()[0] / 1024**3
					logger.info(f"VRAM free after job: {free_vram:.1f} GiB")

		try:
			meshes, resolution = await loop.run_in_executor(None, _run_generation)
		except AttributeError:
			def _run_full():
				pipeline._device = torch.device("cuda")
				try:
					return pipeline.run(
						image,
						seed=job.request.seed,
						preprocess_image=True,
						pipeline_type=pipeline_type,
						sparse_structure_sampler_params=sparse_structure_sampler_params,
						shape_slat_sampler_params=shape_slat_sampler_params,
						tex_slat_sampler_params=tex_slat_sampler_params,
					)
				finally:
					pipeline._device = torch.device("cpu")
					gc.collect()
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
			meshes = await loop.run_in_executor(None, _run_full)
			resolution = None

		del image
		gc.collect()

		if not meshes:
			raise RuntimeError("Pipeline returned no meshes — check input image quality")

		job.message = "Processing mesh..."
		job.progress = 70.0
		await _save_job(job)
		mesh = meshes[0]
		del meshes
		for _ in range(3):
			gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
			logger.info("VRAM cleared — all models are on CPU, ready for GLB export")

		mesh_vertices = mesh.vertices
		mesh_faces = mesh.faces
		mesh_attrs = mesh.attrs
		mesh_coords = mesh.coords
		mesh_layout = mesh.layout
		mesh_voxel_size = mesh.voxel_size
		n_vertices = int(mesh_vertices.shape[0])
		n_faces = int(mesh_faces.shape[0])
		del mesh
		for _ in range(3):
			gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
		logger.info(f"RAM before GLB export: {_rss_mb:.0f} MB")

		job.message = "Exporting & uploading GLB..."
		job.progress = 80.0
		await _save_job(job)
		glb_url = await loop.run_in_executor(
			None, functools.partial(
				_export_and_upload_glb,
				mesh_vertices, mesh_faces, mesh_attrs, mesh_coords,
				mesh_layout, mesh_voxel_size,
				job.request, job.job_id,
			)
		)
		del mesh_vertices, mesh_faces, mesh_attrs, mesh_coords, mesh_layout, mesh_voxel_size
		for _ in range(3):
			gc.collect()

		job.message = "Finalizing..."
		job.progress = 95.0

		generation_time = time.time() - job.started_at
		job.result = GenerateResponse(
			glb_url=glb_url,
			vertices=n_vertices,
			faces=n_faces,
			generation_time=round(generation_time, 2),
		)

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
			torch.cuda.synchronize()
			logger.info(f"GPU cache cleared for job {job.job_id}")

		for _ in range(5):
			gc.collect()
		try:
			import ctypes
			ctypes.CDLL("libc.so.6").malloc_trim(0)
			logger.info("malloc_trim: RAM returned to OS")
		except Exception:
			pass
		_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
		logger.info(f"RAM after job cleanup: {_rss_mb:.0f} MB")



def _export_and_upload_glb(
	vertices, faces, attrs, coords, layout, voxel_size,
	request: GenerateRequest, job_id: str,
) -> str:

	_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
	logger.info(f"_export_and_upload_glb start — RAM: {_rss_mb:.0f} MB")

	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
	for _ in range(3):
		gc.collect()

	_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
	logger.info(f"Memory freed before GLB export — RAM: {_rss_mb:.0f} MB")

	_GLB_TIMEOUT = int(os.environ.get("GLB_EXPORT_TIMEOUT", "300"))  # 5 min hard cap

	try:
		logger.info(
			f"to_glb params: texture_size={request.texture_size} "
			f"decimation_target={request.decimation_target}"
		)
		glb = o_voxel.postprocess.to_glb(
			vertices=vertices,
			faces=faces,
			attr_volume=attrs,
			coords=coords,
			attr_layout=layout,
			voxel_size=voxel_size,
			aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
			decimation_target=request.decimation_target,
			texture_size=request.texture_size,
			remesh=False,
			verbose=False,
		)
		_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
		logger.info(f"GLB object created successfully — RAM: {_rss_mb:.0f} MB")
	except MemoryError:
		logger.error("OOM during GLB creation")
		raise MemoryError(
			"Not enough RAM to export this model. "
			"Try reducing Texture size (e.g. 1024 px) or GLB decimation target."
		)
	except Exception as exc:
		logger.error(f"GLB object creation failed: {exc}\n{traceback.format_exc()}")
		raise

	del vertices, faces, attrs, coords, layout, voxel_size
	for _ in range(3):
		gc.collect()

	with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
		glb_path = tmp.name

	try:
		logger.info(f"Exporting GLB to temporary file: {glb_path}")
		glb.export(glb_path, extension_webp=False)
		logger.info("GLB export to file successful (PNG textures for Blender compatibility)")

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


# FBX export cache
_fbx_cache: dict[str, tuple[str, float]] = {}  # glb_url -> (fbx_url, timestamp)
_FBX_CACHE_TTL = 3600  # 1 hour
_fbx_locks: dict[str, asyncio.Lock] = {}  # per-URL lock to prevent duplicate conversions
# Concurrency limit: each Blender process uses ~1GB CPU RAM. With 31GB total
# and ~14GB free, max 2 concurrent Blender processes is safe.
MAX_CONCURRENT_BLENDER = int(os.environ.get("MAX_CONCURRENT_BLENDER", "2"))
_fbx_semaphore: asyncio.Semaphore | None = None  # initialized in startup


def _get_fbx_cache_key(glb_url: str) -> str:
	"""Generate safe, deterministic cache key from GLB URL."""
	return hashlib.sha256(glb_url.encode()).hexdigest()[:32]


def _find_blender_executable() -> str:

	explicit = os.environ.get("BLENDER_EXECUTABLE")
	if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
		return explicit

	candidates = [
		"/usr/bin/blender",
		"/usr/local/bin/blender",
		"/opt/blender/blender",
		# macOS .app bundles
		"/Applications/Blender.app/Contents/MacOS/Blender",
		os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
		# Homebrew / Linuxbrew
		"/opt/homebrew/bin/blender",
		"/usr/local/homebrew/bin/blender",
	]
	for c in candidates:
		if os.path.isfile(c) and os.access(c, os.X_OK):
			return c

	try:
		import shutil
		w = shutil.which("blender")
		if w:
			return w
	except Exception:
		pass

	# Fallback: return the env default; caller will surface FileNotFoundError clearly
	return explicit or "/usr/bin/blender"


async def _convert_glb_to_fbx(glb_url: str) -> str:
	"""Convert GLB to FBX using Blender script."""
	import aiohttp

	# Check cache first (with TTL enforcement)
	cache_key = _get_fbx_cache_key(glb_url)
	if cache_key in _fbx_cache:
		cached_fbx_url, cached_ts = _fbx_cache[cache_key]
		if time.time() - cached_ts < _FBX_CACHE_TTL:
			# Verify cached URL is still reachable (only fail on explicit 404)
			try:
				async with aiohttp.ClientSession() as session:
					async with session.head(cached_fbx_url) as resp:
						if resp.status != 404:
							logger.info(f"FBX cache hit for {glb_url}")
							return cached_fbx_url
						logger.warning(f"Cached FBX returned 404, regenerating")
			except (aiohttp.ClientError, asyncio.TimeoutError):
				# Transient network error — keep cache, assume valid
				logger.info(f"FBX cache hit (HEAD check failed transiently) for {glb_url}")
				return cached_fbx_url
		else:
			logger.info(f"FBX cache expired for {glb_url}")
			del _fbx_cache[cache_key]

	# Per-URL lock prevents duplicate concurrent Blender conversions
	if cache_key not in _fbx_locks:
		_fbx_locks[cache_key] = asyncio.Lock()

	async with _fbx_locks[cache_key]:
		# Double-check: another coroutine may have completed while we waited
		if cache_key in _fbx_cache:
			cached_fbx_url, cached_ts = _fbx_cache[cache_key]
			if time.time() - cached_ts < _FBX_CACHE_TTL:
				logger.info(f"FBX cache hit after lock wait for {glb_url}")
				return cached_fbx_url

		logger.info(f"Converting GLB to FBX: {glb_url}")

		with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb_tmp:
			glb_path = glb_tmp.name

		fbx_path = glb_path.replace(".glb", ".fbx")

		try:
			# Download GLB from URL
			async with aiohttp.ClientSession() as session:
				async with session.get(glb_url) as resp:
					if resp.status != 200:
						raise HTTPException(status_code=400, detail=f"Failed to download GLB: HTTP {resp.status}")
					glb_content = await resp.read()

			if len(glb_content) == 0:
				raise HTTPException(status_code=400, detail="Downloaded GLB is empty (0 bytes)")

			with open(glb_path, "wb") as f:
				f.write(glb_content)

			logger.info(f"GLB saved to temp file: {glb_path} ({len(glb_content)} bytes)")

			# Locate Blender executable
			blender_script_path = os.path.join(os.path.dirname(__file__), "blender_glb_to_fbx.py")

			if not os.path.exists(blender_script_path):
				raise HTTPException(status_code=500, detail=f"Blender conversion script not found: {blender_script_path}")

			blender_exec = _find_blender_executable()
			logger.info(f"Using Blender executable: {blender_exec}")

			if not os.path.isfile(blender_exec):
				raise HTTPException(
					status_code=500,
					detail=(
						f"Blender executable not found at '{blender_exec}'. "
						f"Set BLENDER_EXECUTABLE env var or install Blender. "
						f"Searched: /usr/bin/blender, /usr/local/bin/blender, "
						f"/Applications/Blender.app/Contents/MacOS/Blender, /opt/homebrew/bin/blender"
					)
				)

			# Run Blender in background mode (with concurrency limit)
			cmd = [
				blender_exec, "-b", "-P", blender_script_path, "--",
				glb_path, fbx_path
			]

			logger.info(f"Running Blender command: {' '.join(cmd)}")

			loop = asyncio.get_running_loop()
			try:
				if _fbx_semaphore is None:
					raise HTTPException(status_code=503, detail="FBX conversion not ready — server still starting up")
				async with _fbx_semaphore:
					available = _fbx_semaphore._value if hasattr(_fbx_semaphore, '_value') else '?'
					logger.info(f"Blender semaphore acquired ({available} remaining)")
					process = await loop.run_in_executor(
						None,
						lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=300)
					)
			except FileNotFoundError as exc:
				raise HTTPException(
					status_code=500,
					detail=f"Blender executable not found: {blender_exec}. Set BLENDER_EXECUTABLE env var. ({exc})"
				)
			except subprocess.TimeoutExpired:
				raise HTTPException(
					status_code=500,
					detail="FBX conversion timed out (Blender took >300s)"
				)

			# Always log Blender output (stdout + stderr) for debugging
			if process.stdout:
				logger.info(f"Blender stdout: {process.stdout}")
			if process.stderr:
				logger.warning(f"Blender stderr: {process.stderr}")

			if process.returncode != 0:
				stdout_text = (process.stdout or '').strip()
				stderr_text = (process.stderr or '').strip()
				logger.error(f"Blender conversion failed (exit {process.returncode})")
				logger.error(f"Blender stdout: {stdout_text}")
				logger.error(f"Blender stderr: {stderr_text}")
				raise HTTPException(
					status_code=500,
					detail=(
						f"FBX conversion failed (Blender exit code {process.returncode}). "
						f"stdout: {stdout_text[:4000]}\nstderr: {stderr_text[:4000]}"
					)
				)

			logger.info("Blender process exited 0 — verifying output file...")

			if not os.path.exists(fbx_path):
				logger.error(
					f"Blender returned exit code 0 but FBX file does NOT exist: {fbx_path}. "
					f"stdout: {(process.stdout or '')[:500]}. "
					f"stderr: {(process.stderr or '')[:500]}"
				)
				raise HTTPException(
					status_code=500,
					detail=(
						"Blender reported success but wrote no FBX output file. "
						"This usually means the GLB import or FBX export failed silently inside Blender. "
						"Check server logs for Blender's stdout/stderr."
					)
				)

			fbx_size = os.path.getsize(fbx_path)
			if fbx_size == 0:
				logger.error(f"Blender returned exit code 0 but FBX file is EMPTY: {fbx_path}")
				raise HTTPException(
					status_code=500,
					detail="Blender reported success but FBX output is 0 bytes."
				)

			logger.info(f"Blender conversion successful: {fbx_path} ({fbx_size} bytes)")

			# Upload FBX to GCS (cache_key is already a safe hex hash)
			object_name = f"{GCS_GLB_PREFIX}/fbx/{cache_key}.fbx"
			bucket = _get_gcs_client().bucket(GCS_BUCKET)
			blob = bucket.blob(object_name)
			blob.upload_from_filename(fbx_path, content_type="application/octet-stream")
			public_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{object_name}"

			# Cache the result with timestamp
			_fbx_cache[cache_key] = (public_url, time.time())

			logger.info(f"FBX uploaded to GCS: {public_url}")
			return public_url

		finally:
			# Clean up temporary files
			if os.path.exists(glb_path):
				try:
					os.unlink(glb_path)
					logger.info(f"Cleaned up temp GLB: {glb_path}")
				except Exception as exc:
					logger.warning(f"Failed to clean up temp GLB {glb_path}: {exc}")
			if os.path.exists(fbx_path):
				try:
					os.unlink(fbx_path)
					logger.info(f"Cleaned up temp FBX: {fbx_path}")
				except Exception as exc:
					logger.warning(f"Failed to clean up temp FBX {fbx_path}: {exc}")


@app.post("/export/fbx")
async def export_fbx(request: dict):
	"""Export GLB to FBX format using Blender."""
	
	glb_url = request.get("glb_url")
	if not glb_url:
		raise HTTPException(status_code=400, detail="glb_url is required")
	
	try:
		fbx_url = await _convert_glb_to_fbx(glb_url)
		return {"fbx_url": fbx_url, "status": "success"}
	except HTTPException:
		raise
	except Exception as exc:
		logger.error(f"FBX export error: {exc}\n{traceback.format_exc()}")
		raise HTTPException(status_code=500, detail=f"FBX export failed: {str(exc)}")


@app.post("/export/fbx/cache/clear")
async def clear_fbx_cache():
	"""Clear FBX export cache."""
	_fbx_cache.clear()
	return {"status": "success", "message": "FBX cache cleared"}


def main():
	parser = argparse.ArgumentParser(description="TRELLIS.2 API Server")
	parser.add_argument("--host", type=str, default="0.0.0.0")
	parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
	args = parser.parse_args()
	uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
	main()
