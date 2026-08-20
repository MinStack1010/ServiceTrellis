"""
FastAPI server for Microsoft TRELLIS.2 image-to-3D generation.

- GET  /health   — server status
- POST /generate — base64 image → GLB

Usage:
    python api_server.py --port 8080
"""
import argparse
import base64
import io
import logging
import os
import tempfile
import time
import traceback
from contextlib import asynccontextmanager

import o_voxel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from api_models import GenerateRequest, GenerateResponse, HealthResponse
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

pipeline: Trellis2ImageTo3DPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    model_id = os.environ.get("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    yield
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


def _export_glb(mesh, request: GenerateRequest) -> bytes:
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
            remesh=True,
            remesh_band=1,
            remesh_project=0,
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
        with open(glb_path, "rb") as f:
            data = f.read()
            logger.info(f"GLB file read successfully, size: {len(data)} bytes")
            return data
    except Exception as exc:
        logger.error(f"GLB file export/read failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if os.path.exists(glb_path):
            os.unlink(glb_path)
            logger.info(f"Temporary file cleaned up: {glb_path}")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    t_start = time.time()

    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.error(f"Image decoding failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    pipeline_type = PIPELINE_TYPE_MAP.get(request.pipeline_type, request.pipeline_type)

    try:
        meshes = pipeline.run(
            image,
            seed=request.seed,
            preprocess_image=True,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={
                "steps": request.ss_sampling_steps,
                "guidance_strength": request.ss_guidance_strength,
                "guidance_rescale": request.ss_guidance_rescale,
                "rescale_t": request.ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": request.shape_slat_sampling_steps,
                "guidance_strength": request.shape_slat_guidance_strength,
                "guidance_rescale": request.shape_slat_guidance_rescale,
                "rescale_t": request.shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": request.tex_slat_sampling_steps,
                "guidance_strength": request.tex_slat_guidance_strength,
                "guidance_rescale": request.tex_slat_guidance_rescale,
                "rescale_t": request.tex_slat_rescale_t,
            },
        )
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    mesh = meshes[0]
    mesh.simplify(16777216)

    try:
        glb_bytes = _export_glb(mesh, request)
        logger.info(f"GLB export successful, size: {len(glb_bytes)} bytes")
    except Exception as exc:
        logger.error(f"GLB export failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"GLB export failed: {exc}") from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        response = GenerateResponse(
            glb=base64.b64encode(glb_bytes).decode(),
            vertices=int(mesh.vertices.shape[0]),
            faces=int(mesh.faces.shape[0]),
            generation_time=round(time.time() - t_start, 2),
        )
        logger.info(f"Response created successfully, generation time: {response.generation_time}s")
        return response
    except Exception as exc:
        logger.error(f"Response creation/serialization failed: {exc}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Response serialization failed: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
