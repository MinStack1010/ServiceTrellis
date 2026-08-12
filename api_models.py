"""Pydantic request/response models for the TRELLIS.2 API server."""
from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request to generate a 3D model from an image."""

    image: str = Field(..., description="Base64-encoded image (PNG/JPEG)")
    seed: int = Field(default=0, description="Random seed")
    pipeline_type: str = Field(
        default="1024_cascade",
        description="Pipeline type: 512, 1024, 1024_cascade, 1536_cascade",
    )
    decimation_target: int = Field(
        default=500000, description="Target face count for decimation"
    )
    texture_size: int = Field(default=2048, description="Texture resolution")
    ss_sampling_steps: int = Field(default=12, description="Sparse structure sampling steps")
    shape_slat_sampling_steps: int = Field(default=12, description="Shape sampling steps")
    tex_slat_sampling_steps: int = Field(default=12, description="Texture sampling steps")


class GenerateResponse(BaseModel):
    """Response containing the generated 3D model."""

    glb: str = Field(..., description="Base64-encoded GLB file")
    vertices: int = Field(..., description="Number of vertices in the output mesh")
    faces: int = Field(..., description="Number of faces in the output mesh")
    generation_time: float = Field(..., description="Generation time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    backend: str = "cuda"
    weights_loaded: bool = False
