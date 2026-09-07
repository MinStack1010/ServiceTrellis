from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class GenerateRequest(BaseModel):

    images: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=4,
        description=(
            "1-4 Base64-encoded PNG/JPEG views of the same object. Multiple "
            "views require a trained multi-view conditioning checkpoint."
        ),
    )
    image: Optional[str] = Field(
        default=None,
        description="Deprecated single-image alias; use images=[image] instead.",
        deprecated=True,
    )
    conditioning_mode: Literal["auto", "single_view", "multi_view"] = Field(
        default="auto",
        description=(
            "auto selects by image count. multi_view is available only when a "
            "trained fusion/adapter checkpoint is loaded."
        ),
    )
    camera_metadata: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Optional per-view camera metadata. Its feature count must match "
            "the loaded multi-view checkpoint; poses are never inferred."
        ),
    )
    seed: int = Field(default=0, ge=0, le=4294967295, description="Random seed")
    pipeline_type: str = Field(
        default="1024_cascade",
        description="Pipeline type: 512, 1024, 1024_cascade, 1536_cascade",
    )
    decimation_target: int = Field(
        default=500000, ge=10000, le=2000000,
        description="Maximum target face count for mesh decimation"
    )
    texture_size: int = Field(default=1024, ge=512, le=4096, description="Texture resolution")

    ss_guidance_strength: float = Field(default=7.5, ge=1, le=10)
    ss_guidance_rescale: float = Field(default=0.7, ge=0, le=1)
    ss_sampling_steps: int = Field(default=12, ge=1, le=50)
    ss_rescale_t: float = Field(default=5.0, ge=1, le=6)

    shape_slat_guidance_strength: float = Field(default=7.5, ge=1, le=10)
    shape_slat_guidance_rescale: float = Field(default=0.5, ge=0, le=1)
    shape_slat_sampling_steps: int = Field(default=12, ge=1, le=50)
    shape_slat_rescale_t: float = Field(default=3.0, ge=1, le=6)

    tex_slat_guidance_strength: float = Field(default=1.0, ge=1, le=10)
    tex_slat_guidance_rescale: float = Field(default=0.0, ge=0, le=1)
    tex_slat_sampling_steps: int = Field(default=12, ge=1, le=50)
    tex_slat_rescale_t: float = Field(default=3.0, ge=1, le=6)

    @model_validator(mode="after")
    def normalize_image_inputs(self) -> "GenerateRequest":
        if self.images is None:
            if self.image is None:
                raise ValueError("Provide images with 1-4 Base64-encoded images")
            self.images = [self.image]
        elif self.image is not None:
            raise ValueError("Use either images or the deprecated image field, not both")

        if self.conditioning_mode == "single_view" and len(self.images) != 1:
            raise ValueError("conditioning_mode='single_view' requires exactly one image")
        if self.conditioning_mode == "multi_view" and len(self.images) < 2:
            raise ValueError("conditioning_mode='multi_view' requires 2-4 images")
        if self.camera_metadata is not None and len(self.camera_metadata) != len(self.images):
            raise ValueError("camera_metadata must contain exactly one entry per image")
        return self


class GenerateResponse(BaseModel):

    glb_url: str = Field(..., description="Public GCS URL of the exported GLB file")
    vertices: int = Field(..., description="Number of vertices in the output mesh")
    faces: int = Field(..., description="Number of faces in the output mesh")
    generation_time: float = Field(..., description="Generation time in seconds")


class HealthResponse(BaseModel):

    status: str = "ok"
    weights_loaded: bool = False
    multi_view_configured: bool = False
    multi_view_checkpoint_loaded: bool = False


class JobStatus(str, Enum):
    """Job status enum."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")


class JobStatusResponse(BaseModel):

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage (0-100)")
    message: str = Field(default="", description="Status message")
    result: Optional[GenerateResponse] = Field(default=None, description="Generation result when completed")
    error: str = Field(default="", description="Error message when failed")
    queue_position: Optional[int] = Field(
        default=None,
        description="Position in queue (1 = next to run). None when not queued."
    )


class QueueStatusResponse(BaseModel):

    busy: bool = Field(..., description="True when at least one job is processing or queued")
    processing_count: int = Field(default=0, description="Number of jobs currently processing (0 or 1)")
    queued_count: int = Field(default=0, description="Number of jobs waiting in queue")
    total_active: int = Field(default=0, description="processing_count + queued_count")
    estimated_wait_seconds: Optional[float] = Field(
        default=None,
        description="Rough ETA in seconds based on average generation time. None if unknown.",
    )
