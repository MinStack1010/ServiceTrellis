"""Pydantic request/response models for the TRELLIS.2 API server."""
from enum import Enum
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request to generate a 3D model from an image."""

    image: str = Field(..., description="Base64-encoded image (PNG/JPEG)")
    seed: int = Field(default=0, ge=0, le=4294967295, description="Random seed")
    pipeline_type: str = Field(
        default="1024_cascade",
        description="Pipeline type: 512, 1024, 1024_cascade, 1536_cascade",
    )
    decimation_target: int = Field(
        default=500000, ge=10000, le=2000000,
        description="Maximum target face count for mesh decimation"
    )
    texture_size: int = Field(default=2048, ge=512, le=4096, description="Texture resolution")

    # These defaults and ranges match the official TRELLIS.2 image-to-3D UI.
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


class GenerateResponse(BaseModel):
    """Response containing the generated 3D model."""

    glb_url: str = Field(..., description="Public GCS URL of the exported GLB file")
    vertices: int = Field(..., description="Number of vertices in the output mesh")
    faces: int = Field(..., description="Number of faces in the output mesh")
    generation_time: float = Field(..., description="Generation time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    weights_loaded: bool = False


class JobStatus(str, Enum):
    """Job status enum."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Response for job creation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")


class JobStatusResponse(BaseModel):
    """Response for job status check."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage (0-100)")
    message: str = Field(default="", description="Status message")
    result: GenerateResponse | None = Field(default=None, description="Generation result when completed")
    error: str = Field(default="", description="Error message when failed")
    queue_position: int | None = Field(
        default=None,
        description="Position in queue (1 = next to run). None when not queued."
    )


class QueueStatusResponse(BaseModel):
    """Public queue status — no authentication required.

    Tells any visitor whether the server is busy and how many jobs are pending.
    Used by clients that don't own a job (e.g. a second browser/machine) so
    they can show a friendly 'busy' banner instead of letting the user try to
    generate and get rejected.
    """

    busy: bool = Field(..., description="True when at least one job is processing or queued")
    processing_count: int = Field(default=0, description="Number of jobs currently processing (0 or 1)")
    queued_count: int = Field(default=0, description="Number of jobs waiting in queue")
    total_active: int = Field(default=0, description="processing_count + queued_count")
    estimated_wait_seconds: float | None = Field(
        default=None,
        description="Rough ETA in seconds based on average generation time. None if unknown.",
    )
