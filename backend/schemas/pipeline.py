"""Pipeline execution requests, stage responses, and status schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    experiment_id: str = Field(..., description="Target experiment ID")
    normalize_intensity: bool = Field(default=True, description="Apply z-score intensity normalization")
    apply_brain_mask: bool = Field(default=True, description="Compute and apply binary brain extraction mask")


class StageResponse(BaseModel):
    experiment_id: str
    stage: str
    status: str
    result_id: str
    details: Optional[dict] = None


class SegmentRequest(BaseModel):
    experiment_id: str
    probability_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class StatisticsRequest(BaseModel):
    experiment_id: str
    test_type: str = Field(default="one_sample", description="Statistical hypothesis test design")


class FCHMRFLISRequest(BaseModel):
    experiment_id: str
    nominal_alpha: float = Field(default=0.05, gt=0.0, lt=0.5, description="Target False Discovery Rate")
    beta: float = Field(default=0.8, ge=0.0, le=5.0, description="Spatial coupling strength")
    lambda_dl: float = Field(default=0.5, ge=0.0, le=1.0, description="DL prior weight")
    max_iterations: int = Field(default=25, ge=5, le=100)


class PipelineRunRequest(BaseModel):
    experiment_id: str = Field(..., description="Target experiment ID")
    run_segmentation: bool = Field(default=True)
    run_statistics: bool = Field(default=True)
    run_fchmrf_lis: bool = Field(default=True)
    run_fdr: bool = Field(default=True)
    nominal_alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    beta: float = Field(default=0.8, ge=0.0, le=5.0)
    lambda_dl: float = Field(default=0.5, ge=0.0, le=1.0)


class PipelineRunResponse(BaseModel):
    experiment_id: str
    job_id: str
    status: str = "queued"
    message: str = "Pipeline computation triggered successfully."


class PipelineStatusResponse(BaseModel):
    experiment_id: str
    job_id: Optional[str] = None
    status: str = Field(
        ...,
        description="Current state: uploaded, preprocessing, segmentation, statistics, fchmrf-lis, fdr, visualization, completed, failed",
    )
    current_stage: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100, description="Estimated execution percentage")
    error: Optional[str] = None
