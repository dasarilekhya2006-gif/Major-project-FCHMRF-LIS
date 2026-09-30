"""Pydantic request and response data contracts for the research REST API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    project: str = "Adaptive Deep Learning-Assisted fcHMRF-LIS"


class SyntheticGenerateRequest(BaseModel):
    grid_size: List[int] = Field(default_factory=lambda: [48, 48, 24])
    snr: float = 2.0
    num_clusters: int = 3
    seed: int = 42


class SyntheticGenerateResponse(BaseModel):
    sample_id: str
    shape: List[int]
    snr: float
    true_active_voxels: int
    total_voxels: int
    max_zscore: float
    min_zscore: float


class InferenceRequest(BaseModel):
    sample_id: Optional[str] = None
    nominal_alpha: float = Field(default=0.05, ge=0.001, le=0.5)
    use_dl_prior: bool = True
    adaptive_tuning: bool = True
    beta: float = Field(default=0.8, ge=0.0, le=5.0)
    lambda_dl: float = Field(default=0.5, ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    sample_id: str
    nominal_alpha: float
    num_discoveries: int
    discovery_percentage: float
    lis_cutoff_threshold: float
    estimated_fdr: float
    beta_used: float
    lambda_dl_used: float
    execution_time_seconds: float
    method: str


class MethodBenchmarkResult(BaseModel):
    method: str
    empirical_fdr: float
    power_tpr: float
    fndr: float
    f1_score: float
    dice: float
    mcc: float
    num_discoveries: int


class BenchmarkRunResponse(BaseModel):
    nominal_alpha: float
    snr: float
    results: List[MethodBenchmarkResult]
