"""Research results, layer availability, and experiment metadata schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LayerAvailability(BaseModel):
    available: bool = False
    result_id: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


class ResultsResponse(BaseModel):
    """Aggregate research results summary for an experiment."""

    experiment_id: str
    status: str
    original: LayerAvailability
    preprocessed: LayerAvailability
    roi: LayerAvailability
    statistical_map: LayerAvailability
    lis_map: LayerAvailability
    fdr: LayerAvailability
    ground_truth: LayerAvailability
    baseline: LayerAvailability
    proposed: LayerAvailability


class ExperimentMetadataResponse(BaseModel):
    """Complete metadata record for experiment tracking and reproducibility."""

    experiment_id: str
    created_at: str
    filename: str
    status: str
    shape: List[int]
    voxel_spacing: List[float]
    orientation: str
    stages: Dict[str, str]
    parameters: Dict[str, Any]
    discoveries_proposed: Optional[int] = None
    discoveries_baseline: Optional[int] = None
    ground_truth_available: bool = False
