"""Visualization layer metadata, 2D slice data, and 3D surface mesh schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValueRange(BaseModel):
    min: float
    max: float


class VisualizationMetadataResponse(BaseModel):
    """Metadata required by the Phase 10 frontend to configure slice sliders and layer toggles."""

    experiment_id: str
    layers: Dict[str, Dict[str, Any]]
    visualization: Dict[str, bool] = Field(default_factory=lambda: {"supports_2d": True, "supports_3d": True})
    dimensions: List[int]
    voxel_spacing: List[float]
    affine: List[List[float]]
    value_ranges: Dict[str, ValueRange]


class SliceDataResponse(BaseModel):
    """2D tri-planar slice data preserving raw scientific values for grayscale + heatmap rendering."""

    experiment_id: str
    orientation: str
    slice_idx: int
    max_slice: int
    width: int
    height: int
    mri_slice: List[List[float]]
    statistical_slice: Optional[List[List[float]]] = None
    lis_slice: Optional[List[List[float]]] = None
    roi_slice: Optional[List[List[int]]] = None
    significant_slice: Optional[List[List[int]]] = None
    ground_truth_slice: Optional[List[List[int]]] = None
    baseline_significant_slice: Optional[List[List[int]]] = None


class Mesh3DData(BaseModel):
    num_vertices: int
    num_faces: int
    x: List[float]
    y: List[float]
    z: List[float]
    i: List[int]
    j: List[int]
    k: List[int]


class Mesh3DResponse(BaseModel):
    """Reconstructed 3D surfaces for interactive 360-degree brain and abnormality visualization."""

    experiment_id: str
    brain_shell: Optional[Mesh3DData] = None
    roi_mesh: Optional[Mesh3DData] = None
    significant_mesh: Optional[Mesh3DData] = None
    ground_truth_mesh: Optional[Mesh3DData] = None
