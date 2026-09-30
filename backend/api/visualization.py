"""Visualization API routes specifically tailored for the Phase 10 interactive frontend."""

from typing import Optional
from fastapi import APIRouter, Query
from backend.schemas.visualization import (
    VisualizationMetadataResponse,
    SliceDataResponse,
    Mesh3DResponse,
)
from backend.schemas.common import ErrorResponse
from backend.services.visualization_service import visualization_service

router = APIRouter(tags=["Visualization"])


@router.get(
    "/visualization/{experiment_id}",
    response_model=VisualizationMetadataResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Get visualization configuration and value bounds",
    description="Provides layer availability, volume shape, voxel spacing, affine matrix, and value min/max bounds.",
)
def get_visualization_info(experiment_id: str) -> VisualizationMetadataResponse:
    return visualization_service.get_metadata(experiment_id)


@router.get(
    "/visualization/{experiment_id}/slice",
    response_model=SliceDataResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Experiment or slice data not found"},
    },
    summary="Get 2D tri-planar slice data preserving raw scientific values",
    description="Returns raw 2D arrays for grayscale MRI base and transparent abnormality heatmap overlay without cosmetic distortion.",
)
def get_slice_data(
    experiment_id: str,
    orientation: str = Query("axial", description="Anatomical plane: axial, coronal, or sagittal"),
    slice_idx: Optional[int] = Query(None, description="Slice index along specified anatomical orientation"),
) -> SliceDataResponse:
    return visualization_service.get_slice_data(
        experiment_id=experiment_id,
        orientation=orientation,
        slice_idx=slice_idx,
    )


@router.get(
    "/visualization/{experiment_id}/3d-mesh",
    response_model=Mesh3DResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Get 3D Marching Cubes surface meshes",
    description="Returns vertices and faces for brain shell, 3D U-Net ROI, and significant abnormalities for 360-degree rotation.",
)
def get_3d_mesh_data(
    experiment_id: str,
    step_size: int = Query(1, ge=1, le=4, description="Marching cubes step size for mesh decimation"),
) -> Mesh3DResponse:
    return visualization_service.get_3d_mesh_data(
        experiment_id=experiment_id,
        step_size=step_size,
    )
