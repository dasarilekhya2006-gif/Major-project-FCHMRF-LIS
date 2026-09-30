"""Segmentation API route for 3D U-Net ROI extraction."""

from fastapi import APIRouter
from backend.schemas.pipeline import SegmentRequest, StageResponse
from backend.schemas.common import ErrorResponse
from backend.services.pipeline_service import pipeline_service

router = APIRouter(tags=["Segmentation"])


@router.post(
    "/segment",
    response_model=StageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment or preprocessed volume not found"},
    },
    summary="Execute 3D U-Net ROI segmentation",
    description="Generates prior probability map and binary ROI mask while preserving spatial coordinates.",
)
def segment_volume(req: SegmentRequest) -> StageResponse:
    return pipeline_service.run_segmentation_stage(
        experiment_id=req.experiment_id,
        probability_threshold=req.probability_threshold,
    )
