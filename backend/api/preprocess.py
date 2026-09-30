"""Preprocessing API route."""

from fastapi import APIRouter
from backend.schemas.pipeline import PreprocessRequest, StageResponse
from backend.schemas.common import ErrorResponse
from backend.services.pipeline_service import pipeline_service

router = APIRouter(tags=["Preprocessing"])


@router.post(
    "/preprocess",
    response_model=StageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Preprocess uploaded MRI/fMRI volume",
    description="Executes intensity normalization, missing voxel handling, and brain extraction.",
)
def preprocess_volume(req: PreprocessRequest) -> StageResponse:
    return pipeline_service.run_preprocess_stage(
        experiment_id=req.experiment_id,
        normalize_intensity=req.normalize_intensity,
        apply_brain_mask=req.apply_brain_mask,
    )
