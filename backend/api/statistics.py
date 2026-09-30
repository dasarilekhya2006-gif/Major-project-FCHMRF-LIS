"""Statistics API route for voxel-level hypothesis testing."""

from fastapi import APIRouter
from backend.schemas.pipeline import StatisticsRequest, StageResponse
from backend.schemas.common import ErrorResponse
from backend.services.pipeline_service import pipeline_service

router = APIRouter(tags=["Statistics"])


@router.post(
    "/statistics",
    response_model=StageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment or input volume not found"},
    },
    summary="Compute voxel-level test statistics and p-values",
    description="Generates z-score maps and two-sided p-values restricted to brain tissue.",
)
def compute_statistics(req: StatisticsRequest) -> StageResponse:
    return pipeline_service.run_statistics_stage(experiment_id=req.experiment_id)
