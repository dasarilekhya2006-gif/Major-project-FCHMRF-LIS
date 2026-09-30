"""End-to-end pipeline execution and asynchronous status tracking API routes."""

from fastapi import APIRouter
from backend.schemas.pipeline import (
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
)
from backend.schemas.common import ErrorResponse
from backend.services.pipeline_service import pipeline_service

router = APIRouter(tags=["Pipeline"])


@router.post(
    "/run-pipeline",
    response_model=PipelineRunResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid pipeline request"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Trigger end-to-end research pipeline",
    description="Asynchronously runs Preprocessing -> 3D U-Net -> Statistics -> fcHMRF-LIS -> FDR control.",
)
def run_complete_pipeline(req: PipelineRunRequest) -> PipelineRunResponse:
    return pipeline_service.trigger_complete_pipeline(req)


@router.get(
    "/pipeline/status/{experiment_id}",
    response_model=PipelineStatusResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Check pipeline execution state and progress",
    description="Returns current stage (e.g. preprocessing, fchmrf-lis, completed) and progress indicator.",
)
def get_pipeline_status(experiment_id: str) -> PipelineStatusResponse:
    return pipeline_service.get_job_status(experiment_id)
