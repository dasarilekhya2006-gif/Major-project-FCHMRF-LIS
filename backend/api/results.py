"""Results retrieval and experiment metadata API routes."""

from typing import Any, Dict, List
from fastapi import APIRouter
from fastapi.responses import FileResponse
from backend.schemas.results import ResultsResponse, ExperimentMetadataResponse
from backend.schemas.common import ErrorResponse
from backend.services.result_service import result_service
from backend.services.storage_service import storage_service

router = APIRouter(tags=["Results", "Experiments"])


@router.get(
    "/experiments",
    response_model=List[Dict[str, Any]],
    summary="List all experiments",
    description="Returns list of past experiments with metadata, timestamps, and stage statuses for experiment history.",
)
def list_experiments() -> List[Dict[str, Any]]:
    return storage_service.list_experiments()


@router.get(
    "/results/{experiment_id}",
    response_model=ResultsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Get all available research result layers",
    description="Returns layer availability flags, discovery counts, and safe artifact references.",
)
def get_experiment_results(experiment_id: str) -> ResultsResponse:
    return result_service.get_results(experiment_id)


@router.get(
    "/experiment/{experiment_id}",
    response_model=ExperimentMetadataResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID"},
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
    summary="Get comprehensive experiment metadata and history",
    description="Returns timestamps, dimensions, voxel spacing, orientation, and pipeline stage execution status.",
)
def get_experiment_details(experiment_id: str) -> ExperimentMetadataResponse:
    return result_service.get_experiment_metadata(experiment_id)


@router.get(
    "/download/{experiment_id}/{stage}/{filename}",
    summary="Download research artifact",
    description="Safely downloads a generated NIfTI volume, mask, or metadata JSON without exposing server paths.",
)
def download_artifact(experiment_id: str, stage: str, filename: str):
    file_path = storage_service.get_artifact_path(experiment_id, stage, filename)
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )
