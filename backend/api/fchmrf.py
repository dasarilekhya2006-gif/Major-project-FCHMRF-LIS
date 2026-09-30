"""fcHMRF-LIS spatial modeling and FDR control API route."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.schemas.pipeline import FCHMRFLISRequest, StageResponse
from backend.schemas.common import ErrorResponse
from backend.services.pipeline_service import pipeline_service
from src.core.exceptions import MemoryLimitExceededError


router = APIRouter(tags=["FCHMRF-LIS"])


@router.post(
    "/fchmrf-lis",
    response_model=StageResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid experiment ID or parameters"},
        404: {"model": ErrorResponse, "description": "Required stage inputs not found"},
        507: {"model": ErrorResponse, "description": "Spatial graph exceeds memory-safe configuration"},
    },
    summary="Execute proposed fcHMRF-LIS spatial statistical inference",
    description=(
        "Fits two-group mixture, runs sparse lattice bilateral mean-field inference "
        "(26-connectivity local lattice, no N×N dense matrix), computes LIS, and controls FDR. "
        "Returns graph diagnostic information (edge count, memory estimate, connectivity) "
        "alongside LIS and FDR results."
    ),
)
def run_fchmrf_lis(req: FCHMRFLISRequest):
    try:
        return pipeline_service.run_fchmrf_lis_stage(
            experiment_id=req.experiment_id,
            nominal_alpha=req.nominal_alpha,
            beta=req.beta,
            lambda_dl=req.lambda_dl,
            max_iterations=req.max_iterations,
        )
    except MemoryLimitExceededError as e:
        details = getattr(e, "details", {})
        return JSONResponse(
            status_code=507,  # 507 Insufficient Storage — more semantically correct
            content={
                "error": "FC_HMRF_MEMORY_LIMIT",
                "message": str(e),
                "details": details,
            },
        )
