"""Main FastAPI backend application for Adaptive fcHMRF-LIS Neuroimaging Analysis."""

import sys
from pathlib import Path

# Ensure the project root (parent of `backend/`) is always on sys.path
# so `from backend.xxx import yyy` works regardless of where uvicorn is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.schemas.common import ErrorResponse, HealthResponse
from backend.api import api_router

logger = get_logger(__name__)

# OpenAPI metadata tags
TAGS_METADATA = [
    {"name": "Health", "description": "Service health and readiness probes."},
    {"name": "Upload", "description": "Secure NIfTI volume ingestion and validation."},
    {"name": "Preprocessing", "description": "Neuroimaging preprocessing and brain masking."},
    {"name": "Segmentation", "description": "3D U-Net prior probability and ROI mask generation."},
    {"name": "Statistics", "description": "Voxel-level hypothesis testing and p-values."},
    {"name": "FCHMRF-LIS", "description": "Dense bilateral spatial HMRF inference and LIS FDR control."},
    {"name": "Pipeline", "description": "End-to-end pipeline coordination and asynchronous status."},
    {"name": "Results", "description": "Artifact availability, discovery summaries, and metadata."},
    {"name": "Visualization", "description": "2D tri-planar slicing and 3D surface meshes for Phase 10 frontend."},
    {"name": "Experiments", "description": "Provenance and parameter tracking records."},
]


def create_app() -> FastAPI:
    """Instantiate and configure the research-grade FastAPI application."""
    settings.ensure_directories()

    app = FastAPI(
        title="Adaptive fcHMRF-LIS Neuroimaging Analysis Backend",
        description=(
            "Research-grade REST API backend supporting Phase 10 interactive neuroimaging visualization "
            "and spatial False Discovery Rate control using deep-learning-assisted fcHMRF-LIS."
        ),
        version="1.0.0",
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Middleware configured for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins if not settings.debug else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Structured Exception Handlers preventing leakage of stack traces or internal paths
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTP {exc.status_code} at {request.url.path}: {exc.detail}")
        if isinstance(exc.detail, dict):
            code = exc.detail.get("error", "HTTP_ERROR")
            msg = exc.detail.get("message", str(exc.detail))
            exp_id = exc.detail.get("experiment_id")
        else:
            code = "HTTP_ERROR"
            msg = str(exc.detail)
            exp_id = None

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=code,
                message=msg,
                experiment_id=exp_id,
                timestamp=datetime.utcnow(),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error at {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="VALIDATION_ERROR",
                message=f"Request validation failed: {exc.errors()[0].get('msg', 'Invalid parameters')}",
                timestamp=datetime.utcnow(),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error at {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred during scientific processing.",
                timestamp=datetime.utcnow(),
            ).model_dump(mode="json"),
        )

    # 3. Health Endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health_check() -> HealthResponse:
        storage_ok = settings.storage_dir.exists()
        return HealthResponse(
            status="ok",
            version="1.0.0",
            service="fcHMRF-LIS Neuroimaging Backend",
            storage_accessible=storage_ok,
        )

    # 4. Mount API routes at both root and /api/v1 prefix for maximum flexibility
    app.include_router(api_router, prefix="")
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
