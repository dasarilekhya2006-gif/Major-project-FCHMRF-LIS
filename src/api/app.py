"""FastAPI application factory and middleware configuration."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.logging import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Adaptive DL-Assisted fcHMRF-LIS",
        description="Spatial False Discovery Rate Control in Neuroimaging REST API",
        version="0.1.0",
    )

    # Enable CORS for local research UI development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST router
    app.include_router(router, prefix="/api/v1", tags=["Spatial-FDR"])

    # Mount frontend static directory if it exists
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info(f"Mounted frontend static directory from {frontend_dir}")

    return app


app = create_app()
