"""API routers aggregator for FastAPI backend."""

from fastapi import APIRouter

from backend.api.upload import router as upload_router
from backend.api.preprocess import router as preprocess_router
from backend.api.segmentation import router as segmentation_router
from backend.api.statistics import router as statistics_router
from backend.api.fchmrf import router as fchmrf_router
from backend.api.pipeline import router as pipeline_router
from backend.api.results import router as results_router
from backend.api.visualization import router as visualization_router

api_router = APIRouter()

# Register modular sub-routers
api_router.include_router(upload_router)
api_router.include_router(preprocess_router)
api_router.include_router(segmentation_router)
api_router.include_router(statistics_router)
api_router.include_router(fchmrf_router)
api_router.include_router(pipeline_router)
api_router.include_router(results_router)
api_router.include_router(visualization_router)

__all__ = ["api_router"]
