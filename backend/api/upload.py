"""Upload API route for NIfTI neuroimaging volume ingestion."""

from fastapi import APIRouter, File, UploadFile, status
from backend.schemas.upload import UploadResponse
from backend.schemas.common import ErrorResponse
from backend.services.upload_service import upload_service

router = APIRouter(tags=["Upload"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid filename or request"},
        422: {"model": ErrorResponse, "description": "Corrupted or invalid NIfTI file"},
    },
    summary="Upload MRI/fMRI neuroimaging volume",
    description="Validates NIfTI structure and headers, generates a unique experiment ID, and isolates storage.",
)
async def upload_nifti_file(file: UploadFile = File(..., description="Neuroimaging volume (.nii or .nii.gz)")) -> UploadResponse:
    return await upload_service.handle_upload(file)
