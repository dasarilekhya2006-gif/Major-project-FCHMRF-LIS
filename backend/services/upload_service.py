"""Upload and initial file validation service."""

from pathlib import Path
import tempfile
import uuid
from typing import Tuple
from fastapi import HTTPException, UploadFile, status
import nibabel as nib
import numpy as np

from backend.core.security import sanitize_filename, validate_nifti_buffer
from backend.schemas.upload import UploadResponse
from backend.services.storage_service import storage_service
from src.core.logging import get_logger

logger = get_logger(__name__)


class UploadService:
    """Handles secure NIfTI file validation, extraction of metadata, and isolated sandbox storage."""

    async def handle_upload(self, file: UploadFile) -> UploadResponse:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "MISSING_FILENAME", "message": "Uploaded file must have a valid filename."},
            )

        clean_name = sanitize_filename(file.filename)
        contents = await file.read()

        # 1. Byte-level security and magic bytes validation
        is_valid, err_msg = validate_nifti_buffer(contents, clean_name)
        if not is_valid:
            logger.warning(f"Upload rejected for '{file.filename}': {err_msg}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "INVALID_NIFTI", "message": err_msg or "Failed NIfTI structure validation."},
            )

        # 2. Structural parsing with NiBabel to ensure valid NIfTI header and array
        suffix = ".nii.gz" if clean_name.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            try:
                img = nib.load(str(tmp_path))
                shape = list(img.shape)
                header = img.header
                zooms = [float(z) for z in header.get_zooms()[:3]]
                affine = img.affine
                # Anatomical orientation string (e.g. RAS, LPI)
                ornt = "".join(nib.aff2axcodes(affine))
            except Exception as e:
                logger.error(f"NiBabel failed to parse uploaded volume: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "CORRUPT_NIFTI",
                        "message": "The uploaded file could not be parsed as a valid NIfTI-1/2 volume.",
                    },
                )

            # Ensure minimum 3D dimensions
            if len(shape) < 3 or any(s < 4 for s in shape[:3]):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "INVALID_DIMENSIONS",
                        "message": f"Volume dimensions {shape} are too small or invalid for neuroimaging analysis.",
                    },
                )

            # 3. Generate unique experiment ID and persist to sandboxed storage
            exp_id = f"exp_{uuid.uuid4().hex[:12]}"
            initial_meta = {
                "shape": shape[:3],
                "voxel_spacing": zooms,
                "orientation": ornt,
                "affine": affine.tolist(),
            }
            storage_service.create_experiment(exp_id, clean_name, initial_meta=initial_meta)

            # Load actual data array and save in standard compressed format (.nii.gz)
            data_arr = np.asanyarray(img.dataobj)
            if data_arr.ndim == 4:
                data_arr = np.mean(data_arr, axis=-1)

            storage_service.save_stage_volume(
                experiment_id=exp_id,
                stage="input",
                filename="raw_volume.nii.gz",
                data=data_arr.astype(np.float32),
                affine=affine,
            )

            logger.info(f"Successfully ingested NIfTI '{clean_name}' as experiment '{exp_id}' with shape {shape[:3]}")

            return UploadResponse(
                experiment_id=exp_id,
                filename=clean_name,
                status="uploaded",
                shape=shape[:3],
                voxel_spacing=zooms,
                orientation=ornt,
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


upload_service = UploadService()
