"""Security utilities for NIfTI validation, path sanitization, and traversal prevention."""

import gzip
import io
from pathlib import Path
import re
from typing import Optional, Tuple
from fastapi import HTTPException, status

from backend.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Valid experiment ID pattern: letters, numbers, underscores, dashes only
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")
SAFE_FILENAME_REGEX = re.compile(r"[^a-zA-Z0-9_.-]")


def validate_experiment_id(experiment_id: str) -> str:
    """Validate experiment identifier to prevent path traversal and injection attacks."""
    if not experiment_id or not SAFE_ID_REGEX.match(experiment_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_EXPERIMENT_ID",
                "message": f"Experiment ID '{experiment_id}' is invalid. Allowed: alphanumeric, hyphens, underscores (3-64 chars).",
            },
        )
    return experiment_id


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename by stripping directories, null bytes, and unsafe characters."""
    clean_name = Path(filename).name
    clean_name = clean_name.replace("\x00", "")
    clean_name = SAFE_FILENAME_REGEX.sub("_", clean_name)
    if not (clean_name.endswith(".nii") or clean_name.endswith(".nii.gz")):
        # Ensure it maintains valid nifti extension
        if ".nii" in clean_name:
            clean_name = clean_name.split(".nii")[0] + ".nii.gz"
        else:
            clean_name += ".nii.gz"
    return clean_name


def validate_nifti_buffer(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """Validate buffer for size limits, valid extensions, and NIfTI magic headers.

    Returns:
        (is_valid, error_message)
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return False, f"File size ({len(file_bytes) / 1024 / 1024:.1f} MB) exceeds maximum allowed limit ({settings.max_upload_size_mb} MB)."

    if len(file_bytes) < 352:
        return False, "File is too small to contain a valid NIfTI header (minimum 352 bytes)."

    is_gzipped = filename.endswith(".nii.gz") or file_bytes[:2] == b"\x1f\x8b"

    if is_gzipped:
        # Validate gzip structure by streaming only the first 352 decompressed bytes.
        # gzip.decompress() requires a complete stream, so we must use GzipFile on
        # the full buffer to read the header bytes without decompressing everything.
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(file_bytes)) as gz:
                decompressed_header = gz.read(352)
            if len(decompressed_header) >= 352:
                magic = decompressed_header[344:348]
                if magic not in (b"n+1\0", b"ni1\0", b"n+2\0", b"ni2\0"):
                    return False, f"Gzipped file does not contain valid NIfTI magic bytes (found: {magic})."
        except Exception as e:
            return False, f"Corrupted or invalid gzip stream: {e}"
    else:
        # Uncompressed NIfTI-1 / NIfTI-2 magic at offset 344
        magic = file_bytes[344:348]
        if magic not in (b"n+1\0", b"ni1\0", b"n+2\0", b"ni2\0"):
            return False, f"Invalid uncompressed NIfTI magic header (found: {magic})."

    return True, None
