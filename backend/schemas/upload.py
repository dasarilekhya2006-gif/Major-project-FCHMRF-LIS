"""Upload response and file metadata schemas."""

from typing import List, Tuple
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned upon successful NIfTI file ingestion."""

    experiment_id: str = Field(..., description="Unique generated experiment ID")
    filename: str = Field(..., description="Sanitized client filename")
    status: str = Field(default="uploaded", description="Initial dataset lifecycle state")
    shape: List[int] = Field(..., description="Volumetric spatial dimensions [X, Y, Z]")
    voxel_spacing: List[float] = Field(..., description="Voxel dimensions in mm [dx, dy, dz]")
    orientation: str = Field(..., description="Radiological/anatomical orientation (e.g. RAS, LPI)")
