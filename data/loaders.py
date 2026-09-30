"""Top-level neuroimaging data loading and validation interface.

Provides primary reusable functions:
  - load_nifti: Load and normalize 3D or 4D NIfTI neuroimaging files.
  - validate_nifti: Validate structural, spatial, and numerical integrity.
  - get_nifti_metadata: Extract spatial dimensions, zooms, affine, and orientation.
"""

from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from src.data.loaders import (
    load_nifti,
    validate_image as validate_nifti,
    get_image_metadata as get_nifti_metadata,
    load_numpy_volume,
    save_nifti,
)
from src.data.schemas import (
    NeuroimagingMetadata,
    NeuroimagingVolume,
    ValidationReport,
)

__all__ = [
    "load_nifti",
    "validate_nifti",
    "get_nifti_metadata",
    "load_numpy_volume",
    "save_nifti",
    "NeuroimagingMetadata",
    "NeuroimagingVolume",
    "ValidationReport",
]
