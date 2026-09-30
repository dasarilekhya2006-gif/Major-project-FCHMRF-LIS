"""Neuroimaging preprocessing package (Phase 2).

Provides clean, modular, and configurable preprocessing routines:
  - handle_nan_inf: NaN and Infinite voxel handling and imputation.
  - normalize_intensity: Voxel intensity standardization and scaling.
  - denoise_volume: Optional spatial Gaussian or median filtering.
  - resample_volume: Optional spatial grid resampling with affine preservation.
  - apply_brain_mask: Optional brain masking when a valid mask exists.
  - preprocess_volume: Configurable pipeline orchestration preserving geometry.
"""

from preprocessing.operations import (
    handle_nan_inf,
    normalize_intensity,
    denoise_volume,
    resample_volume,
    apply_brain_mask,
    preprocess_volume,
    PreprocessedVolume,
)
from src.preprocessing.schemas import (
    PreprocessingConfig,
    PreprocessingMetadata,
    PreprocessingStepRecord,
)

__all__ = [
    "handle_nan_inf",
    "normalize_intensity",
    "denoise_volume",
    "resample_volume",
    "apply_brain_mask",
    "preprocess_volume",
    "PreprocessedVolume",
    "PreprocessingConfig",
    "PreprocessingMetadata",
    "PreprocessingStepRecord",
]
