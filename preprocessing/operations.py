"""Neuroimaging preprocessing operations module (Phase 2).

Provides clean, configurable, and reusable functions for:
  - NaN/Inf handling and missing voxel imputation.
  - Intensity normalization (z-score, min-max, robust percentile scaling).
  - Optional spatial denoising (Gaussian, median, or none).
  - Optional spatial resampling (updating affine and spacing consistently).
  - Optional brain masking when a valid mask exists.
  - Full pipeline orchestration preserving affine, voxel spacing, spatial dimensions,
    and orientation unless explicitly configured.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.core.exceptions import PreprocessingError
from src.data.schemas import NeuroimagingVolume
from src.preprocessing.imputation import handle_missing_voxels
from src.preprocessing.normalization import normalize_intensities
from src.preprocessing.denoising import denoise_volume as _denoise_array
from src.preprocessing.resampling import resample_to_spacing
from src.preprocessing.masking import apply_mask, load_or_compute_mask
from src.preprocessing.schemas import (
    PreprocessingConfig,
    PreprocessingMetadata,
    PreprocessingStepRecord,
)
from src.preprocessing.pipeline import PreprocessedVolume


def handle_nan_inf(
    volume: NeuroimagingVolume,
    method: str = "zero",
    mask: Optional[np.ndarray] = None,
) -> NeuroimagingVolume:
    """Handle NaN and Infinite voxel values in a neuroimaging volume.

    Preserves affine matrix, voxel spacing, spatial dimensions, and orientation.

    Args:
        volume: Input NeuroimagingVolume from Phase 1.
        method: Imputation strategy: 'zero', 'mean', 'median', or 'nearest_finite'.
        mask: Optional boolean spatial mask to restrict imputation stats.

    Returns:
        NeuroimagingVolume: Cleaned volume with finite voxel intensities.
    """
    effective_mask = mask if mask is not None else volume.mask
    cleaned_data, _ = handle_missing_voxels(volume.data, mask=effective_mask, method=method)

    return NeuroimagingVolume(
        data=cleaned_data.astype(np.float32),
        affine=volume.affine.copy(),
        voxel_dimensions=volume.voxel_dimensions,
        mask=volume.mask.copy() if volume.mask is not None else None,
        subject_id=volume.subject_id,
        metadata=volume.metadata,
    )


def normalize_intensity(
    volume: NeuroimagingVolume,
    method: str = "z_score",
    mask: Optional[np.ndarray] = None,
    percentile_bounds: Tuple[float, float] = (1.0, 99.0),
) -> NeuroimagingVolume:
    """Normalize intensity distribution of a neuroimaging volume.

    Preserves affine matrix, voxel spacing, spatial dimensions, and orientation.

    Args:
        volume: Input NeuroimagingVolume from Phase 1.
        method: 'z_score' (standardized), 'min_max' ([0, 1] bounded), or 'robust'.
        mask: Optional boolean mask; if provided, normalization statistics are computed
              only over in-mask voxels.
        percentile_bounds: Lower and upper percentiles for outlier robust normalization.

    Returns:
        NeuroimagingVolume: Normalized volume with preserved geometry.
    """
    effective_mask = mask if mask is not None else volume.mask
    method_clean = method.lower().replace("-", "_")
    if method_clean in ("z_score", "standard"):
        method_clean = "zscore"
    elif method_clean in ("robust", "median_iqr"):
        method_clean = "robust_zscore"

    norm_data, _, _ = normalize_intensities(
        volume.data,
        mask=effective_mask,
        method=method_clean,
        percentile_bounds=percentile_bounds,
    )

    return NeuroimagingVolume(
        data=norm_data.astype(np.float32),
        affine=volume.affine.copy(),
        voxel_dimensions=volume.voxel_dimensions,
        mask=volume.mask.copy() if volume.mask is not None else None,
        subject_id=volume.subject_id,
        metadata=volume.metadata,
    )


def denoise_volume(
    volume: NeuroimagingVolume,
    method: str = "none",
    fwhm_mm: float = 4.0,
    median_kernel_size: int = 3,
    mask: Optional[np.ndarray] = None,
) -> NeuroimagingVolume:
    """Apply optional spatial denoising to a neuroimaging volume.

    Preserves affine matrix, voxel spacing, spatial dimensions, and orientation.
    If method is 'none', returns volume data unaltered.

    Args:
        volume: Input NeuroimagingVolume from Phase 1.
        method: 'none', 'gaussian', or 'median'.
        fwhm_mm: Full Width at Half Maximum in mm for Gaussian smoothing.
        median_kernel_size: Neighborhood kernel diameter for median filter.
        mask: Optional brain mask to confine filtering.

    Returns:
        NeuroimagingVolume: Denoised volume with preserved spatial metadata.
    """
    if method == "none" or method is None:
        return volume

    effective_mask = mask if mask is not None else volume.mask
    denoised_data = _denoise_array(
        volume.data,
        method=method,
        fwhm_mm=fwhm_mm,
        voxel_dims=volume.voxel_dimensions,
        median_kernel_size=median_kernel_size,
        mask=effective_mask,
    )

    return NeuroimagingVolume(
        data=denoised_data.astype(np.float32),
        affine=volume.affine.copy(),
        voxel_dimensions=volume.voxel_dimensions,
        mask=volume.mask.copy() if volume.mask is not None else None,
        subject_id=volume.subject_id,
        metadata=volume.metadata,
    )


def resample_volume(
    volume: NeuroimagingVolume,
    target_spacing: Tuple[float, float, float],
    interpolation_order: int = 1,
) -> NeuroimagingVolume:
    """Optionally resample volume to target isotropic/anisotropic voxel spacing.

    Updates the affine matrix and voxel_dimensions to reflect the new spatial grid
    with exact coordinate consistency.

    Args:
        volume: Input NeuroimagingVolume from Phase 1.
        target_spacing: Desired voxel dimensions in mm (dx, dy, dz).
        interpolation_order: Spline order (0=nearest, 1=trilinear, 3=cubic).

    Returns:
        NeuroimagingVolume: Resampled volume with updated affine and voxel spacing.
    """
    return resample_to_spacing(
        volume=volume,
        target_spacing=target_spacing,
        order=interpolation_order,
    )


def apply_brain_mask(
    volume: NeuroimagingVolume,
    mask: Optional[Union[np.ndarray, Path, str]] = None,
    background_value: float = 0.0,
) -> NeuroimagingVolume:
    """Apply brain mask when a valid mask exists.

    Does not automatically fabricate or apply an unjustified mask if none exists.
    Preserves affine matrix, voxel spacing, spatial dimensions, and orientation.

    Args:
        volume: Input NeuroimagingVolume from Phase 1.
        mask: Explicit binary mask array or path to mask NIfTI file.
              If None, uses volume.mask if already attached.
        background_value: Fill value for voxels outside the mask (default 0.0).

    Returns:
        NeuroimagingVolume: Masked volume (out-of-mask voxels zeroed), or original
                            volume unaltered if no valid mask exists.
    """
    resolved_mask: Optional[np.ndarray] = None
    if mask is not None:
        if isinstance(mask, (str, Path)):
            import nibabel as nib
            mask_img = nib.load(str(mask))
            resolved_mask = np.asarray(mask_img.get_fdata(), dtype=bool)
        elif isinstance(mask, np.ndarray):
            resolved_mask = mask.astype(bool)
    elif volume.mask is not None:
        resolved_mask = volume.mask.astype(bool)

    if resolved_mask is None:
        # No valid mask provided or attached; preserve original volume without alteration
        return volume

    # Verify spatial dimension alignment
    if resolved_mask.shape[:3] != volume.spatial_shape:
        raise PreprocessingError(
            f"Mask dimensions {resolved_mask.shape[:3]} do not match volume spatial dimensions {volume.spatial_shape}"
        )

    # Apply mask across 3D or 4D data
    masked_data = volume.data.copy()
    if volume.is_4d:
        for t in range(volume.num_timepoints):
            masked_data[~resolved_mask, t] = background_value
    else:
        masked_data[~resolved_mask] = background_value

    return NeuroimagingVolume(
        data=masked_data.astype(np.float32),
        affine=volume.affine.copy(),
        voxel_dimensions=volume.voxel_dimensions,
        mask=resolved_mask,
        subject_id=volume.subject_id,
        metadata=volume.metadata,
    )


def preprocess_volume(
    volume: NeuroimagingVolume,
    config: Optional[PreprocessingConfig] = None,
    mask: Optional[Union[np.ndarray, Path, str]] = None,
) -> PreprocessedVolume:
    """Execute clean, configurable neuroimaging preprocessing pipeline.

    Preserves affine matrix, voxel spacing, spatial dimensions, and orientation
    unless explicit resampling is enabled. Transformations are only applied
    when explicitly justified and configured.

    Order of operations:
      1. NaN/Inf handling
      2. Optional brain masking (when valid mask exists)
      3. Optional spatial denoising (default: none)
      4. Intensity normalization
      5. Optional resampling (if target_voxel_spacing configured)

    Args:
        volume: Input NeuroimagingVolume from Phase 1.
        config: PreprocessingConfig specifying active steps and parameters.
        mask: Optional explicit brain mask.

    Returns:
        PreprocessedVolume: Preprocessed volume containing attached execution audit metadata.
    """
    cfg = config or PreprocessingConfig(
        denoising_method="none",
        compute_mask_if_missing=False,
        pad_crop_mode="none",
    )

    current_vol = volume
    history = []

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        fin = arr[np.isfinite(arr)]
        if len(fin) == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min": float(np.min(fin)),
            "max": float(np.max(fin)),
            "mean": float(np.mean(fin)),
            "std": float(np.std(fin)),
        }

    # 1. NaN/Inf handling
    if np.isnan(current_vol.data).any() or np.isinf(current_vol.data).any():
        shape_before = current_vol.shape
        current_vol = handle_nan_inf(current_vol, method=cfg.imputation_method)
        history.append(
            PreprocessingStepRecord(
                step_name="imputation",
                parameters={"method": cfg.imputation_method},
                input_shape=shape_before,
                output_shape=current_vol.shape,
                summary_stats=_stats(current_vol.data),
            )
        )

    # 2. Optional brain masking (when a valid mask exists)
    if mask is not None or current_vol.mask is not None:
        shape_before = current_vol.shape
        current_vol = apply_brain_mask(current_vol, mask=mask)
        history.append(
            PreprocessingStepRecord(
                step_name="brain_masking",
                parameters={"applied": True},
                input_shape=shape_before,
                output_shape=current_vol.shape,
                summary_stats=_stats(current_vol.data),
            )
        )

    # 3. Optional denoising (only if configured != 'none')
    if cfg.denoising_method != "none" and cfg.denoising_method is not None:
        shape_before = current_vol.shape
        current_vol = denoise_volume(
            current_vol,
            method=cfg.denoising_method,
            fwhm_mm=cfg.gaussian_fwhm_mm,
            median_kernel_size=cfg.median_kernel_size,
        )
        history.append(
            PreprocessingStepRecord(
                step_name="denoising",
                parameters={"method": cfg.denoising_method, "fwhm_mm": cfg.gaussian_fwhm_mm},
                input_shape=shape_before,
                output_shape=current_vol.shape,
                summary_stats=_stats(current_vol.data),
            )
        )

    # 4. Intensity normalization
    if cfg.normalization_method != "none":
        shape_before = current_vol.shape
        current_vol = normalize_intensity(
            current_vol,
            method=cfg.normalization_method,
            percentile_bounds=cfg.percentile_bounds,
        )
        history.append(
            PreprocessingStepRecord(
                step_name="intensity_normalization",
                parameters={"method": cfg.normalization_method},
                input_shape=shape_before,
                output_shape=current_vol.shape,
                summary_stats=_stats(current_vol.data),
            )
        )

    # 5. Optional resampling (only if explicitly specified)
    if cfg.target_voxel_spacing is not None:
        shape_before = current_vol.shape
        current_vol = resample_volume(current_vol, target_spacing=cfg.target_voxel_spacing)
        history.append(
            PreprocessingStepRecord(
                step_name="spatial_resampling",
                parameters={"target_voxel_spacing": list(cfg.target_voxel_spacing)},
                input_shape=shape_before,
                output_shape=current_vol.shape,
                summary_stats=_stats(current_vol.data),
            )
        )

    # Compile audit metadata
    audit_meta = PreprocessingMetadata(
        original_shape=volume.shape,
        original_affine=volume.affine.copy(),
        original_voxel_dimensions=volume.voxel_dimensions,
        preprocessed_shape=current_vol.shape,
        preprocessed_affine=current_vol.affine.copy(),
        preprocessed_voxel_dimensions=current_vol.voxel_dimensions,
        history=history,
    )

    return PreprocessedVolume(
        data=current_vol.data,
        affine=current_vol.affine,
        voxel_dimensions=current_vol.voxel_dimensions,
        mask=current_vol.mask,
        subject_id=current_vol.subject_id,
        metadata=current_vol.metadata,
        preprocessing_metadata=audit_meta,
    )
