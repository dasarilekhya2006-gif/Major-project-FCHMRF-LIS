"""Spatial resampling module for isotropic voxel alignment and coordinate preservation."""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import zoom

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger
from src.data.schemas import NeuroimagingVolume

logger = get_logger(__name__)


def resample_to_spacing(
    volume: NeuroimagingVolume,
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    order: int = 1,
) -> NeuroimagingVolume:
    """Resample a 3D or 4D NeuroimagingVolume to a target physical voxel spacing (in mm).

    Preserves world-space coordinate alignment by mathematically updating the 4x4 affine matrix:
        A_new = A_old @ diag(target_spacing / orig_spacing, 1)

    Args:
        volume: Input NeuroimagingVolume.
        target_spacing: Target (dx, dy, dz) physical voxel resolution in millimeters.
        order: Spline interpolation order (0: nearest-neighbor, 1: trilinear, 3: cubic spline).

    Returns:
        NeuroimagingVolume: Resampled volume with updated data, affine, and zooms.

    Raises:
        PreprocessingError: If target_spacing is invalid.
    """
    for s in target_spacing:
        if s <= 0.0 or not np.isfinite(s):
            raise PreprocessingError(f"Target voxel spacing must be positive and finite, received {target_spacing}")

    orig_spacing = volume.voxel_dimensions[:3]
    if len(orig_spacing) < 3:
        raise PreprocessingError(f"Volume must have at least 3 spatial voxel dimensions, found {orig_spacing}")

    # Check if resampling is needed
    if np.allclose(orig_spacing, target_spacing, atol=1e-3):
        logger.debug(f"Volume already matches target spacing {target_spacing} mm; skipping resampling.")
        return volume

    # Zoom factors along spatial axes: z_j = orig_spacing_j / target_spacing_j
    zoom_factors = [orig_spacing[j] / float(target_spacing[j]) for j in range(3)]
    logger.info(
        f"Resampling volume from {orig_spacing}mm to {target_spacing}mm (zoom factors: {zoom_factors})"
    )

    if volume.is_3d:
        resampled_data = zoom(volume.data, zoom=zoom_factors, order=order, prefilter=(order > 1))
    elif volume.is_4d:
        full_zoom = (*zoom_factors, 1.0)
        resampled_data = zoom(volume.data, zoom=full_zoom, order=order, prefilter=(order > 1))
    else:
        raise PreprocessingError(f"Expected 3D or 4D volume, got shape {volume.shape}")

    # Resample binary mask using nearest-neighbor interpolation if present
    resampled_mask: Optional[np.ndarray] = None
    if volume.mask is not None:
        mask_float = zoom(volume.mask.astype(np.float32), zoom=zoom_factors, order=0)
        resampled_mask = mask_float > 0.5

    # Update affine matrix: scale directional columns by (target_spacing / orig_spacing)
    # Since index_old = index_new * (target / orig) = index_new / zoom_factor:
    # A_new[:, j] = A_old[:, j] * (target_spacing[j] / orig_spacing[j])
    scale_matrix = np.eye(4, dtype=np.float64)
    for j in range(3):
        scale_matrix[j, j] = float(target_spacing[j]) / float(orig_spacing[j])

    new_affine = volume.affine @ scale_matrix

    new_zooms = target_spacing if volume.is_3d else (*target_spacing, volume.voxel_dimensions[3])

    logger.debug(f"Resampled volume shape from {volume.shape} to {resampled_data.shape}")

    return NeuroimagingVolume(
        data=resampled_data.astype(np.float32),
        affine=new_affine,
        voxel_dimensions=new_zooms,
        mask=resampled_mask,
        subject_id=volume.subject_id,
        metadata=volume.metadata,
    )
