"""Spatial Gaussian smoothing utilities for neuroimaging volumes."""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from src.core.logging import get_logger

logger = get_logger(__name__)


def smooth_volume(
    data: np.ndarray,
    fwhm_mm: float = 4.0,
    voxel_dims: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply 3D Gaussian spatial smoothing with specified FWHM in physical millimeters.

    Converts FWHM in mm to standard deviation sigma in voxels along each spatial axis:
        sigma_axis = (fwhm_mm / voxel_dim_axis) / sqrt(8 * ln(2))

    Args:
        data: 3D numpy array of values to smooth.
        fwhm_mm: Full Width at Half Maximum in millimeters (default: 4.0).
        voxel_dims: (dx, dy, dz) physical voxel spacing in mm.
        mask: Optional mask to restrict smoothing or re-apply boundary preservation.

    Returns:
        np.ndarray: Smoothed 3D numpy array.
    """
    if fwhm_mm <= 0.0:
        return data.copy()

    fwhm_to_sigma = 1.0 / np.sqrt(8.0 * np.log(2.0))
    sigmas = [
        (fwhm_mm / float(voxel_dims[i])) * fwhm_to_sigma
        for i in range(3)
    ]

    logger.debug(f"Applying Gaussian smoothing with FWHM={fwhm_mm}mm -> sigmas={sigmas} voxels")

    if mask is not None:
        # Normalized convolution to handle boundary effects
        data_masked = np.where(mask, data, 0.0)
        weights = mask.astype(np.float32)

        smoothed_data = gaussian_filter(data_masked, sigma=sigmas)
        smoothed_weights = gaussian_filter(weights, sigma=sigmas)

        valid = smoothed_weights > 1e-4
        out = np.zeros_like(data)
        out[valid] = smoothed_data[valid] / smoothed_weights[valid]
        out[~mask] = 0.0
        return out

    return gaussian_filter(data, sigma=sigmas)
