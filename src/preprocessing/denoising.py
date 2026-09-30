"""Optional spatial denoising filters for neuroimaging data."""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def denoise_volume(
    data: np.ndarray,
    method: str = "none",
    fwhm_mm: float = 4.0,
    voxel_dims: Tuple[float, ...] = (2.0, 2.0, 2.0),
    median_kernel_size: int = 3,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply spatial denoising to a 3D or 4D neuroimaging volume.

    Methods:
      - 'none': Passthrough without modification (preserves fine activation topology).
      - 'gaussian': 3D Gaussian spatial filtering with boundary-preserving normalized convolution.
      - 'median': 3D rank-order median filter (removes impulse artifacts while preserving sharp edges).

    Args:
        data: 3D (or 4D) array of voxel values.
        method: Denoising strategy ('none', 'gaussian', 'median').
        fwhm_mm: Gaussian filter Full Width at Half Maximum in mm.
        voxel_dims: Physical voxel dimensions in mm (dx, dy, dz).
        median_kernel_size: Size of the median filter window (e.g. 3).
        mask: Optional boolean brain mask to restrict filtering and eliminate boundary leakage.

    Returns:
        np.ndarray: Denoised array matching input shape.

    Raises:
        PreprocessingError: If method is unsupported.
    """
    if method == "none":
        return data.copy()

    logger.info(f"Applying spatial denoising using method='{method}'")

    if method == "gaussian":
        if fwhm_mm <= 0.0:
            return data.copy()

        fwhm_to_sigma = 1.0 / np.sqrt(8.0 * np.log(2.0))
        sigmas = [
            (fwhm_mm / float(voxel_dims[i])) * fwhm_to_sigma
            for i in range(3)
        ]

        if data.ndim == 3:
            return _gaussian_denoise_3d(data, sigmas, mask)
        elif data.ndim == 4:
            out_4d = np.zeros_like(data)
            for t in range(data.shape[3]):
                out_4d[..., t] = _gaussian_denoise_3d(data[..., t], sigmas, mask)
            return out_4d
        else:
            raise PreprocessingError(f"Gaussian denoising expects 3D or 4D array, received ndim={data.ndim}")

    elif method == "median":
        if median_kernel_size <= 1:
            return data.copy()

        size = (median_kernel_size, median_kernel_size, median_kernel_size)
        if data.ndim == 3:
            filtered = median_filter(data, size=size)
            if mask is not None:
                filtered[~mask] = 0.0
            return filtered.astype(np.float32)
        elif data.ndim == 4:
            out_4d = np.zeros_like(data)
            for t in range(data.shape[3]):
                filtered_t = median_filter(data[..., t], size=size)
                if mask is not None:
                    filtered_t[~mask] = 0.0
                out_4d[..., t] = filtered_t
            return out_4d.astype(np.float32)
        else:
            raise PreprocessingError(f"Median denoising expects 3D or 4D array, received ndim={data.ndim}")

    else:
        raise PreprocessingError(
            f"Unsupported denoising method: '{method}'. Choose 'none', 'gaussian', or 'median'."
        )


def _gaussian_denoise_3d(
    data: np.ndarray,
    sigmas: list,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Helper for 3D Gaussian denoising with boundary-preserving normalized convolution."""
    if mask is not None:
        data_masked = np.where(mask, data, 0.0)
        weights = mask.astype(np.float32)

        smoothed_data = gaussian_filter(data_masked, sigma=sigmas)
        smoothed_weights = gaussian_filter(weights, sigma=sigmas)

        valid = smoothed_weights > 1e-4
        out = np.zeros_like(data, dtype=np.float32)
        out[valid] = (smoothed_data[valid] / smoothed_weights[valid]).astype(np.float32)
        out[~mask] = 0.0
        return out

    return gaussian_filter(data, sigma=sigmas).astype(np.float32)
