"""Spatial brain masking and indexing utilities for neuroimaging volumes."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def compute_brain_mask(
    volume: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 10.0,
    fill_holes: bool = True,
) -> np.ndarray:
    """Compute a binary brain mask from a neuroimaging volume using intensity thresholding.

    Args:
        volume: 3D or 4D numpy array of image or statistical values.
        threshold: Absolute threshold value. If None, computed from the given percentile.
        percentile: Percentile of non-zero values to use as lower cutoff.
        fill_holes: Whether to apply morphological hole filling.

    Returns:
        np.ndarray: 3D boolean mask matching spatial shape.
    """
    if volume.ndim not in (3, 4):
        raise PreprocessingError(f"Expected 3D or 4D volume, got {volume.shape}")

    # If 4D, compute summary spatial field across time
    spatial_data = np.mean(volume, axis=-1) if volume.ndim == 4 else volume

    if threshold is None:
        finite_vals = spatial_data[np.isfinite(spatial_data)]
        non_zeros = finite_vals[finite_vals > 0]
        if len(non_zeros) == 0:
            logger.warning("Volume contains no positive values; returning full mask.")
            return np.ones(spatial_data.shape, dtype=bool)
        threshold = float(np.percentile(non_zeros, percentile))

    binary = spatial_data > threshold
    if fill_holes:
        mask = binary_fill_holes(binary)
    else:
        mask = binary

    logger.debug(f"Computed brain mask: {np.sum(mask)} / {mask.size} voxels in mask.")
    return mask.astype(bool)


def load_or_compute_mask(
    volume_data: np.ndarray,
    mask_path: Optional[Union[str, Path]] = None,
    percentile: float = 10.0,
    fill_holes: bool = True,
) -> np.ndarray:
    """Load an existing brain mask from disk or compute one dynamically if missing.

    Args:
        volume_data: 3D or 4D numpy array.
        mask_path: Optional path to a mask NIfTI file.
        percentile: Threshold percentile for fallback computation.
        fill_holes: Whether to fill internal cavities.

    Returns:
        np.ndarray: 3D boolean mask array.
    """
    spatial_shape = volume_data.shape[:3]

    if mask_path is not None:
        p = Path(mask_path)
        if p.exists():
            try:
                import nibabel as nib
                mask_img = nib.load(str(p))
                mask_arr = np.asarray(mask_img.get_fdata(), dtype=bool)
                if mask_arr.shape[:3] != spatial_shape:
                    raise PreprocessingError(
                        f"Mask shape {mask_arr.shape[:3]} does not match volume shape {spatial_shape}"
                    )
                logger.info(f"Loaded existing brain mask from {p} with {np.sum(mask_arr)} voxels.")
                return mask_arr
            except Exception as e:
                logger.warning(f"Failed to load mask from {p}: {e}. Falling back to dynamic computation.")

    return compute_brain_mask(volume_data, percentile=percentile, fill_holes=fill_holes)


def apply_mask(
    volume: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Extract a 1D vector of voxel values located inside the mask.

    Args:
        volume: 3D numpy array.
        mask: 3D boolean numpy array.

    Returns:
        np.ndarray: 1D array of extracted values of length N_mask.
    """
    if volume.shape[:3] != mask.shape[:3]:
        raise PreprocessingError(
            f"Shape mismatch: volume {volume.shape[:3]} vs mask {mask.shape[:3]}"
        )
    return volume[mask]


def unmask(
    values_1d: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Reconstruct a 3D volume from a 1D vector of masked values.

    Args:
        values_1d: 1D numpy array of length equal to number of True elements in mask.
        mask: 3D boolean numpy array.
        fill_value: Value to assign to out-of-mask voxels.

    Returns:
        np.ndarray: 3D volume matching the shape of mask.
    """
    if values_1d.ndim != 1:
        raise PreprocessingError(f"Expected 1D vector, got shape {values_1d.shape}")
    num_mask_voxels = int(np.sum(mask))
    if values_1d.shape[0] != num_mask_voxels:
        raise PreprocessingError(
            f"Vector length {len(values_1d)} does not match mask voxel count {num_mask_voxels}"
        )

    out = np.full(mask.shape, fill_value, dtype=values_1d.dtype)
    out[mask] = values_1d
    return out


def extract_bounding_box(
    mask: np.ndarray,
    margin_voxels: int = 2,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Compute the spatial bounding box enclosing all in-mask voxels.

    Args:
        mask: 3D boolean numpy array.
        margin_voxels: Safety margin voxels to expand bounding box on each side.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
            ((x_min, x_max), (y_min, y_max), (z_min, z_max)) inclusive intervals.
    """
    if not np.any(mask):
        return (0, mask.shape[0]), (0, mask.shape[1]), (0, mask.shape[2])

    coords = np.argwhere(mask)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1

    x_min = max(0, x_min - margin_voxels)
    y_min = max(0, y_min - margin_voxels)
    z_min = max(0, z_min - margin_voxels)

    x_max = min(mask.shape[0], x_max + margin_voxels)
    y_max = min(mask.shape[1], y_max + margin_voxels)
    z_max = min(mask.shape[2], z_max + margin_voxels)

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)
