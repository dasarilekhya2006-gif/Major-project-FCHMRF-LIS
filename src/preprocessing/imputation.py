"""Missing and non-finite voxel imputation module for neuroimaging data."""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def handle_missing_voxels(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "zero",
) -> Tuple[np.ndarray, int]:
    """Identify and impute non-finite (NaN, Inf, -Inf) voxels in a neuroimaging volume.

    Supported methods:
      - 'zero': Replace non-finite voxels with 0.0 (safest for background air/null fields).
      - 'mean': Replace non-finite voxels with the arithmetic mean of valid in-mask voxels.
      - 'nearest_finite': Replace each non-finite voxel with the intensity of its nearest
        finite neighbor using 3D Euclidean distance transforms.

    Args:
        data: 3D (or 4D) numpy array containing image or statistical intensities.
        mask: Optional boolean brain mask defining valid spatial domain.
        method: Imputation strategy ('zero', 'mean', 'nearest_finite').

    Returns:
        Tuple[np.ndarray, int]: Imputed data array (copy) and count of imputed voxels.

    Raises:
        PreprocessingError: If method is unknown or all voxels are non-finite.
    """
    cleaned = data.copy()
    invalid_mask = ~np.isfinite(cleaned)
    num_invalid = int(np.sum(invalid_mask))

    if num_invalid == 0:
        logger.debug("No non-finite voxels detected; data is intact.")
        return cleaned, 0

    logger.info(f"Handling {num_invalid} / {data.size} non-finite voxels using method='{method}'")

    if method == "zero":
        cleaned[invalid_mask] = 0.0

    elif method == "mean":
        valid_mask = ~invalid_mask
        if mask is not None:
            valid_mask = valid_mask & mask

        if not np.any(valid_mask):
            raise PreprocessingError("Cannot compute mean imputation: no valid finite voxels exist.")

        fill_val = float(np.mean(cleaned[valid_mask]))
        cleaned[invalid_mask] = fill_val

    elif method == "nearest_finite":
        if np.all(invalid_mask):
            raise PreprocessingError("Cannot perform nearest-finite imputation: entire volume is non-finite.")

        if cleaned.ndim == 3:
            # Use distance transform to find indices of nearest valid voxels
            valid_indicators = (~invalid_mask).astype(bool)
            _, indices = distance_transform_edt(~valid_indicators, return_indices=True)
            cleaned = cleaned[tuple(indices)]
        else:
            # For 4D, perform nearest finite interpolation slice-by-slice across time
            for t in range(cleaned.shape[-1]):
                vol_t = cleaned[..., t]
                inv_t = ~np.isfinite(vol_t)
                if np.any(inv_t):
                    valid_t = (~inv_t).astype(bool)
                    if not np.any(valid_t):
                        vol_t[inv_t] = 0.0
                    else:
                        _, indices = distance_transform_edt(~valid_t, return_indices=True)
                        vol_t = vol_t[tuple(indices)]
                    cleaned[..., t] = vol_t

    else:
        raise PreprocessingError(f"Unsupported imputation method: '{method}'. Choose 'zero', 'mean', or 'nearest_finite'.")

    return cleaned, num_invalid
