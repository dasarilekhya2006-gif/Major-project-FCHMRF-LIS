"""Normalization and standardization utilities for statistical neuroimaging maps."""

from typing import Optional, Tuple
import numpy as np
from scipy import stats

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def normalize_intensities(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "zscore",
    percentile_bounds: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[np.ndarray, float, float]:
    """Normalize intensity or statistical values across valid voxels.

    Supported methods:
      - 'zscore': Standard score: (x - mean) / std.
      - 'robust_zscore' (or 'median_iqr'): Median and IQR-based scaling: (x - median) / (IQR * 0.7413).
      - 'min_max': Linear rescaling of in-mask voxels to [0, 1].
      - 'percentile_clip': Robust clipping at given percentiles, followed by unit variance scaling.
      - 'none': Identity transformation (leaves values untouched).

    Args:
        data: NumPy ndarray of intensities or statistical values.
        mask: Optional boolean spatial mask defining valid voxels.
        method: Normalization strategy.
        percentile_bounds: Tuple of (lower_percentile, upper_percentile) for clipping.

    Returns:
        Tuple[np.ndarray, float, float]: (normalized_array, center, scale)

    Raises:
        PreprocessingError: If method is unsupported or no valid voxels are present.
    """
    valid_data = data[mask] if mask is not None else data.ravel()
    valid_data = valid_data[np.isfinite(valid_data)]

    if len(valid_data) == 0:
        raise PreprocessingError("No valid finite voxels available for normalization.")

    norm_data = data.copy()

    if method in ("zscore", "standard"):
        center = float(np.mean(valid_data))
        scale = float(np.std(valid_data))
        if scale < 1e-8:
            logger.warning("Voxel standard deviation is near zero; retaining scale=1.0.")
            scale = 1.0
        norm_data = (norm_data - center) / scale

    elif method in ("robust_zscore", "median_iqr"):
        center = float(np.median(valid_data))
        iqr = float(stats.iqr(valid_data))
        scale = iqr * 0.7413 if iqr > 1e-8 else 1.0
        norm_data = (norm_data - center) / scale

    elif method == "min_max":
        center = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        scale = max_val - center
        if scale < 1e-8:
            scale = 1.0
        norm_data = (norm_data - center) / scale

    elif method == "percentile_clip":
        p_low = float(np.percentile(valid_data, percentile_bounds[0]))
        p_high = float(np.percentile(valid_data, percentile_bounds[1]))
        clipped = np.clip(valid_data, p_low, p_high)
        center = float(np.mean(clipped))
        scale = float(np.std(clipped))
        if scale < 1e-8:
            scale = 1.0
        norm_data = np.clip(norm_data, p_low, p_high)
        norm_data = (norm_data - center) / scale

    elif method == "none":
        return norm_data, 0.0, 1.0

    else:
        raise PreprocessingError(
            f"Unsupported normalization method: '{method}'. "
            f"Supported methods: 'zscore', 'robust_zscore', 'min_max', 'percentile_clip', 'none'."
        )

    # If mask is provided, keep out-of-mask voxels strictly zero
    if mask is not None:
        norm_data[~mask] = 0.0

    logger.debug(f"Normalized intensities with method='{method}': center={center:.4f}, scale={scale:.4f}")
    return norm_data.astype(np.float32), center, scale


def denormalize_intensities(
    normalized_data: np.ndarray,
    center: float,
    scale: float,
) -> np.ndarray:
    """Invert normalization back to the original intensity scale."""
    return (normalized_data * scale) + center


# Backward-compatible alias
def standardize_zscores(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "zscore",
) -> Tuple[np.ndarray, float, float]:
    """Backward-compatible wrapper for normalize_intensities."""
    return normalize_intensities(data=data, mask=mask, method=method)


def convert_pvalues_to_zscores(
    p_values: np.ndarray,
    two_tailed: bool = False,
    eps: float = 1e-15,
) -> np.ndarray:
    """Transform p-values to standard normal z-scores using the inverse normal CDF.

    Args:
        p_values: Array of p-values in (0, 1].
        two_tailed: If True, uses two-tailed quantile conversion.
        eps: Small epsilon to prevent infinite z-scores for p=0 or p=1.

    Returns:
        np.ndarray: Array of corresponding z-scores.
    """
    clipped = np.clip(p_values, eps, 1.0 - eps)
    if two_tailed:
        # z = Phi^{-1}(1 - p / 2)
        z_scores = stats.norm.ppf(1.0 - clipped / 2.0)
    else:
        # One-tailed upper tail: z = Phi^{-1}(1 - p)
        z_scores = stats.norm.ppf(1.0 - clipped)
    return z_scores
