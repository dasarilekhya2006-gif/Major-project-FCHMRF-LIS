"""Efron's Local False Discovery Rate (locfdr) and empirical null estimation."""

from typing import Optional, Tuple
import numpy as np
from scipy import stats, optimize

from src.core.logging import get_logger

logger = get_logger(__name__)


def estimate_empirical_null_central_matching(
    z_scores: np.ndarray,
    pct_interval: Tuple[float, float] = (25.0, 75.0),
) -> Tuple[float, float]:
    """Estimate empirical null mean (delta_0) and scale (sigma_0) via central matching.

    Args:
        z_scores: 1D array of z-statistics across voxels.
        pct_interval: Lower and upper percentiles containing predominantly null voxels.

    Returns:
        Tuple[float, float]: (delta_0, sigma_0)
    """
    q_low = float(np.percentile(z_scores, pct_interval[0]))
    q_high = float(np.percentile(z_scores, pct_interval[1]))

    # Median is robust estimate of center delta_0
    delta_0 = float(np.median(z_scores))

    # Scale from IQR: under normal, IQR = 1.349 * sigma
    iqr = q_high - q_low
    sigma_0 = float(iqr / 1.34898) if iqr > 1e-6 else 1.0

    logger.debug(f"Central matching empirical null: delta_0={delta_0:.4f}, sigma_0={sigma_0:.4f}")
    return delta_0, sigma_0


def efron_locfdr(
    z_scores: np.ndarray,
    threshold: float = 0.20,
    use_empirical_null: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Compute Efron's local false discovery rate: fdr(z) = pi_0 * f_0(z) / f(z).

    Args:
        z_scores: 1D array of z-statistics.
        threshold: Decision threshold for locfdr (commonly 0.20 for discovery).
        use_empirical_null: Whether to estimate (delta_0, sigma_0) empirically.

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
            - rejected: Boolean array indicating significant voxels (locfdr <= threshold).
            - locfdr_values: Array of local FDR values in [0, 1].
            - null_params: (delta_0, sigma_0) used.
    """
    z = np.asarray(z_scores, dtype=np.float64).ravel()
    m = len(z)

    if use_empirical_null:
        delta_0, sigma_0 = estimate_empirical_null_central_matching(z)
    else:
        delta_0, sigma_0 = 0.0, 1.0

    # Null density f_0(z)
    f0 = stats.norm.pdf(z, loc=delta_0, scale=sigma_0)

    # Estimate overall density f(z) using Gaussian KDE
    kde = stats.gaussian_kde(z)
    fz = kde(z)

    # Estimate pi_0
    pi0 = min(1.0, float(np.max(fz / np.maximum(f0, 1e-12))))
    if pi0 <= 0.0 or np.isnan(pi0):
        pi0 = 0.9

    # Local FDR: fdr(z) = pi_0 * f_0(z) / f(z)
    locfdr_vals = np.clip((pi0 * f0) / np.maximum(fz, 1e-12), 0.0, 1.0)
    rejected = locfdr_vals <= threshold

    logger.debug(
        f"locfdr (thresh={threshold}): rejected {np.sum(rejected)} / {m} voxels "
        f"(null: mu={delta_0:.2f}, s={sigma_0:.2f})"
    )
    return rejected, locfdr_vals, (delta_0, sigma_0)
