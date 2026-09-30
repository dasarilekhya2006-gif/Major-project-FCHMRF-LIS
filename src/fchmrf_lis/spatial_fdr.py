"""Spatial FDR control procedure using ranked Local Index of Significance (LIS)."""

from typing import Tuple
import numpy as np

from src.core.exceptions import StatisticalInferenceError
from src.core.logging import get_logger

logger = get_logger(__name__)


def control_spatial_fdr_lis(
    lis_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, float, float]:
    """Execute the optimal step-up spatial FDR controlling procedure (Sun & Cai 2009).

    Algorithm:
      1. Sort LIS values in ascending order:
         LIS_(1) <= LIS_(2) <= ... <= LIS_(m)
      2. Find cutoff index:
         k = max { j in {1, ..., m} : (1 / j) * sum_{i=1}^j LIS_(i) <= alpha }
      3. Reject all hypotheses with LIS_i <= LIS_(k).
         If no such j exists, reject nothing.

    This procedure guarantees FDR <= alpha while minimizing the False Non-Discovery Rate (FNDR).

    Args:
        lis_values: 1D array of LIS values in [0, 1].
        alpha: Nominal False Discovery Rate level (e.g. 0.05).

    Returns:
        Tuple[np.ndarray, float, float]:
            - rejected: 1D boolean array indicating rejected null hypotheses (discoveries).
            - lis_threshold: The maximum LIS value rejected (0.0 if none rejected).
            - estimated_fdr: Actual average LIS among the rejected voxels (empirical FDR estimate).
    """
    lis = np.asarray(lis_values, dtype=np.float64).ravel()
    m = len(lis)

    if m == 0:
        raise StatisticalInferenceError("Empty LIS array provided.")

    sorted_indices = np.argsort(lis)
    sorted_lis = lis[sorted_indices]

    # Cumulative mean of sorted LIS: (1 / j) * sum_{i=1}^j LIS_(i)
    cumulative_sums = np.cumsum(sorted_lis)
    ranks = np.arange(1, m + 1)
    cumulative_fdr = cumulative_sums / ranks

    # Find maximum index where cumulative FDR <= alpha
    eligible = cumulative_fdr <= alpha
    rejected = np.zeros(m, dtype=bool)

    if np.any(eligible):
        k_idx = int(np.where(eligible)[0][-1])
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[: k_idx + 1] = True
        rejected[sorted_indices] = rejected_sorted
        lis_threshold = float(sorted_lis[k_idx])
        estimated_fdr = float(cumulative_fdr[k_idx])
    else:
        lis_threshold = 0.0
        estimated_fdr = 0.0

    logger.info(
        f"Spatial FDR control (alpha={alpha}): rejected {np.sum(rejected)} / {m} voxels "
        f"({100 * np.mean(rejected):.2f}%), cutoff LIS={lis_threshold:.4f}, est_FDR={estimated_fdr:.4f}"
    )

    return rejected, lis_threshold, estimated_fdr


# Backward-compatible alias
control_spatial_fdr = control_spatial_fdr_lis

__all__ = ["control_spatial_fdr_lis", "control_spatial_fdr"]

