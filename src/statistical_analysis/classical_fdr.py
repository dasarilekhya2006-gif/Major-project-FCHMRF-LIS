"""Classical False Discovery Rate (FDR) controlling procedures."""

from typing import Tuple
import numpy as np

from src.core.exceptions import StatisticalInferenceError
from src.core.logging import get_logger

logger = get_logger(__name__)


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute the Benjamini-Hochberg (BH) step-up FDR procedure.

    Finds k = max { i : p_(i) <= (i / m) * alpha }.
    Rejects H_0 for all hypotheses with p_(i) <= p_(k).

    Args:
        p_values: 1D array of p-values in (0, 1].
        alpha: Nominal FDR control level (default: 0.05).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - rejected: 1D boolean array indicating rejected null hypotheses.
            - adjusted_pvalues: BH-adjusted p-values (q-values).
            - p_threshold: The maximum p-value threshold cut-off (0.0 if none rejected).
    """
    p_vals = np.asarray(p_values, dtype=np.float64).ravel()
    m = len(p_vals)
    if m == 0:
        raise StatisticalInferenceError("Empty p-values array provided.")

    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]

    # Threshold vector: (i / m) * alpha for i = 1, ..., m
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha

    below_threshold = sorted_p <= thresholds
    rejected_sorted = np.zeros(m, dtype=bool)

    if np.any(below_threshold):
        max_k_idx = int(np.where(below_threshold)[0][-1])
        rejected_sorted[: max_k_idx + 1] = True
        p_threshold = float(sorted_p[max_k_idx])
    else:
        p_threshold = 0.0

    # Unsort rejection array back to original input ordering
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_indices] = rejected_sorted

    # Compute step-up adjusted p-values
    # q_(i) = min_{j >= i} { (m / j) * p_(j) }
    adj_p = np.zeros(m, dtype=np.float64)
    running_min = 1.0
    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        val = (m / rank) * sorted_p[idx]
        running_min = min(running_min, val)
        adj_p[idx] = running_min

    adjusted_pvalues = np.zeros(m, dtype=np.float64)
    adjusted_pvalues[sorted_indices] = np.clip(adj_p, 0.0, 1.0)

    logger.debug(
        f"BH FDR (alpha={alpha}): rejected {np.sum(rejected)} / {m} voxels (p_thresh={p_threshold:.6e})"
    )
    return rejected, adjusted_pvalues, p_threshold


def benjamini_yekutieli(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute the Benjamini-Yekutieli (BY) procedure for arbitrary dependence.

    Scales alpha by c(m) = sum_{i=1}^m 1/i.

    Args:
        p_values: 1D array of p-values in (0, 1].
        alpha: Nominal FDR control level.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (rejected, adjusted_pvalues, p_threshold)
    """
    m = len(p_values)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    alpha_conservative = alpha / c_m
    return benjamini_hochberg(p_values, alpha=alpha_conservative)
