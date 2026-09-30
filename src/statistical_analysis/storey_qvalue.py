"""Storey's positive False Discovery Rate (pFDR) and q-value estimation."""

from typing import Optional, Tuple
import numpy as np
from scipy import interpolate

from src.core.logging import get_logger

logger = get_logger(__name__)


def estimate_pi0(
    p_values: np.ndarray,
    lambda_tuning: Optional[float] = None,
) -> float:
    """Estimate the true null hypothesis proportion pi_0 = m_0 / m.

    Uses Storey (2002):
        pi_0_hat(lambda) = #{p_i > lambda} / (m * (1 - lambda))

    Args:
        p_values: 1D array of p-values.
        lambda_tuning: Tuning parameter in [0.0, 1.0). If None, uses a cubic spline fit over a grid.

    Returns:
        float: Estimated proportion pi_0 in (0.0, 1.0].
    """
    m = len(p_values)
    if lambda_tuning is not None:
        pi0 = np.sum(p_values > lambda_tuning) / (m * (1.0 - lambda_tuning))
        return float(np.clip(pi0, 0.01, 1.0))

    # Grid evaluation and cubic smoothing spline extrapolation to lambda -> 1
    lambdas = np.arange(0.05, 0.95, 0.05)
    pi0_grid = np.array([np.sum(p_values > l) / (m * (1.0 - l)) for l in lambdas])

    try:
        spline = interpolate.UnivariateSpline(lambdas, pi0_grid, k=3, s=len(lambdas))
        pi0_est = float(spline(0.95))
    except Exception:
        # Fallback to median or high lambda evaluation if spline fails
        pi0_est = float(pi0_grid[-1])

    pi0_clamped = float(np.clip(pi0_est, 0.01, 1.0))
    logger.debug(f"Estimated Storey's pi_0 = {pi0_clamped:.4f}")
    return pi0_clamped


def storey_qvalues(
    p_values: np.ndarray,
    alpha: float = 0.05,
    pi0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Storey's q-values and reject hypotheses with q <= alpha.

    Args:
        p_values: 1D array of p-values.
        alpha: Nominal FDR threshold.
        pi0: Optional pre-estimated null proportion pi_0.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - rejected: Boolean array of rejections.
            - q_values: Storey q-values matching p_values length.
            - pi0: Estimated null proportion used.
    """
    p_vals = np.asarray(p_values, dtype=np.float64).ravel()
    m = len(p_vals)

    if pi0 is None:
        pi0 = estimate_pi0(p_vals)

    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]

    ranks = np.arange(1, m + 1)
    q_sorted = (pi0 * m * sorted_p) / ranks

    # Enforce monotonicity: q_(i) = min_{j >= i} q_(j)
    running_min = 1.0
    for idx in range(m - 1, -1, -1):
        running_min = min(running_min, q_sorted[idx])
        q_sorted[idx] = running_min

    q_values = np.zeros(m, dtype=np.float64)
    q_values[sorted_indices] = np.clip(q_sorted, 0.0, 1.0)

    rejected = q_values <= alpha
    logger.debug(f"Storey q-value (alpha={alpha}, pi0={pi0:.3f}): rejected {np.sum(rejected)} / {m}")
    return rejected, q_values, pi0
