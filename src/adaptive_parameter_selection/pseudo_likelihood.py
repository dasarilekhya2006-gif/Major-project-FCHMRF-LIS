"""Maximum Pseudo-Likelihood Estimation (MPLE) for HMRF spatial interaction parameters."""

from typing import Tuple
import numpy as np
from scipy import optimize

from src.core.logging import get_logger

logger = get_logger(__name__)


def compute_mple_beta(
    binary_states_or_marginals: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    beta_bounds: Tuple[float, float] = (0.05, 3.0),
) -> float:
    """Estimate the spatial interaction parameter beta using Maximum Pseudo-Likelihood.

    The pseudo-likelihood approximates the intractable partition function by conditioning
    each site on its immediate neighborhood:
        PL(beta) = prod_i P(theta_i | theta_N(i); beta)

    Args:
        binary_states_or_marginals: (N,) binary states or posterior marginal probabilities q_i(1).
        neighbor_indices: (N, K) neighbor index matrix.
        neighbor_weights: (N, K) edge affinity weights.
        beta_bounds: Search interval for beta.

    Returns:
        float: Estimated optimal beta.
    """
    N = len(binary_states_or_marginals)
    q1 = np.asarray(binary_states_or_marginals, dtype=np.float64)

    # Pre-normalize neighbor weights row-wise
    row_sums = np.sum(neighbor_weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_weights = neighbor_weights / row_sums

    # Compute local neighbor consensus field: sum_j w_ij * (2 * q_j - 1) in [-1, 1]
    neighbor_spins = 2.0 * q1[neighbor_indices] - 1.0
    spin_field = np.sum(norm_weights * neighbor_spins, axis=1)  # shape (N,)
    target_spins = 2.0 * q1 - 1.0

    def neg_log_pseudo_likelihood(beta_val: float) -> float:
        # P(s_i = +1 | s_N(i)) = 1 / (1 + exp(-2 * beta * spin_field))
        exponent = np.clip(2.0 * beta_val * spin_field, -30.0, 30.0)
        # log P(s_i | s_N(i)) = s_i * beta * spin_field - log(2 * cosh(beta * spin_field))
        log_prob = (
            target_spins * beta_val * spin_field - np.log(np.exp(exponent) + np.exp(-exponent))
        )
        return -float(np.sum(log_prob))

    res = optimize.minimize_scalar(
        neg_log_pseudo_likelihood,
        bounds=beta_bounds,
        method="bounded",
    )

    opt_beta = float(res.x)
    logger.debug(f"MPLE estimated beta = {opt_beta:.4f} (bounds={beta_bounds})")
    return opt_beta
