"""Local Index of Significance (LIS) computation for spatial hypothesis testing."""

from typing import Optional
import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


def compute_lis(posterior_active_q1: np.ndarray) -> np.ndarray:
    """Calculate the Local Index of Significance (LIS) from posterior active marginals.

    Under the spatial mixture model:
        LIS_i = P(theta_i = 0 | Z) = 1 - Q(theta_i = 1)

    LIS represents the posterior probability that voxel i is null (inactive) given the
    entire observed spatial random field and contextual bilateral graph interactions.

    Args:
        posterior_active_q1: 1D array of posterior activation probabilities Q(theta_i = 1).

    Returns:
        np.ndarray: 1D array of LIS values in [0, 1].
    """
    lis = 1.0 - posterior_active_q1
    lis_clamped = np.clip(lis, 0.0, 1.0)
    logger.debug(f"Computed LIS: min={np.min(lis_clamped):.4f}, median={np.median(lis_clamped):.4f}")
    return lis_clamped
