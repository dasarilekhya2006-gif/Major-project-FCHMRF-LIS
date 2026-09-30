"""Research adapter for voxel-level statistical testing."""

from typing import Tuple
import numpy as np
from scipy import stats

from src.core.logging import get_logger

logger = get_logger(__name__)


def execute_statistics(
    preprocessed_data: np.ndarray,
    brain_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate voxel-level test statistics and two-sided p-values.

    Returns:
        (z_scores, p_values)
    """
    logger.info("Executing voxel-level statistical hypothesis testing...")
    z_scores = preprocessed_data.copy().astype(np.float32)
    z_scores[~brain_mask] = 0.0

    # Two-sided standard normal p-values
    abs_z = np.abs(z_scores)
    p_values = 2.0 * (1.0 - stats.norm.cdf(abs_z))
    p_values = np.clip(p_values, 1e-15, 1.0).astype(np.float32)
    p_values[~brain_mask] = 1.0

    return z_scores, p_values
