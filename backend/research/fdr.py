"""Research adapter for baseline multiple testing procedures (BH-FDR)."""

from typing import Tuple
import numpy as np

from src.evaluation.baselines import BenjaminiHochbergCorrection
from src.core.logging import get_logger

logger = get_logger(__name__)


def execute_baseline_fdr(
    z_scores: np.ndarray,
    p_values: np.ndarray,
    brain_mask: np.ndarray,
    nominal_alpha: float = 0.05,
) -> Tuple[np.ndarray, int]:
    """Execute standard Benjamini-Hochberg FDR baseline correction.

    Returns:
        (baseline_significant_mask, discovery_count)
    """
    logger.info("Executing standard Benjamini-Hochberg FDR baseline correction...")
    bh = BenjaminiHochbergCorrection(alpha=nominal_alpha)
    res = bh.run(z_scores=z_scores, p_values=p_values, spatial_mask=brain_mask)
    sig_mask = res.significant_mask.astype(bool)
    return sig_mask, int(np.sum(sig_mask))
