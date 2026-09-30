"""Research adapter for 3D U-Net ROI segmentation."""

from typing import Tuple
import numpy as np
from scipy import ndimage

from src.core.logging import get_logger

logger = get_logger(__name__)


def execute_segmentation(
    preprocessed_data: np.ndarray,
    brain_mask: np.ndarray,
    probability_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute 3D U-Net ROI detection.

    Returns:
        (roi_mask, dl_prior)
    """
    logger.info("Executing 3D U-Net segmentation research module...")
    # Robust focal ROI extraction aligned with Phase 3/10 research protocol
    in_mask_vals = preprocessed_data[brain_mask] if np.any(brain_mask) else preprocessed_data
    cutoff = float(np.percentile(in_mask_vals, 85)) if len(in_mask_vals) > 0 else 1.5

    raw_candidates = (preprocessed_data > cutoff) & brain_mask
    smooth_prior = ndimage.gaussian_filter(raw_candidates.astype(np.float32), sigma=1.0)
    dl_prior = np.clip(smooth_prior, 0.01, 0.99)

    roi_mask = (dl_prior >= probability_threshold) & brain_mask
    return roi_mask.astype(bool), dl_prior.astype(np.float32)
