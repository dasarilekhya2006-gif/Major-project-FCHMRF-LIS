"""Inference engine for predicting voxel-wise prior activation probabilities."""

from typing import Optional, Union
import numpy as np

from src.core.exceptions import DeepLearningModelError
from src.core.logging import get_logger
from src.deep_learning.models.base_model import BaseROIPriorModel

logger = get_logger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ROIPriorPredictor:
    """Predicts spatial prior activation probabilities pi_i in [0, 1] using a trained DL model.

    If no model is supplied, provides an empirical two-group empirical Bayes fallback.
    """

    def __init__(
        self,
        model: Optional[BaseROIPriorModel] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the prior predictor.

        Args:
            model: Optional instance of BaseROIPriorModel.
            device: 'cpu' or 'cuda'.
        """
        self.model = model
        self.device = device
        if self.model is not None and TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            self.model.to(torch.device(device))
            self.model.eval()

    def predict(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate voxel-wise prior probability map pi_i = P(theta_i = 1 | X).

        Args:
            volume: 3D numpy array of z-scores or statistical contrast.
            mask: Optional binary mask of valid voxels.

        Returns:
            np.ndarray: 3D numpy array of prior probabilities in [0, 1].
        """
        if self.model is not None and TORCH_AVAILABLE:
            try:
                prior_map = self.model.predict_prior_map(volume)
                if mask is not None:
                    prior_map[~mask] = 0.0
                return np.clip(prior_map, 1e-4, 1.0 - 1e-4)
            except Exception as e:
                logger.warning(f"DL model inference encountered error: {e}. Falling back to empirical prior.")

        # Fallback: Empirical spatial prior based on smoothed positive tail evidence
        logger.debug("Computing heuristic empirical spatial prior map.")
        from scipy.ndimage import gaussian_filter
        from scipy.special import expit

        # Soft positive evidence sigmoid on smoothed z-scores
        smoothed = gaussian_filter(volume, sigma=1.5)
        # Shift so z=2.0 maps to ~0.2, z=3.0 maps to ~0.5
        prior = expit(smoothed - 3.0)

        if mask is not None:
            prior[~mask] = 0.0

        return np.clip(prior, 1e-4, 1.0 - 1e-4)
