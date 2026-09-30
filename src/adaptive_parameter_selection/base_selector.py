"""Abstract base class for adaptive parameter selection in fcHMRF."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters


class BaseParameterSelector(ABC):
    """Abstract interface for tuning fcHMRF spatial coupling beta and DL prior weight lambda."""

    @abstractmethod
    def fit(
        self,
        z_scores: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        dl_prior_p1: np.ndarray,
        initial_mixture: TwoGroupMixtureParameters,
    ) -> Tuple[float, float, TwoGroupMixtureParameters]:
        """Estimate optimal hyperparameters (beta, lambda_dl, updated_mixture_params).

        Args:
            z_scores: 1D array of z-scores (length N).
            neighbor_indices: (N, K) neighbor indices.
            neighbor_weights: (N, K) edge affinity weights.
            dl_prior_p1: (N,) deep learning prior probabilities.
            initial_mixture: Initial emission mixture parameters.

        Returns:
            Tuple[float, float, TwoGroupMixtureParameters]:
                - beta_opt: Selected spatial coupling strength.
                - lambda_opt: Selected DL prior mixing weight.
                - mixture_opt: Optimized emission mixture parameters.
        """
        pass
