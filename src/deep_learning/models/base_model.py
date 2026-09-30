"""Abstract base class for deep learning ROI prior estimators."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    nn = object  # type: ignore


class BaseROIPriorModel(ABC, nn.Module if hasattr(nn, "Module") else object):
    """Abstract interface for lightweight deep learning ROI prior detectors.

    Generates voxel-wise prior activation probabilities:
        pi_i = P(theta_i = 1 | X_local)
    which feed directly into the fcHMRF unary energy potentials.
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass taking input tensor of shape (B, C, D, H, W) or (B, C, H, W).

        Returns:
            Logits or probabilities of shape (B, 1, D, H, W) in [0, 1].
        """
        pass

    @abstractmethod
    def predict_prior_map(self, volume: np.ndarray) -> np.ndarray:
        """Convenience method taking a 3D numpy array and returning a 3D numpy prior map in [0, 1].

        Args:
            volume: 3D numpy array of statistical values.

        Returns:
            np.ndarray: 3D float numpy array of probabilities matching volume shape.
        """
        pass
