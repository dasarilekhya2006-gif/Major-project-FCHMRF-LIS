"""Reproducibility module for setting global random seeds across libraries."""

import os
import random
from typing import Optional
import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> int:
    """Set seeds across Python standard library, NumPy, and PyTorch (if available).

    Ensures that all stochastic processes (random field generation, dataset sampling,
    weight initialization, and expectation-maximization perturbations) remain fully
    deterministic and reproducible.

    Args:
        seed: Integer random seed (default: 42).
        deterministic_cuda: If True, instructs PyTorch to use deterministic cuDNN algorithms.

    Returns:
        int: The seed that was set.
    """
    logger.info(f"Setting global random seed to {seed} (deterministic_cuda={deterministic_cuda})")

    # Python built-in random and hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (lazy import to allow lightweight non-torch environments if needed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic_cuda:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("PyTorch is not installed or unavailable; skipped torch seed initialization.")

    return seed
