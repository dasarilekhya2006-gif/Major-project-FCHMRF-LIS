"""Unit tests for the core package (seed, logging, exceptions, config)."""

from pathlib import Path
import pytest
import numpy as np

from src.core.seed import set_seed
from src.core.logging import get_logger, setup_logging
from src.core.exceptions import NeuroimagingError, DataValidationError
from src.core.config import load_config, ProjectConfig


def test_seed_reproducibility() -> None:
    """Test that set_seed guarantees deterministic random numbers."""
    set_seed(123)
    val1 = np.random.randn(5)
    set_seed(123)
    val2 = np.random.randn(5)
    np.testing.assert_array_equal(val1, val2)


def test_custom_exception_hierarchy() -> None:
    """Test that custom exceptions inherit properly from NeuroimagingError."""
    with pytest.raises(NeuroimagingError):
        raise DataValidationError("Test error message")


def test_load_default_config() -> None:
    """Test that default configuration can be loaded and validated."""
    cfg = load_config(Path("configs/default.yaml"))
    assert isinstance(cfg, ProjectConfig)
    assert cfg.statistical_analysis.nominal_fdr_alpha == 0.05
    assert cfg.fchmrf_lis.spatial_sigma > 0.0
