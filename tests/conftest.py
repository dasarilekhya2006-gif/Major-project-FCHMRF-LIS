"""Shared pytest fixtures for the test suite."""

import pytest
import numpy as np

from src.core.seed import set_seed
from src.data.schemas import NeuroimagingVolume, SyntheticBenchmarkSample
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters


@pytest.fixture(autouse=True)
def init_seed() -> None:
    """Ensure every test runs under fixed seed 42."""
    set_seed(42)


@pytest.fixture
def small_synthetic_sample() -> SyntheticBenchmarkSample:
    """Generate a fast, small synthetic 3D sample for unit testing."""
    return generate_synthetic_spatial_dataset(
        grid_size=(16, 16, 8),
        snr=2.5,
        num_clusters=2,
        seed=42,
    )


@pytest.fixture
def sample_volume(small_synthetic_sample: SyntheticBenchmarkSample) -> NeuroimagingVolume:
    """Provide a validated NeuroimagingVolume fixture."""
    mask = np.ones(small_synthetic_sample.z_map.shape, dtype=bool)
    # Mask out outer boundary
    mask[0, :, :] = False
    mask[-1, :, :] = False
    return NeuroimagingVolume(
        data=small_synthetic_sample.z_map,
        mask=mask,
    )


@pytest.fixture
def default_mixture_params() -> TwoGroupMixtureParameters:
    """Provide sample two-group mixture parameters."""
    return TwoGroupMixtureParameters(
        p0=0.90,
        mu0=0.0,
        sigma0=1.0,
        mu1=2.5,
        sigma1=1.0,
    )
