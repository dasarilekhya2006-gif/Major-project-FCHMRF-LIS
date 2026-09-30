"""Unit tests for data generation, schemas, and loaders."""

import pytest
import numpy as np

from src.data.schemas import NeuroimagingVolume, SyntheticBenchmarkSample
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.data.loaders import load_numpy_volume
from src.core.exceptions import DataValidationError


def test_synthetic_dataset_generation() -> None:
    """Test synthetic 3D spatial field generator."""
    sample = generate_synthetic_spatial_dataset(
        grid_size=(20, 20, 10),
        snr=2.0,
        num_clusters=2,
        seed=42,
    )
    assert isinstance(sample, SyntheticBenchmarkSample)
    assert sample.z_map.shape == (20, 20, 10)
    assert sample.ground_truth.shape == (20, 20, 10)
    assert np.any(sample.ground_truth)
    assert np.max(sample.z_map) > np.min(sample.z_map)


def test_neuroimaging_volume_validation() -> None:
    """Test validation in NeuroimagingVolume container."""
    data_3d = np.zeros((10, 10, 10), dtype=np.float32)
    vol = load_numpy_volume(data_3d)
    assert vol.num_voxels == 1000

    # 4D array should be supported
    data_4d = np.zeros((10, 10, 10, 2), dtype=np.float32)
    vol_4d = load_numpy_volume(data_4d)
    assert vol_4d.is_4d
    assert vol_4d.num_timepoints == 2

    # 5D array should fail
    data_5d = np.zeros((10, 10, 10, 2, 2))
    with pytest.raises(DataValidationError):
        load_numpy_volume(data_5d)
