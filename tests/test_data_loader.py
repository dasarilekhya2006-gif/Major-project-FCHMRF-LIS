"""Comprehensive unit test suite for Phase 1 data loader, validation, and metadata extraction."""

from pathlib import Path
import pytest
import numpy as np

from src.core.exceptions import (
    CorruptNiftiError,
    DataValidationError,
    DimensionMismatchError,
    EmptyVolumeError,
    InvalidAffineError,
)
from src.data.schemas import NeuroimagingMetadata, NeuroimagingVolume, ValidationReport
from src.data.loaders import (
    get_image_metadata,
    load_nifti,
    load_numpy_volume,
    save_nifti,
    validate_image,
)
from src.data.sample_data import (
    create_corrupt_nifti,
    create_empty_nifti,
    create_nan_infested_nifti,
    create_sample_3d_statistical_nifti,
    create_sample_4d_fmri_nifti,
    create_singular_affine_nifti,
)


@pytest.fixture
def tmp_nifti_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for NIfTI test files."""
    d = tmp_path / "nifti_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_load_valid_3d_nifti(tmp_nifti_dir: Path) -> None:
    """Test loading and normalizing a valid 3D statistical NIfTI volume."""
    fpath = tmp_nifti_dir / "valid_3d.nii.gz"
    create_sample_3d_statistical_nifti(
        fpath, shape=(20, 20, 10), voxel_size=(2.0, 2.0, 3.0), snr=2.5, seed=42
    )

    vol = load_nifti(fpath)
    assert isinstance(vol, NeuroimagingVolume)
    assert vol.is_3d
    assert not vol.is_4d
    assert vol.shape == (20, 20, 10)
    assert vol.spatial_shape == (20, 20, 10)
    assert vol.data.dtype == np.float32
    assert vol.affine.shape == (4, 4)
    assert vol.num_timepoints == 1
    assert vol.num_voxels == 4000
    assert vol.voxel_dimensions[:3] == (2.0, 2.0, 3.0)
    assert vol.metadata is not None
    assert isinstance(vol.metadata, NeuroimagingMetadata)
    assert vol.metadata.ndim == 3
    assert not vol.metadata.is_empty


def test_load_valid_4d_nifti(tmp_nifti_dir: Path) -> None:
    """Test loading a 4D fMRI timeseries volume and extracting timepoints."""
    fpath = tmp_nifti_dir / "valid_4d.nii.gz"
    create_sample_4d_fmri_nifti(
        fpath, spatial_shape=(16, 16, 8), num_timepoints=6, tr=2.0, seed=42
    )

    # 1. Load full 4D
    vol_4d = load_nifti(fpath)
    assert vol_4d.is_4d
    assert not vol_4d.is_3d
    assert vol_4d.shape == (16, 16, 8, 6)
    assert vol_4d.spatial_shape == (16, 16, 8)
    assert vol_4d.num_timepoints == 6
    assert vol_4d.voxel_dimensions[3] == 2.0

    # 2. Extract 3D slice via time_index argument
    vol_t2 = load_nifti(fpath, time_index=2)
    assert vol_t2.is_3d
    assert vol_t2.shape == (16, 16, 8)

    # 3. Test .to_3d() method on 4D instance
    vol_t4 = vol_4d.to_3d(time_index=4)
    assert vol_t4.is_3d
    assert vol_t4.shape == (16, 16, 8)

    # 4. Test .mean_across_time()
    vol_mean = vol_4d.mean_across_time()
    assert vol_mean.is_3d
    assert vol_mean.shape == (16, 16, 8)


def test_squeeze_singleton_dimension(tmp_nifti_dir: Path) -> None:
    """Test that a 4D volume with shape (X, Y, Z, 1) is squeezed to 3D."""
    fpath = tmp_nifti_dir / "singleton_4d.nii.gz"
    create_sample_4d_fmri_nifti(fpath, spatial_shape=(12, 12, 6), num_timepoints=1)

    vol = load_nifti(fpath, squeeze_singleton=True)
    assert vol.is_3d
    assert vol.shape == (12, 12, 6)


def test_get_image_metadata(tmp_nifti_dir: Path) -> None:
    """Test metadata extraction from a NIfTI file."""
    fpath = tmp_nifti_dir / "meta_test.nii.gz"
    create_sample_3d_statistical_nifti(fpath, shape=(16, 16, 8), snr=3.0)

    meta = get_image_metadata(fpath)
    assert isinstance(meta, NeuroimagingMetadata)
    assert meta.dimensions == (16, 16, 8)
    assert meta.ndim == 3
    assert meta.num_voxels == 16 * 16 * 8
    assert meta.max_value > meta.min_value
    assert not meta.has_nan
    assert not meta.has_inf
    assert not meta.is_empty
    assert "RAS" in meta.orientation or "UNKNOWN" in meta.orientation

    d = meta.to_dict()
    assert isinstance(d, dict)
    assert "affine" in d


def test_validate_image_valid_file(tmp_nifti_dir: Path) -> None:
    """Test validate_image on a valid NIfTI file."""
    fpath = tmp_nifti_dir / "valid.nii.gz"
    create_sample_3d_statistical_nifti(fpath, shape=(16, 16, 8))

    report = validate_image(fpath)
    assert isinstance(report, ValidationReport)
    assert report.is_valid
    assert len(report.errors) == 0


def test_empty_volume_detection(tmp_nifti_dir: Path) -> None:
    """Test that an all-zeros NIfTI file is detected as invalid and raises EmptyVolumeError."""
    fpath = tmp_nifti_dir / "empty.nii.gz"
    create_empty_nifti(fpath, shape=(12, 12, 6))

    # Test via validate_image
    report = validate_image(fpath, check_empty=True)
    assert not report.is_valid
    assert any("empty" in err.lower() for err in report.errors)

    # Test via load_nifti
    with pytest.raises(EmptyVolumeError):
        load_nifti(fpath, validate=True)


def test_corrupt_nifti_file(tmp_nifti_dir: Path) -> None:
    """Test that a corrupt/truncated NIfTI file raises CorruptNiftiError."""
    fpath = tmp_nifti_dir / "corrupt.nii.gz"
    create_corrupt_nifti(fpath)

    report = validate_image(fpath)
    assert not report.is_valid

    with pytest.raises(CorruptNiftiError):
        load_nifti(fpath)


def test_singular_affine_rejection(tmp_nifti_dir: Path) -> None:
    """Test that a singular/degenerate affine matrix is rejected."""
    fpath = tmp_nifti_dir / "singular_affine.nii.gz"
    create_singular_affine_nifti(fpath)

    report = validate_image(fpath, check_affine=True)
    assert not report.is_valid
    assert any("affine" in err.lower() or "singular" in err.lower() for err in report.errors)

    with pytest.raises(InvalidAffineError):
        load_nifti(fpath, validate=True)


def test_nan_policy_handling(tmp_nifti_dir: Path) -> None:
    """Test NaN handling policies ('zero' vs 'error')."""
    fpath = tmp_nifti_dir / "nan_volume.nii.gz"
    create_nan_infested_nifti(fpath, shape=(12, 12, 6), nan_ratio=0.3)

    # Under nan_policy='error', should raise DataValidationError
    with pytest.raises(DataValidationError):
        load_nifti(fpath, nan_policy="error", validate=False)

    # Under nan_policy='zero', should replace NaNs with 0.0 and succeed
    vol = load_nifti(fpath, nan_policy="zero", validate=False)
    assert not np.isnan(vol.data).any()
    assert np.all(np.isfinite(vol.data))


def test_nonexistent_file_handling() -> None:
    """Test loading a non-existent file path raises DataValidationError."""
    with pytest.raises(DataValidationError) as exc_info:
        load_nifti("non_existent_file_path.nii.gz")
    assert "does not exist" in str(exc_info.value)


def test_mask_spatial_dimension_mismatch(tmp_nifti_dir: Path) -> None:
    """Test that providing a mask with mismatched dimensions raises DimensionMismatchError."""
    vol_path = tmp_nifti_dir / "vol_3d.nii.gz"
    mask_path = tmp_nifti_dir / "mask_mismatched.nii.gz"

    create_sample_3d_statistical_nifti(vol_path, shape=(20, 20, 10))
    create_sample_3d_statistical_nifti(mask_path, shape=(16, 16, 8))

    with pytest.raises(DimensionMismatchError):
        load_nifti(vol_path, mask_path=mask_path)


def test_save_and_reload_roundtrip(tmp_nifti_dir: Path) -> None:
    """Test saving a NeuroimagingVolume and reloading it preserves data and spatial metadata."""
    save_path = tmp_nifti_dir / "roundtrip.nii.gz"
    data = np.random.randn(14, 14, 8).astype(np.float32)
    affine = np.diag([2.5, 2.5, 3.0, 1.0])

    original_vol = NeuroimagingVolume(
        data=data,
        affine=affine,
        voxel_dimensions=(2.5, 2.5, 3.0),
    )
    save_nifti(original_vol, save_path)

    reloaded_vol = load_nifti(save_path)
    np.testing.assert_array_almost_equal(reloaded_vol.data, data, decimal=5)
    np.testing.assert_array_almost_equal(reloaded_vol.affine, affine, decimal=5)
    assert reloaded_vol.voxel_dimensions[:3] == (2.5, 2.5, 3.0)
