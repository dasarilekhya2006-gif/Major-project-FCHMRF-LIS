"""Unit tests verifying Phase 1 neuroimaging data loading requirements:
- load_nifti()
- validate_nifti()
- get_nifti_metadata()
Supports 3D and 4D NIfTI volumes via NiBabel, reads dimensions, voxel spacing,
affine matrix, orientation, metadata, detects empty, invalid, NaN, and infinite data,
and verifies the original disk image remains unaltered.
"""

from pathlib import Path
import pytest
import numpy as np
import nibabel as nib

from data import (
    load_nifti,
    validate_nifti,
    get_nifti_metadata,
    NeuroimagingMetadata,
    NeuroimagingVolume,
    ValidationReport,
)
from src.core.exceptions import (
    DataValidationError,
    EmptyVolumeError,
    InvalidAffineError,
    CorruptNiftiError,
)


@pytest.fixture
def tmp_nifti_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for NIfTI test files."""
    d = tmp_path / "phase1_test_nifti"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_load_and_metadata_3d(tmp_nifti_dir: Path) -> None:
    """Verify loading 3D NIfTI, extracting dimensions, spacing, affine, orientation, metadata."""
    fpath = tmp_nifti_dir / "test_3d.nii.gz"
    arr = np.random.RandomState(42).randn(15, 12, 8).astype(np.float32)
    affine = np.diag([2.0, 2.5, 3.0, 1.0])
    img = nib.Nifti1Image(arr, affine=affine)
    nib.save(img, str(fpath))

    # Read metadata
    meta = get_nifti_metadata(fpath)
    assert isinstance(meta, NeuroimagingMetadata)
    assert meta.dimensions == (15, 12, 8)
    assert meta.voxel_dimensions[:3] == (2.0, 2.5, 3.0)
    assert np.allclose(meta.affine, affine)
    assert meta.orientation is not None
    assert meta.ndim == 3
    assert not meta.is_empty
    assert not meta.has_nan
    assert not meta.has_inf

    # Load volume
    vol = load_nifti(fpath)
    assert isinstance(vol, NeuroimagingVolume)
    assert vol.is_3d
    assert not vol.is_4d
    assert vol.shape == (15, 12, 8)
    assert vol.voxel_dimensions[:3] == (2.0, 2.5, 3.0)
    assert np.allclose(vol.affine, affine)
    assert vol.num_voxels == 15 * 12 * 8


def test_load_and_metadata_4d(tmp_nifti_dir: Path) -> None:
    """Verify loading 4D NIfTI, extracting dimensions, spacing, affine, metadata."""
    fpath = tmp_nifti_dir / "test_4d.nii.gz"
    arr = np.random.RandomState(101).randn(10, 10, 6, 5).astype(np.float32)
    affine = np.diag([3.0, 3.0, 4.0, 1.0])
    img = nib.Nifti1Image(arr, affine=affine)
    img.header.set_zooms((3.0, 3.0, 4.0, 2.0))
    nib.save(img, str(fpath))

    meta = get_nifti_metadata(fpath)
    assert meta.ndim == 4
    assert meta.dimensions == (10, 10, 6, 5)
    assert meta.voxel_dimensions[:4] == (3.0, 3.0, 4.0, 2.0)

    vol = load_nifti(fpath)
    assert vol.is_4d
    assert vol.num_timepoints == 5
    assert vol.shape == (10, 10, 6, 5)


def test_detect_empty_volume(tmp_nifti_dir: Path) -> None:
    """Verify empty (all-zero) volume detection."""
    fpath = tmp_nifti_dir / "empty.nii.gz"
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(fpath))

    report = validate_nifti(fpath, check_empty=True)
    assert not report.is_valid
    assert any("empty" in err.lower() for err in report.errors)

    with pytest.raises(EmptyVolumeError):
        load_nifti(fpath, validate=True, allow_empty=False)


def test_detect_nan_and_inf_data(tmp_nifti_dir: Path) -> None:
    """Verify detection of NaN and Inf voxel values."""
    fpath = tmp_nifti_dir / "nan_inf.nii.gz"
    arr = np.ones((8, 8, 8), dtype=np.float32)
    arr[2, 2, 2] = np.nan
    arr[4, 4, 4] = np.inf
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(fpath))

    meta = get_nifti_metadata(fpath)
    assert meta.has_nan
    assert meta.has_inf

    # nan_policy='error' must reject
    with pytest.raises(DataValidationError):
        load_nifti(fpath, nan_policy="error")

    # nan_policy='zero' replaces them without failing
    vol = load_nifti(fpath, nan_policy="zero")
    assert np.isfinite(vol.data).all()
    assert vol.data[2, 2, 2] == 0.0
    assert vol.data[4, 4, 4] == 0.0


def test_unaltered_original_file(tmp_nifti_dir: Path) -> None:
    """Verify that loading/validating does NOT alter the original file on disk."""
    fpath = tmp_nifti_dir / "immutable.nii.gz"
    arr = np.random.RandomState(7).randn(10, 10, 10).astype(np.float32)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(fpath))

    original_bytes = fpath.read_bytes()
    original_mtime = fpath.stat().st_mtime_ns

    _ = get_nifti_metadata(fpath)
    _ = validate_nifti(fpath)
    _ = load_nifti(fpath, nan_policy="zero")

    assert fpath.read_bytes() == original_bytes
    assert fpath.stat().st_mtime_ns == original_mtime
