"""Unit tests verifying Phase 2 neuroimaging preprocessing requirements:
- Input: NIfTI volume from Phase 1 (load_nifti)
- Configurable NaN/Inf handling
- Intensity normalization (z-score, min-max, robust)
- Optional denoising (gaussian, median, none)
- Optional resampling (with consistent affine and voxel dimensions)
- Optional brain masking when a valid mask exists
- Strict preservation of affine matrix, voxel spacing, spatial dimensions, and orientation
- Reusable functions in preprocessing/
"""

from pathlib import Path
import pytest
import numpy as np
import nibabel as nib

from data import load_nifti, get_nifti_metadata
from preprocessing import (
    handle_nan_inf,
    normalize_intensity,
    denoise_volume,
    resample_volume,
    apply_brain_mask,
    preprocess_volume,
    PreprocessingConfig,
    PreprocessedVolume,
)


@pytest.fixture
def phase2_synthetic_nifti(tmp_path: Path) -> Path:
    """Create a synthetic 3D statistical NIfTI file with known geometry."""
    fpath = tmp_path / "synthetic_phase2.nii.gz"
    rng = np.random.RandomState(42)
    data = rng.randn(16, 16, 10).astype(np.float32) * 5.0 + 20.0
    affine = np.array([
        [2.0, 0.0, 0.0, -15.0],
        [0.0, 2.0, 0.0, -15.0],
        [0.0, 0.0, 3.0, -5.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    img = nib.Nifti1Image(data, affine=affine)
    img.header.set_zooms((2.0, 2.0, 3.0))
    nib.save(img, str(fpath))
    return fpath


def test_handle_nan_inf_preserves_geometry(phase2_synthetic_nifti: Path) -> None:
    """Verify NaN/Inf imputation removes non-finite values and preserves geometry."""
    vol = load_nifti(phase2_synthetic_nifti)

    # Inject NaNs and Infs
    vol.data[3, 3, 3] = np.nan
    vol.data[5, 5, 5] = np.inf
    vol.data[7, 7, 7] = -np.inf

    cleaned = handle_nan_inf(vol, method="zero")

    assert np.isfinite(cleaned.data).all()
    assert cleaned.data[3, 3, 3] == 0.0
    assert cleaned.data[5, 5, 5] == 0.0
    assert cleaned.data[7, 7, 7] == 0.0

    # Strict geometry preservation
    assert cleaned.shape == vol.shape
    assert cleaned.voxel_dimensions == vol.voxel_dimensions
    assert np.allclose(cleaned.affine, vol.affine)
    assert cleaned.spatial_shape == vol.spatial_shape


def test_normalize_intensity_preserves_geometry(phase2_synthetic_nifti: Path) -> None:
    """Verify intensity normalization normalizes intensities and strictly preserves geometry."""
    vol = load_nifti(phase2_synthetic_nifti)

    # Z-score normalization
    norm_z = normalize_intensity(vol, method="z_score")
    assert np.isclose(np.mean(norm_z.data), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm_z.data), 1.0, atol=1e-4)

    assert norm_z.shape == vol.shape
    assert norm_z.voxel_dimensions == vol.voxel_dimensions
    assert np.allclose(norm_z.affine, vol.affine)

    # Min-max normalization
    norm_mm = normalize_intensity(vol, method="min_max")
    assert np.isclose(np.min(norm_mm.data), 0.0, atol=1e-5)
    assert np.isclose(np.max(norm_mm.data), 1.0, atol=1e-5)
    assert norm_mm.shape == vol.shape
    assert np.allclose(norm_mm.affine, vol.affine)


def test_denoise_volume_optional_and_preserves_geometry(phase2_synthetic_nifti: Path) -> None:
    """Verify denoising is optional (none returns unaltered) and preserves geometry when applied."""
    vol = load_nifti(phase2_synthetic_nifti)

    # When method is 'none', returns identical array
    denoised_none = denoise_volume(vol, method="none")
    assert np.array_equal(denoised_none.data, vol.data)
    assert np.allclose(denoised_none.affine, vol.affine)

    # When method is 'gaussian'
    denoised_gauss = denoise_volume(vol, method="gaussian", fwhm_mm=2.0)
    assert not np.array_equal(denoised_gauss.data, vol.data)
    assert denoised_gauss.shape == vol.shape
    assert denoised_gauss.voxel_dimensions == vol.voxel_dimensions
    assert np.allclose(denoised_gauss.affine, vol.affine)


def test_apply_brain_mask_optional_when_valid_mask_exists(phase2_synthetic_nifti: Path) -> None:
    """Verify masking only applies when a valid mask exists, preserving geometry."""
    vol = load_nifti(phase2_synthetic_nifti)

    # 1. No mask provided -> unaltered
    unmasked = apply_brain_mask(vol, mask=None)
    assert np.array_equal(unmasked.data, vol.data)
    assert np.allclose(unmasked.affine, vol.affine)

    # 2. Valid mask provided
    mask = np.zeros(vol.shape, dtype=bool)
    mask[4:12, 4:12, 3:8] = True

    masked = apply_brain_mask(vol, mask=mask)
    assert masked.mask is not None
    assert np.array_equal(masked.mask, mask)
    assert np.all(masked.data[~mask] == 0.0)
    assert np.all(masked.data[mask] == vol.data[mask])

    # Geometry preservation
    assert masked.shape == vol.shape
    assert masked.voxel_dimensions == vol.voxel_dimensions
    assert np.allclose(masked.affine, vol.affine)


def test_resample_volume_updates_affine_consistently(phase2_synthetic_nifti: Path) -> None:
    """Verify optional resampling updates affine and spacing with coordinate consistency."""
    vol = load_nifti(phase2_synthetic_nifti)
    target_spacing = (1.0, 1.0, 1.5)

    resampled = resample_volume(vol, target_spacing=target_spacing)

    # Spacing updated to target
    assert np.allclose(resampled.voxel_dimensions[:3], target_spacing, atol=1e-3)
    # Origin preserved in physical world space
    assert np.allclose(resampled.affine[:3, 3], vol.affine[:3, 3])
    # Resolution scaled
    assert resampled.shape[0] > vol.shape[0]
    assert resampled.shape[1] > vol.shape[1]
    assert resampled.shape[2] > vol.shape[2]


def test_preprocess_volume_pipeline_no_unjustified_transforms(phase2_synthetic_nifti: Path) -> None:
    """Verify preprocess_volume does NOT apply unjustified transformations by default."""
    vol = load_nifti(phase2_synthetic_nifti)

    # Default configuration: no denoising, no resampling, no auto-masking
    cfg = PreprocessingConfig(
        denoising_method="none",
        compute_mask_if_missing=False,
        pad_crop_mode="none",
        target_voxel_spacing=None,
    )

    out = preprocess_volume(vol, config=cfg)
    assert isinstance(out, PreprocessedVolume)

    # Exact preservation of spatial geometry
    assert out.shape == vol.shape
    assert out.voxel_dimensions == vol.voxel_dimensions
    assert np.allclose(out.affine, vol.affine)
    assert out.metadata.orientation == vol.metadata.orientation
