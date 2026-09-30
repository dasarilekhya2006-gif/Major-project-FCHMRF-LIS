"""Comprehensive unit tests for Phase 2 neuroimaging preprocessing pipeline."""

from pathlib import Path
import pytest
import numpy as np

from src.core.exceptions import PreprocessingError
from src.data.schemas import NeuroimagingVolume
from src.data.loaders import load_nifti
from src.data.sample_data import create_sample_3d_statistical_nifti
from src.preprocessing.schemas import PreprocessingConfig, PreprocessingMetadata
from src.preprocessing.imputation import handle_missing_voxels
from src.preprocessing.normalization import (
    denormalize_intensities,
    normalize_intensities,
    standardize_zscores,
    convert_pvalues_to_zscores,
)
from src.preprocessing.denoising import denoise_volume
from src.preprocessing.resampling import resample_to_spacing
from src.preprocessing.resizing import standardize_spatial_dimensions
from src.preprocessing.masking import (
    apply_mask,
    compute_brain_mask,
    extract_bounding_box,
    load_or_compute_mask,
    unmask,
)
from src.preprocessing.spatial_graph import (
    build_knn_spatial_adjacency,
    extract_masked_coordinates,
)
from src.preprocessing.pipeline import (
    PreprocessedVolume,
    PreprocessingPipeline,
    preprocess_neuroimaging_volume,
)
from src.preprocessing.qc import generate_preprocessing_qc_plot


@pytest.fixture
def sample_vol_with_nans() -> NeuroimagingVolume:
    """Fixture providing a 3D volume with embedded NaNs and Infs."""
    data = np.random.randn(20, 20, 10).astype(np.float32)
    # Inject non-finite values
    data[2, 2, 2] = np.nan
    data[3, 3, 3] = np.inf
    data[4, 4, 4] = -np.inf

    mask = np.ones((20, 20, 10), dtype=bool)
    mask[0, :, :] = False
    mask[-1, :, :] = False

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-20.0, -20.0, -10.0]

    return NeuroimagingVolume(
        data=data,
        affine=affine,
        voxel_dimensions=(2.0, 2.0, 2.0),
        mask=mask,
        subject_id="test_sub_nan",
    )


# -----------------------------------------------------------------------------
# Imputation Tests
# -----------------------------------------------------------------------------

def test_imputation_zero(sample_vol_with_nans: NeuroimagingVolume) -> None:
    """Test replacing non-finite values with 0.0."""
    cleaned, count = handle_missing_voxels(sample_vol_with_nans.data, method="zero")
    assert count == 3
    assert not np.isnan(cleaned).any()
    assert not np.isinf(cleaned).any()
    assert cleaned[2, 2, 2] == 0.0
    assert cleaned[3, 3, 3] == 0.0


def test_imputation_mean(sample_vol_with_nans: NeuroimagingVolume) -> None:
    """Test replacing non-finite values with in-mask mean."""
    cleaned, count = handle_missing_voxels(
        sample_vol_with_nans.data, mask=sample_vol_with_nans.mask, method="mean"
    )
    assert count == 3
    assert not np.isnan(cleaned).any()
    assert np.isfinite(cleaned).all()


def test_imputation_nearest_finite(sample_vol_with_nans: NeuroimagingVolume) -> None:
    """Test replacing non-finite values with nearest finite neighbors."""
    cleaned, count = handle_missing_voxels(sample_vol_with_nans.data, method="nearest_finite")
    assert count == 3
    assert not np.isnan(cleaned).any()
    assert np.isfinite(cleaned).all()


# -----------------------------------------------------------------------------
# Normalization Tests
# -----------------------------------------------------------------------------

def test_normalize_intensities_zscore() -> None:
    """Test z-score normalization on valid finite data."""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    norm, center, scale = normalize_intensities(data, method="zscore")
    assert np.isclose(center, 30.0)
    np.testing.assert_almost_equal(np.mean(norm), 0.0, decimal=5)
    np.testing.assert_almost_equal(np.std(norm), 1.0, decimal=5)

    # Inversion test
    inverted = denormalize_intensities(norm, center, scale)
    np.testing.assert_array_almost_equal(inverted, data, decimal=5)


def test_normalize_intensities_min_max() -> None:
    """Test min-max normalization mapping to [0, 1]."""
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    norm, center, scale = normalize_intensities(data, method="min_max")
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[-1], 1.0)
    assert np.all(norm >= 0.0) and np.all(norm <= 1.0)


def test_normalize_intensities_percentile_clip() -> None:
    """Test robust percentile clipping normalization."""
    data = np.random.randn(1000).astype(np.float32)
    data[0] = 50.0  # extreme outlier
    data[1] = -50.0 # extreme outlier
    norm, center, scale = normalize_intensities(data, method="percentile_clip", percentile_bounds=(2.0, 98.0))
    assert np.max(norm) < 10.0
    assert np.min(norm) > -10.0


# -----------------------------------------------------------------------------
# Denoising Tests
# -----------------------------------------------------------------------------

def test_denoise_volume_modes() -> None:
    """Test Gaussian and median denoising filters."""
    vol = np.random.randn(16, 16, 8).astype(np.float32)
    mask = np.ones((16, 16, 8), dtype=bool)

    # 1. Passthrough 'none'
    none_vol = denoise_volume(vol, method="none")
    np.testing.assert_array_equal(vol, none_vol)

    # 2. Gaussian smoothing
    gauss_vol = denoise_volume(vol, method="gaussian", fwhm_mm=3.0, voxel_dims=(2.0, 2.0, 2.0), mask=mask)
    assert gauss_vol.shape == vol.shape
    # Smoothing reduces local variance
    assert np.std(gauss_vol) < np.std(vol)

    # 3. Median filtering
    median_vol = denoise_volume(vol, method="median", median_kernel_size=3, mask=mask)
    assert median_vol.shape == vol.shape


# -----------------------------------------------------------------------------
# Spatial Resampling Tests
# -----------------------------------------------------------------------------

def test_resample_to_spacing() -> None:
    """Test spatial resampling and affine scaling."""
    data = np.random.randn(12, 12, 6).astype(np.float32)
    orig_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    orig_affine[:3, 3] = [-18.0, -18.0, -9.0]

    vol = NeuroimagingVolume(
        data=data,
        affine=orig_affine,
        voxel_dimensions=(3.0, 3.0, 3.0),
        mask=np.ones((12, 12, 6), dtype=bool),
    )

    target_spacing = (2.0, 2.0, 2.0)
    resampled = resample_to_spacing(vol, target_spacing=target_spacing, order=1)

    # 3.0 / 2.0 = 1.5 zoom factor -> shape ~ 18, 18, 9
    assert resampled.shape == (18, 18, 9)
    assert resampled.voxel_dimensions == (2.0, 2.0, 2.0)

    # Check affine spatial columns scaled proportionally
    expected_scale = 2.0 / 3.0
    for j in range(3):
        assert np.isclose(resampled.affine[j, j], orig_affine[j, j] * expected_scale)


# -----------------------------------------------------------------------------
# Spatial Dimension Standardization (Padding / Cropping) Tests
# -----------------------------------------------------------------------------

def test_standardize_spatial_dimensions_divisible() -> None:
    """Test padding dimensions to be divisible by 8 for 3D U-Net."""
    data = np.random.randn(18, 19, 10).astype(np.float32)
    orig_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    orig_affine[:3, 3] = [10.0, 20.0, 30.0]

    vol = NeuroimagingVolume(data=data, affine=orig_affine, voxel_dimensions=(2.0, 2.0, 2.0))

    resized, offsets = standardize_spatial_dimensions(vol, divisible_by=8, mode="pad_to_divisible")

    # 18 -> 24 (pad 6), 19 -> 24 (pad 5), 10 -> 16 (pad 6)
    assert resized.shape == (24, 24, 16)
    assert all(dim % 8 == 0 for dim in resized.shape)

    # Verify origin was shifted to preserve world-space MNI coordinates
    # For axis 0: pad 6 -> before = 3. Shift = -3 * voxel_size = -6.0
    assert np.isclose(resized.affine[0, 3], orig_affine[0, 3] - 3 * 2.0)


def test_standardize_spatial_dimensions_center_pad_crop() -> None:
    """Test fixed target shape center padding and cropping."""
    data = np.random.randn(30, 30, 20).astype(np.float32)
    vol = NeuroimagingVolume(data=data, affine=np.eye(4), voxel_dimensions=(2.0, 2.0, 2.0))

    target = (32, 24, 16)
    resized, offsets = standardize_spatial_dimensions(vol, target_shape=target, mode="center_pad_crop")
    assert resized.shape == target


# -----------------------------------------------------------------------------
# Masking & Bounding Box Tests
# -----------------------------------------------------------------------------

def test_masking_utilities() -> None:
    """Test brain mask computation, bounding box extraction, and unmasking."""
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[5:15, 5:15, 5:15] = 5.0 # square brain signal

    mask = compute_brain_mask(data, threshold=1.0)
    assert mask.shape == data.shape
    assert np.sum(mask) == 1000

    bbox = extract_bounding_box(mask, margin_voxels=1)
    # 5:15 with margin 1 -> 4:16
    assert bbox[0] == (4, 16)
    assert bbox[1] == (4, 16)
    assert bbox[2] == (4, 16)

    # Apply & unmask
    vec = apply_mask(data, mask)
    assert len(vec) == 1000
    assert np.all(vec == 5.0)

    restored = unmask(vec, mask, fill_value=-1.0)
    assert np.all(restored[mask] == 5.0)
    assert np.all(restored[~mask] == -1.0)


# -----------------------------------------------------------------------------
# Full Integrated Preprocessing Pipeline Tests
# -----------------------------------------------------------------------------

def test_preprocessing_pipeline_execution(tmp_path: Path) -> None:
    """Test end-to-end PreprocessingPipeline with audit tracking and serialization."""
    # 1. Create a sample synthetic 3D statistical NIfTI file
    sample_path = tmp_path / "raw_sub01.nii.gz"
    create_sample_3d_statistical_nifti(
        sample_path, shape=(22, 22, 14), voxel_size=(2.5, 2.5, 2.5), snr=2.5, seed=42
    )

    # 2. Configure pipeline
    cfg = PreprocessingConfig(
        imputation_method="zero",
        normalization_method="zscore",
        denoising_method="gaussian",
        gaussian_fwhm_mm=2.5,
        pad_crop_mode="pad_to_divisible",
        divisible_by=8,
        compute_mask_if_missing=True,
    )
    pipeline = PreprocessingPipeline(config=cfg)

    # 3. Execute
    preproc = pipeline.run(sample_path)
    assert isinstance(preproc, PreprocessedVolume)
    assert all(dim % 8 == 0 for dim in preproc.spatial_shape)
    assert preproc.mask is not None
    assert preproc.data.dtype == np.float32

    # Verify audit metadata
    meta = preproc.preprocessing_metadata
    assert isinstance(meta, PreprocessingMetadata)
    assert meta.original_shape == (22, 22, 14)
    assert len(meta.history) >= 4  # imputation, mask, denoising, normalization, standardization

    # 4. Test conversion to PyTorch 3D tensor for U-Net
    try:
        import torch
        tensor = preproc.to_torch_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 5  # (B, C, D, H, W) -> (1, 1, 24, 24, 16)
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 1
    except ImportError:
        pass

    # 5. Test bundle export to disk
    bundle_dir = tmp_path / "preproc_output"
    saved_files = preproc.save_preprocessed(bundle_dir, prefix="sub01")
    assert saved_files["volume"].exists()
    assert saved_files["metadata"].exists()
    assert "mask" in saved_files and saved_files["mask"].exists()


def test_preprocessing_qc_plot(tmp_path: Path) -> None:
    """Test visual quality-control multi-panel plot generator."""
    sample_path = tmp_path / "qc_input.nii.gz"
    create_sample_3d_statistical_nifti(
        sample_path, shape=(20, 20, 10), snr=3.0, seed=42
    )

    raw_vol = load_nifti(sample_path)
    preproc_vol = preprocess_neuroimaging_volume(
        raw_vol,
        normalization_method="zscore",
        pad_crop_mode="pad_to_divisible",
        divisible_by=8,
    )

    qc_output = tmp_path / "qc_report.png"
    result_path = generate_preprocessing_qc_plot(
        raw_vol, preproc_vol, output_path=qc_output, dpi=150
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Non-empty image file


def test_preprocessing_config_yaml_io(tmp_path: Path) -> None:
    """Test serializing and deserializing PreprocessingConfig from YAML."""
    yaml_path = Path("configs/preprocessing_config.yaml")
    if yaml_path.exists():
        cfg = PreprocessingConfig.from_yaml(yaml_path)
        assert cfg.imputation_method == "zero"
        assert cfg.normalization_method == "zscore"
        assert cfg.divisible_by == 8

    # Test roundtrip
    custom_cfg = PreprocessingConfig(
        imputation_method="mean",
        normalization_method="robust_zscore",
        denoising_method="gaussian",
        gaussian_fwhm_mm=3.5,
        target_spatial_shape=(64, 64, 32),
        pad_crop_mode="center_pad_crop",
    )
    save_file = tmp_path / "custom_preproc.yaml"
    custom_cfg.to_yaml(save_file)
    assert save_file.exists()

    loaded = PreprocessingConfig.from_yaml(save_file)
    assert loaded.imputation_method == "mean"
    assert loaded.normalization_method == "robust_zscore"
    assert loaded.gaussian_fwhm_mm == 3.5
    assert loaded.target_spatial_shape == (64, 64, 32)
    assert loaded.pad_crop_mode == "center_pad_crop"


def test_convert_pvalues_to_zscores() -> None:
    """Test inverse normal CDF transformation from p-values to z-scores."""
    p_vals = np.array([0.5, 0.05, 0.001])
    z_one = convert_pvalues_to_zscores(p_vals, two_tailed=False)
    assert np.isclose(z_one[0], 0.0, atol=1e-3)
    assert z_one[1] > 1.64
    assert z_one[2] > 3.0

    z_two = convert_pvalues_to_zscores(p_vals, two_tailed=True)
    assert z_two[1] > 1.95

