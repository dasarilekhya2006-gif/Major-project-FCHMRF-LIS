"""Phase 4 Verification Test Suite: Voxel-Level Statistical Testing.

Validates:
  - Inputs: Neuroimaging volumetric data (3D and 4D), ExperimentalDesign object, and optional ROI masks.
  - Outputs: Voxel-level test statistics, p-values, effect-size measures (Cohen's d, beta contrast), and aligned NIfTI maps.
  - Experimental design integrity: Stops and raises informative errors when required design information is missing or unsupported.
  - Robust NaN and Infinite value handling: Supports 'mask_out', 'zero', and 'raise' policies.
  - Synthetic spatial benchmark validation: Verifies ground-truth activation detection.
  - Geometric affine preservation across all exported statistical NIfTI volumes.
"""

from pathlib import Path
import numpy as np
import pytest

from statistics import (
    AlternativeHypothesis,
    ExperimentalDesign,
    StatisticalMaps,
    TestType,
    compute_statistics,
    compute_voxel_statistics,
    create_glm_design,
    create_one_sample_design,
    create_two_sample_design,
    validate_statistical_inputs,
)
from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.data.loaders import load_nifti, save_nifti
from src.data.schemas import NeuroimagingVolume


@pytest.fixture
def synthetic_4d_neuroimaging_dataset():
    """Generates synthetic 4D neuroimaging data (12x12x8 spatial grid, 12 subjects) with known ground truth."""
    np.random.seed(42)
    shape = (12, 12, 8)
    n_samples = 12
    # Baseline noise volume: N(0, 1)
    data = np.random.randn(*shape, n_samples).astype(np.float64)

    # True activated cluster in central region [4:8, 4:8, 3:6] with effect size delta = 2.5
    true_active_mask = np.zeros(shape, dtype=bool)
    true_active_mask[4:8, 4:8, 3:6] = True
    data[true_active_mask, :] += 2.5

    affine = np.array([
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -120.0],
        [0.0, 0.0, 2.5, -50.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return {
        "data": data,
        "true_active_mask": true_active_mask,
        "spatial_shape": shape,
        "n_samples": n_samples,
        "affine": affine,
    }


def test_voxel_statistics_outputs(synthetic_4d_neuroimaging_dataset):
    """Verify that compute_voxel_statistics outputs test statistics, p-values, effect sizes, and z-scores."""
    ds = synthetic_4d_neuroimaging_dataset
    design = create_one_sample_design(num_samples=ds["n_samples"], popmean=0.0)

    stat_maps = compute_voxel_statistics(
        data=ds["data"],
        design=design,
        affine=ds["affine"],
    )

    assert isinstance(stat_maps, StatisticalMaps)
    assert stat_maps.spatial_shape == ds["spatial_shape"]
    assert np.allclose(stat_maps.affine, ds["affine"])

    # 1. Test statistics
    assert stat_maps.test_statistic.shape == ds["spatial_shape"]
    assert not np.any(np.isnan(stat_maps.test_statistic))

    # 2. P-values in [0, 1]
    assert stat_maps.p_values.shape == ds["spatial_shape"]
    assert np.all(stat_maps.p_values >= 0.0)
    assert np.all(stat_maps.p_values <= 1.0)

    # 3. Effect size information (Cohen's d for one-sample t-test)
    assert stat_maps.effect_sizes.shape == ds["spatial_shape"]
    assert not np.any(np.isnan(stat_maps.effect_sizes))

    # 4. Standard normal z-scores
    assert stat_maps.z_scores.shape == ds["spatial_shape"]
    assert not np.any(np.isnan(stat_maps.z_scores))

    # Known activated voxels should show strong test statistics and small p-values
    active_idx = ds["true_active_mask"]
    assert np.mean(stat_maps.test_statistic[active_idx]) > 4.0
    assert np.mean(stat_maps.p_values[active_idx]) < 0.01
    assert np.mean(stat_maps.effect_sizes[active_idx]) > 1.5


def test_optional_roi_mask_restricts_computation(synthetic_4d_neuroimaging_dataset):
    """Verify that providing an optional ROI mask confines valid hypothesis testing to that ROI."""
    ds = synthetic_4d_neuroimaging_dataset
    shape = ds["spatial_shape"]

    # Define an ROI mask covering half of the volume
    roi_mask = np.zeros(shape, dtype=bool)
    roi_mask[:6, :, :] = True
    num_roi_voxels = int(np.sum(roi_mask))

    stat_maps = compute_voxel_statistics(
        data=ds["data"],
        roi_mask=roi_mask,
        test_type="one_sample_t",
        affine=ds["affine"],
    )

    assert stat_maps.num_active_voxels == num_roi_voxels

    # Voxels outside the ROI mask must be unperturbed null
    assert np.all(stat_maps.p_values[~roi_mask] == 1.0)
    assert np.all(stat_maps.z_scores[~roi_mask] == 0.0)
    assert np.all(stat_maps.test_statistic[~roi_mask] == 0.0)
    assert np.all(stat_maps.effect_sizes[~roi_mask] == 0.0)

    # Voxels inside the ROI mask must have valid p-values in [0, 1]
    in_roi_p = stat_maps.p_values[roi_mask]
    assert np.all((in_roi_p >= 0.0) & (in_roi_p <= 1.0))


def test_missing_design_information_stops_with_error(synthetic_4d_neuroimaging_dataset):
    """Ensure the system DOES NOT assume an unsupported design when information is missing."""
    ds = synthetic_4d_neuroimaging_dataset

    # 1. Two-sample t-test requested without design_matrix -> Must stop and raise error
    with pytest.raises(StatisticalInferenceError, match="Missing required design information"):
        compute_voxel_statistics(
            data=ds["data"],
            test_type=TestType.TWO_SAMPLE_T,
            design_matrix=None,
        )

    # 2. GLM requested without contrast -> Must stop and raise error
    with pytest.raises(StatisticalInferenceError, match="Missing required design information"):
        compute_voxel_statistics(
            data=ds["data"],
            test_type=TestType.GLM,
            design_matrix=np.column_stack([np.ones(ds["n_samples"]), np.arange(ds["n_samples"])]),
            contrast=None,
        )

    # 3. 3D data supplied for one-sample t-test (which requires sample replicates) -> Must raise error
    single_vol_3d = ds["data"][..., 0]
    with pytest.raises(StatisticalInferenceError, match="requires a 4D volume"):
        compute_voxel_statistics(
            data=single_vol_3d,
            test_type=TestType.ONE_SAMPLE_T,
        )


def test_nan_and_inf_handling(synthetic_4d_neuroimaging_dataset):
    """Verify NaN/Inf detection and controlled handling under different policies."""
    ds = synthetic_4d_neuroimaging_dataset
    corrupted_data = ds["data"].copy()

    # Inject NaN and Inf into specific voxels
    corrupted_data[2, 2, 2, 0] = np.nan
    corrupted_data[3, 3, 3, 1] = np.inf
    corrupted_data[4, 4, 4, 2] = -np.inf

    # Policy 1: 'raise' -> Must immediately raise DataValidationError
    with pytest.raises(DataValidationError, match="non-finite"):
        compute_voxel_statistics(
            data=corrupted_data,
            nan_policy="raise",
        )

    # Policy 2: 'mask_out' -> Must cleanly exclude corrupted voxels from the active analysis mask
    stat_maps = compute_voxel_statistics(
        data=corrupted_data,
        nan_policy="mask_out",
        affine=ds["affine"],
    )
    assert not stat_maps.mask[2, 2, 2]
    assert not stat_maps.mask[3, 3, 3]
    assert not stat_maps.mask[4, 4, 4]

    # No NaN or Inf should propagate into outputs
    assert np.all(np.isfinite(stat_maps.test_statistic))
    assert np.all(np.isfinite(stat_maps.p_values))
    assert np.all(np.isfinite(stat_maps.z_scores))
    assert np.all(np.isfinite(stat_maps.effect_sizes))

    # Masked out voxels receive neutral null values
    assert stat_maps.p_values[2, 2, 2] == 1.0
    assert stat_maps.z_scores[2, 2, 2] == 0.0


def test_nifti_map_saving_and_affine_preservation(synthetic_4d_neuroimaging_dataset, tmp_path):
    """Verify that statistical maps are saved as NIfTI volumes preserving orientation and affine."""
    ds = synthetic_4d_neuroimaging_dataset
    stat_maps = compute_voxel_statistics(
        data=ds["data"],
        test_type="one_sample_t",
        affine=ds["affine"],
    )

    out_dir = tmp_path / "stat_nifti"
    saved = stat_maps.save_nifti_maps(output_dir=out_dir, prefix="test_experiment")

    expected_keys = ["test_stat", "p_values", "z_scores", "effect_sizes", "neg_log10_p", "analysis_mask"]
    for k in expected_keys:
        assert k in saved
        assert saved[k].exists()

        # Load back NIfTI volume and verify geometry preserving on-disk affine
        vol = load_nifti(saved[k], canonical_reorient=False)
        assert vol.shape == ds["spatial_shape"]
        assert np.allclose(vol.affine, ds["affine"])
        assert vol.data.dtype == np.float32
