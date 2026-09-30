"""Comprehensive unit and integration test suite for Phase 4: Voxel-Level Statistical Testing."""

from pathlib import Path
import pytest
import numpy as np

from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.data.schemas import NeuroimagingVolume
from src.data.sample_data import create_sample_3d_statistical_nifti
from src.statistical_analysis.design import (
    AlternativeHypothesis,
    ExperimentalDesign,
    TestType,
    create_glm_design,
    create_one_sample_design,
    create_paired_design,
    create_two_sample_design,
)
from src.statistical_analysis.validation import validate_statistical_inputs
from src.statistical_analysis.voxel_testing import (
    StatisticalMaps,
    compute_statistics,
    glm_voxelwise,
    one_sample_t_test,
    two_sample_t_test,
)
from src.statistical_analysis.visualization import plot_statistical_maps


class TestStatisticalDesign:
    """Tests for experimental design construction and contrast validation."""

    def test_one_sample_design(self):
        """Verify one-sample design matrix has 1 column of ones and contrast [1.0]."""
        design = create_one_sample_design(num_samples=10, popmean=0.0)
        assert design.test_type == TestType.ONE_SAMPLE_T
        assert design.design_matrix.shape == (10, 1)
        assert np.all(design.design_matrix == 1.0)
        assert np.array_equal(design.contrast, [1.0])

    def test_two_sample_design(self):
        """Verify two-sample design matrix partitions groups with [1.0, -1.0] contrast."""
        design = create_two_sample_design(num_group1=6, num_group2=6)
        assert design.test_type == TestType.TWO_SAMPLE_T
        assert design.design_matrix.shape == (12, 2)
        assert np.sum(design.design_matrix[:, 0]) == 6.0
        assert np.sum(design.design_matrix[:, 1]) == 6.0
        assert np.array_equal(design.contrast, [1.0, -1.0])

    def test_contrast_dimension_mismatch_raises_error(self):
        """Contrast vector length must match number of regressors."""
        with pytest.raises(StatisticalInferenceError, match="Contrast length"):
            ExperimentalDesign(
                test_type=TestType.GLM,
                design_matrix=np.ones((10, 3)),
                contrast=np.array([1.0, 0.0]),  # 2 != 3
            )


class TestStatisticalInputValidation:
    """Tests for NaN/Inf detection, rank checks, and dimension validation."""

    def test_invalid_dimension_raises_error(self):
        """2D array where 3D/4D expected must raise DataValidationError."""
        data_2d = np.zeros((10, 10))
        with pytest.raises(DataValidationError, match="expects 3D or 4D"):
            validate_statistical_inputs(data_2d)

    def test_nan_policy_raise(self):
        """nan_policy='raise' raises DataValidationError upon encountering NaN/Inf."""
        data_nan = np.ones((8, 8, 8, 4))
        data_nan[2, 2, 2, 1] = np.nan
        with pytest.raises(DataValidationError, match="non-finite"):
            validate_statistical_inputs(data_nan, nan_policy="raise")

    def test_nan_policy_mask_out(self):
        """nan_policy='mask_out' removes corrupted voxels from active analysis mask."""
        data_nan = np.ones((8, 8, 8, 4))
        data_nan[2, 2, 2, 1] = np.nan
        cleaned, mask = validate_statistical_inputs(data_nan, nan_policy="mask_out")
        assert not mask[2, 2, 2]
        assert np.all(mask[0, 0, :])

    def test_rank_deficient_design_matrix_raises_error(self):
        """Collinear regressors in design matrix must raise StatisticalInferenceError."""
        # Column 1 and Column 2 are identical -> rank deficient
        X = np.ones((10, 2))
        data = np.random.randn(8, 8, 8, 10)
        with pytest.raises(StatisticalInferenceError, match="rank deficient"):
            validate_statistical_inputs(data, design_matrix=X)

    def test_roi_mask_dimension_mismatch_raises_error(self):
        """Spatial shape mismatch between volume and ROI mask must raise DataValidationError."""
        data = np.random.randn(10, 10, 8, 5)
        bad_mask = np.ones((12, 12, 8), dtype=bool)
        with pytest.raises(DataValidationError, match="spatial shape"):
            validate_statistical_inputs(data, roi_mask=bad_mask)


class TestVoxelLevelHypothesisTesting:
    """Unit tests for individual statistical test procedures."""

    def test_one_sample_t_test_accuracy(self):
        """Verify one_sample_t_test matches theoretical t and p values for synthetic cluster."""
        np.random.seed(42)
        shape = (10, 10, 8)
        n_samples = 15
        data = np.random.randn(*shape, n_samples)

        # Inject strong positive activation at voxel (5, 5, 4)
        true_mean = 3.0
        data[5, 5, 4, :] = true_mean + 0.5 * np.random.randn(n_samples)

        t_map, p_map, z_map, d_map, dof = one_sample_t_test(
            data, popmean=0.0, alternative=AlternativeHypothesis.GREATER
        )

        assert dof == 14.0
        assert t_map[5, 5, 4] > 8.0
        assert p_map[5, 5, 4] < 1e-5
        assert z_map[5, 5, 4] > 4.0
        assert d_map[5, 5, 4] > 2.0

    def test_two_sample_t_test_accuracy(self):
        """Verify two-sample Welch's t-test comparing two distinct groups."""
        np.random.seed(42)
        shape = (8, 8, 6)
        n1, n2 = 10, 10
        g1 = np.random.randn(*shape, n1)
        g2 = np.random.randn(*shape, n2)

        # Group 1 has strong activation at (3, 3, 3), Group 2 has 0 mean
        g1[3, 3, 3, :] += 4.0

        t_map, p_map, z_map, d_map, dof = two_sample_t_test(
            g1, g2, equal_var=False, alternative=AlternativeHypothesis.GREATER
        )

        assert t_map[3, 3, 3] > 5.0
        assert p_map[3, 3, 3] < 1e-4
        assert z_map[3, 3, 3] > 3.5

    def test_glm_voxelwise_estimation(self):
        """Verify voxel-wise OLS GLM correctly estimates contrast slope."""
        np.random.seed(42)
        shape = (6, 6, 4)
        n_samples = 20

        # Regressor: age/dose covariate
        covariate = np.linspace(-1.0, 1.0, n_samples)
        X = np.column_stack([np.ones(n_samples), covariate])
        contrast = np.array([0.0, 1.0])  # Test positive slope of covariate

        # Construct data where voxel (2, 2, 2) has slope 3.0
        data = np.random.randn(*shape, n_samples) * 0.2
        for i in range(n_samples):
            data[2, 2, 2, i] += 3.0 * covariate[i]

        t_map, p_map, z_map, eff_map, dof = glm_voxelwise(
            data, design_matrix=X, contrast=contrast, alternative=AlternativeHypothesis.GREATER
        )

        assert dof == 18.0
        assert t_map[2, 2, 2] > 10.0
        assert p_map[2, 2, 2] < 1e-6
        assert eff_map[2, 2, 2] == pytest.approx(3.0, rel=0.1)


class TestComputeStatisticsMasterPipeline:
    """Integration tests for compute_statistics with ROI masking and NIfTI export."""

    def test_compute_statistics_with_roi_mask(self, tmp_path):
        """Verify compute_statistics restricts testing to active ROI voxels."""
        shape = (12, 12, 8)
        n_samples = 10
        data = np.random.randn(*shape, n_samples)

        # Create sparse ROI mask (only central 4x4x4 cube active)
        roi_mask = np.zeros(shape, dtype=bool)
        roi_mask[4:8, 4:8, 2:6] = True

        custom_affine = np.array([
            [-2.0, 0.0, 0.0, 60.0],
            [0.0, 2.0, 0.0, -80.0],
            [0.0, 0.0, 2.0, -30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        stat_maps = compute_statistics(
            data=data,
            roi_mask=roi_mask,
            test_type="one_sample_t",
            alternative="greater",
            affine=custom_affine,
        )

        assert isinstance(stat_maps, StatisticalMaps)
        assert stat_maps.spatial_shape == shape
        assert stat_maps.num_active_voxels == 4 * 4 * 4  # 64 active voxels
        assert np.allclose(stat_maps.affine, custom_affine)

        # Out-of-mask voxels must be unperturbed null (p=1.0, z=0.0, t=0.0)
        assert np.all(stat_maps.p_values[~roi_mask] == 1.0)
        assert np.all(stat_maps.z_scores[~roi_mask] == 0.0)
        assert np.all(stat_maps.test_statistic[~roi_mask] == 0.0)

        # In-mask voxels must have valid statistics
        in_mask_p = stat_maps.p_values[roi_mask]
        assert np.all((in_mask_p >= 0.0) & (in_mask_p <= 1.0))

    def test_save_nifti_maps_preserves_affine(self, tmp_path):
        """Verify save_nifti_maps exports aligned NIfTI files with preserved affine."""
        shape = (8, 8, 6)
        data = np.random.randn(*shape, 8)
        custom_affine = np.diag([2.5, 2.5, 3.0, 1.0])

        stat_maps = compute_statistics(
            data=data,
            test_type="one_sample_t",
            affine=custom_affine,
        )

        out_dir = tmp_path / "nifti_maps"
        saved = stat_maps.save_nifti_maps(output_dir=out_dir, prefix="sub_group")

        assert len(saved) >= 5
        assert saved["z_scores"].exists()
        assert saved["p_values"].exists()
        assert saved["test_stat"].exists()

        # Load back one map to verify affine
        from src.data.loaders import load_nifti
        loaded_vol = load_nifti(saved["z_scores"])
        assert loaded_vol.shape == shape
        assert np.allclose(loaded_vol.affine, custom_affine)

    def test_to_fchmrf_inputs_compatibility(self):
        """Verify to_fchmrf_inputs formats inputs correctly for Phase 5."""
        shape = (10, 10, 6)
        data = np.random.randn(*shape, 10)
        roi_mask = np.zeros(shape, dtype=bool)
        roi_mask[3:7, 3:7, 2:4] = True  # 4 * 4 * 2 = 32 active voxels

        stat_maps = compute_statistics(
            data=data,
            roi_mask=roi_mask,
            test_type="one_sample_t",
        )

        fchmrf_dict = stat_maps.to_fchmrf_inputs(restrict_to_mask=True)
        assert "z_scores" in fchmrf_dict
        assert "spatial_mask" in fchmrf_dict
        assert "voxel_coordinates" in fchmrf_dict
        assert "p_values" in fchmrf_dict

        # Should be 1D vectors with length matching active ROI voxels (32)
        assert fchmrf_dict["z_scores"].shape == (32,)
        assert fchmrf_dict["p_values"].shape == (32,)
        assert fchmrf_dict["voxel_coordinates"].shape == (32, 3)
        assert np.all(np.isfinite(fchmrf_dict["z_scores"]))


class TestStatisticalVisualization:
    """Tests for multi-panel orthogonal slice figure generation."""

    def test_plot_statistical_maps(self, tmp_path):
        """Verify plot_statistical_maps generates publication-grade 3-row orthogonal figure."""
        shape = (16, 16, 10)
        data = np.random.randn(*shape, 12)
        roi_mask = np.zeros(shape, dtype=bool)
        roi_mask[4:12, 4:12, 3:7] = True

        stat_maps = compute_statistics(
            data=data,
            roi_mask=roi_mask,
            test_type="one_sample_t",
        )

        out_png = tmp_path / "stat_summary.png"
        res_path = plot_statistical_maps(
            stat_maps=stat_maps,
            output_path=out_png,
            roi_mask=roi_mask,
            dpi=100,
        )

        assert res_path.exists()
        assert res_path.stat().st_size > 5000
