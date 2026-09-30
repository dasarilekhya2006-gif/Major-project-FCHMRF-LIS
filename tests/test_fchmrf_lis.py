"""Comprehensive unit and integration test suite for Phase 5: fcHMRF-LIS Spatial Statistical Modeling."""

from pathlib import Path
import pytest
import numpy as np

from src.core.exceptions import StatisticalInferenceError
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.dense_kernel import (
    compute_dense_bilateral_matrix,
    dense_bilateral_message_passing,
)
from src.fchmrf_lis.graph_potentials import compute_unary_potentials
from src.fchmrf_lis.mean_field import run_dense_mean_field_inference
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult
from src.fchmrf_lis.visualization import plot_fchmrf_summary
from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters


class TestFCHMRFParametersAndModes:
    """Test parameter validation and strict original vs adaptive mode distinction."""

    def test_original_fchmrf_enforces_zero_lambda_dl(self):
        """Under ORIGINAL_FCHMRF mode, lambda_dl must be strictly 0.0."""
        params = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            lambda_dl=0.75,  # Ignored and forced to 0.0
        )
        assert params.method_mode == MethodMode.ORIGINAL_FCHMRF
        assert params.lambda_dl == 0.0

    def test_adaptive_dl_fchmrf_preserves_lambda_dl(self):
        """Under ADAPTIVE_DL_FCHMRF mode, lambda_dl in [0, 1] is retained."""
        params = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            lambda_dl=0.6,
        )
        assert params.method_mode == MethodMode.ADAPTIVE_DL_FCHMRF
        assert params.lambda_dl == 0.6


class TestUnaryPotentials:
    """Mathematical tests for null and active unary energies."""

    def test_unary_energy_ordering_under_activation(self):
        """A strong positive z-score must produce lower active unary energy psi(1) < psi(0)."""
        z_scores = np.array([4.5, -0.5, 0.0])
        mix_params = TwoGroupMixtureParameters(p0=0.85, mu0=0.0, sigma0=1.0, mu1=3.0, sigma1=1.0)

        u0, u1 = compute_unary_potentials(
            z_scores=z_scores,
            mixture_params=mix_params,
            method_mode=MethodMode.ORIGINAL_FCHMRF,
        )

        # For strong positive z = 4.5, active state 1 must be heavily favored (u1 < u0)
        assert u1[0] < u0[0]
        # For null z = 0.0, null state 0 must be favored (u0 < u1)
        assert u0[2] < u1[2]

    def test_adaptive_dl_identical_to_original_when_lambda_zero(self):
        """When lambda_dl = 0.0, adaptive DL mode must yield identical energies to original mode."""
        z_scores = np.linspace(-2.0, 5.0, 30)
        mix_params = TwoGroupMixtureParameters()
        dl_prior = np.random.uniform(0.0, 1.0, len(z_scores))

        u0_orig, u1_orig = compute_unary_potentials(
            z_scores=z_scores,
            mixture_params=mix_params,
            method_mode=MethodMode.ORIGINAL_FCHMRF,
        )

        u0_adapt, u1_adapt = compute_unary_potentials(
            z_scores=z_scores,
            mixture_params=mix_params,
            dl_prior_p1=dl_prior,
            lambda_dl=0.0,
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
        )

        assert np.allclose(u0_orig, u0_adapt)
        assert np.allclose(u1_orig, u1_adapt)


class TestDenseBilateralKernel:
    """Mathematical tests for the dense fully connected kernel."""

    def test_dense_kernel_properties(self):
        """Verify symmetry, zero diagonal, and non-negativity."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float)
        z = np.array([2.0, 2.1, 1.9, -0.5])

        K = compute_dense_bilateral_matrix(
            coordinates=coords,
            z_scores=z,
            sigma_spatial=2.0,
            sigma_intensity=1.0,
        )

        # 1. Zero diagonal
        assert np.all(np.diag(K) == 0.0)
        # 2. Symmetric
        assert np.allclose(K, K.T)
        # 3. Non-negative
        assert np.all(K >= 0.0)
        # 4. Nearby voxels with similar z have higher weight than distant voxels
        assert K[0, 1] > K[0, 3]

    def test_dense_message_passing_matches_matrix_product(self):
        """Verify chunked message passing produces identical results to full matrix multiplication."""
        np.random.seed(42)
        N = 50
        coords = np.random.rand(N, 3) * 10.0
        z = np.random.randn(N)
        s = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

        K = compute_dense_bilateral_matrix(coords, z)
        expected_m = K @ s

        actual_m = dense_bilateral_message_passing(
            coordinates=coords,
            z_scores=z,
            belief_diff=s,
            chunk_size=16,
        )

        assert np.allclose(expected_m, actual_m, atol=1e-5)


class TestDenseMeanFieldInference:
    """Tests for variational inference convergence and spatial context smoothing."""

    def test_mean_field_bounds_and_convergence(self):
        """Verify marginal probabilities are bounded in [0, 1] and converge within tolerance."""
        np.random.seed(42)
        N = 60
        coords = np.random.rand(N, 3) * 8.0
        z = np.random.randn(N)
        u0 = np.random.uniform(0.5, 3.0, N)
        u1 = np.random.uniform(0.5, 3.0, N)

        q1, iters, delta = run_dense_mean_field_inference(
            unary_0=u0,
            unary_1=u1,
            coordinates=coords,
            z_scores=z,
            beta=0.8,
            max_iterations=50,
            tolerance=1e-4,
        )

        assert len(q1) == N
        assert np.all((q1 >= 0.0) & (q1 <= 1.0))
        assert iters <= 50
        assert delta <= 1e-3

    def test_spatial_coupling_boosts_isolated_neighbor(self):
        """A borderline voxel surrounded by strong active neighbors has higher Q(1) under beta > 0."""
        # 5 voxels in a cross: center voxel is (1, 1, 0), neighbors at (0, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0)
        coords = np.array([
            [1, 1, 0],  # Center (borderline)
            [0, 1, 0],  # Active neighbor
            [2, 1, 0],  # Active neighbor
            [1, 0, 0],  # Active neighbor
            [1, 2, 0],  # Active neighbor
        ], dtype=float)

        z = np.array([1.2, 3.5, 3.8, 3.6, 3.7])
        # Unary energies: center has u0 ~= u1 (neutral)
        u0 = np.array([2.0, 5.0, 5.0, 5.0, 5.0])
        u1 = np.array([2.0, 0.5, 0.5, 0.5, 0.5])

        # Inference without spatial coupling (beta = 0)
        q1_uncoupled, _, _ = run_dense_mean_field_inference(
            u0, u1, coords, z, beta=0.0
        )

        # Inference with positive spatial coupling (beta = 1.5)
        q1_coupled, _, _ = run_dense_mean_field_inference(
            u0, u1, coords, z, beta=1.5
        )

        # Center voxel should have higher active belief due to spatial neighborhood consensus
        assert q1_coupled[0] > q1_uncoupled[0]


class TestLISAndSpatialFDR:
    """Tests for LIS inversion and Sun & Cai step-up FDR control."""

    def test_lis_inversion(self):
        """Verify LIS = 1 - Q(1)."""
        q1 = np.array([0.95, 0.05, 0.50, 1.0, 0.0])
        lis = compute_lis(q1)
        expected = np.array([0.05, 0.95, 0.50, 0.0, 1.0])
        assert np.allclose(lis, expected)

    def test_spatial_fdr_step_up_guarantee(self):
        """Cumulative mean LIS of rejections must not exceed alpha."""
        lis = np.array([0.001, 0.005, 0.02, 0.03, 0.08, 0.25, 0.70, 0.95])
        alpha = 0.05

        rejected, lis_thresh, est_fdr = control_spatial_fdr_lis(lis, alpha=alpha)

        assert np.sum(rejected) > 0
        assert est_fdr <= alpha
        assert np.all(lis[rejected] <= lis_thresh)
        assert np.mean(lis[rejected]) == pytest.approx(est_fdr)


class TestFCHMRFModelSyntheticValidation:
    """End-to-end model validation against synthetic neuroimaging ground truth."""

    def test_fit_predict_synthetic_ground_truth(self, tmp_path):
        """Validate detection sensitivity and FDR control on known synthetic cluster."""
        np.random.seed(42)
        shape = (16, 16, 8)
        total_voxels = np.prod(shape)

        # Generate ground truth active sphere at center
        ground_truth = np.zeros(shape, dtype=bool)
        center = np.array([8, 8, 4])
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if np.linalg.norm(np.array([x, y, z]) - center) <= 3.2:
                        ground_truth[x, y, z] = True

        n_active_true = int(np.sum(ground_truth))
        assert n_active_true > 20

        # Synthesize z-scores: Null ~ N(0, 1), Active ~ N(3.5, 0.8)
        z_data = np.random.randn(*shape)
        z_data[ground_truth] = 3.5 + 0.8 * np.random.randn(n_active_true)

        # 1. Fit original fcHMRF-LIS
        params = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            alpha=0.05,
            beta=0.8,
            max_iterations=30,
        )
        model = FCHMRFModel(params=params)
        result = model.fit_predict(z_scores=z_data)

        assert isinstance(result, FCHMRFResult)
        assert result.lis_map.shape == shape
        assert result.significant_mask.shape == shape
        assert result.num_discoveries > 0

        # True Positive Rate (Sensitivity) on known ground truth active cluster
        tpr = np.sum(result.significant_mask & ground_truth) / n_active_true
        assert tpr >= 0.80  # Detects at least 80% of true ground truth cluster

        # False Discovery Proportion (FDP) on known ground truth null voxels
        false_discoveries = np.sum(result.significant_mask & (~ground_truth))
        fdp = false_discoveries / max(1, result.num_discoveries)
        # Empirical FDR should be well controlled near nominal level 0.05
        assert fdp <= 0.15

        # 2. Test saving aligned NIfTI maps and metadata
        out_dir = tmp_path / "fchmrf_out"
        saved = result.save_nifti_results(output_dir=out_dir, prefix="test_sim")
        assert saved["lis_map"].exists()
        assert saved["significant_mask"].exists()
        assert saved["posterior_prob"].exists()
        assert saved["summary_json"].exists()

        # 3. Test summary visualization
        out_fig = tmp_path / "fchmrf_fig.png"
        fig_path = plot_fchmrf_summary(result=result, z_scores=z_data, output_path=out_fig, dpi=100)
        assert fig_path.exists()
        assert fig_path.stat().st_size > 5000
