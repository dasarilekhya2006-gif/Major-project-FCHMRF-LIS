"""Phase 5 Verification Test Suite: fcHMRF-LIS Core Statistical Engine.

Validates:
1. Public API exports from top-level fchmrf_lis.
2. Methodological separation: Base Paper (Sun & Cai 2009) vs Proposed DL-Adaptive Prior.
3. Invariant reduction: Adaptive mode collapses identically to Base Paper mode when lambda_dl = 0.
4. Dense bilateral kernel mathematical properties (symmetry, zero diagonal, contrast affinity).
5. Mean-field variational inference convergence and numerical bounds in [0, 1].
6. Spatial neighborhood consensus (spatial coupling beta > 0 boosts borderline activation).
7. Local Index of Significance (LIS) inversion: LIS_i = 1 - Q(theta_i = 1).
8. Step-up spatial False Discovery Rate control guarantee: avg(LIS_rejected) <= alpha.
9. End-to-end synthetic ground-truth simulation: TPR (power) >= 0.80 and bounded empirical FDP.
10. Preservation of 4x4 NIfTI affine coordinate matrix and valid output schemas.
"""

from pathlib import Path
import numpy as np
import pytest

import fchmrf_lis
from fchmrf_lis import (
    FCHMRFModel,
    FCHMRFParameters,
    FCHMRFResult,
    MethodMode,
    compute_dense_bilateral_matrix,
    compute_unary_potentials,
    compute_lis,
    control_spatial_fdr_lis,
    dense_bilateral_message_passing,
    run_dense_mean_field_inference,
)
from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters


class TestPhase5PublicAPI:
    """Verify top-level fchmrf_lis exports."""

    def test_all_expected_symbols_exported(self):
        expected_symbols = [
            "FCHMRFModel",
            "FCHMRFParameters",
            "MethodMode",
            "FCHMRFResult",
            "compute_unary_potentials",
            "compute_dense_bilateral_matrix",
            "dense_bilateral_message_passing",
            "run_dense_mean_field_inference",
            "compute_lis",
            "control_spatial_fdr_lis",
        ]
        for symbol in expected_symbols:
            assert hasattr(fchmrf_lis, symbol), f"Symbol '{symbol}' not found in fchmrf_lis"


class TestMethodologicalDistinction:
    """Verify strict separation between base paper method and proposed extensions."""

    def test_original_mode_forces_zero_lambda_dl(self):
        """Under MethodMode.ORIGINAL_FCHMRF, lambda_dl is strictly 0.0 regardless of input."""
        params = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            lambda_dl=0.85,
        )
        assert params.method_mode == MethodMode.ORIGINAL_FCHMRF
        assert params.lambda_dl == 0.0

    def test_adaptive_mode_allows_calibrated_lambda_dl(self):
        """Under MethodMode.ADAPTIVE_DL_FCHMRF, lambda_dl in [0, 1] is preserved."""
        params = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            lambda_dl=0.45,
        )
        assert params.method_mode == MethodMode.ADAPTIVE_DL_FCHMRF
        assert params.lambda_dl == 0.45

    def test_mathematical_reduction_when_lambda_dl_zero(self):
        """Setting lambda_dl = 0.0 in adaptive mode identically yields original unary energies."""
        z = np.array([-1.5, 0.0, 1.2, 2.8, 4.0])
        mix = TwoGroupMixtureParameters(p0=0.8, mu0=0.0, sigma0=1.0, mu1=2.8, sigma1=1.1)
        dl_prior = np.array([0.05, 0.1, 0.4, 0.8, 0.95])

        u0_base, u1_base = compute_unary_potentials(
            z_scores=z,
            mixture_params=mix,
            method_mode=MethodMode.ORIGINAL_FCHMRF,
        )
        u0_adapt, u1_adapt = compute_unary_potentials(
            z_scores=z,
            mixture_params=mix,
            dl_prior_p1=dl_prior,
            lambda_dl=0.0,
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
        )

        assert np.allclose(u0_base, u0_adapt)
        assert np.allclose(u1_base, u1_adapt)


class TestDenseBilateralAffinity:
    """Verify mathematical properties of the fully connected bilateral kernel."""

    def test_kernel_symmetry_and_zero_diagonal(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [3, 3, 3]], dtype=float)
        z = np.array([2.5, 2.3, 0.1, -1.2])

        K = compute_dense_bilateral_matrix(
            coordinates=coords,
            z_scores=z,
            sigma_spatial=2.0,
            sigma_intensity=1.5,
        )
        assert np.all(np.diag(K) == 0.0)
        assert np.allclose(K, K.T)
        assert np.all(K >= 0.0)
        # Closer voxels with similar z-scores have higher affinity than distant ones
        assert K[0, 1] > K[0, 3]

    def test_message_passing_equivalence(self):
        """Verify chunked message passing matches full matrix multiplication without normalization."""
        np.random.seed(101)
        N = 40
        coords = np.random.rand(N, 3) * 5.0
        z = np.random.randn(N)
        s = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

        K = compute_dense_bilateral_matrix(coords, z)
        expected = K @ s
        actual = dense_bilateral_message_passing(
            coordinates=coords,
            z_scores=z,
            belief_diff=s,
            chunk_size=10,
            normalize=False,
        )
        assert np.allclose(expected, actual, atol=1e-5)


class TestMeanFieldInference:
    """Verify variational coordinate-ascent properties."""

    def test_convergence_and_bounds(self):
        np.random.seed(42)
        N = 30
        coords = np.random.rand(N, 3) * 6.0
        z = np.random.randn(N)
        u0 = np.random.uniform(0.5, 2.5, N)
        u1 = np.random.uniform(0.5, 2.5, N)

        q1, iters, delta = run_dense_mean_field_inference(
            unary_0=u0,
            unary_1=u1,
            coordinates=coords,
            z_scores=z,
            beta=0.7,
            max_iterations=40,
            tolerance=1e-4,
        )
        assert len(q1) == N
        assert np.all(q1 >= 0.0) and np.all(q1 <= 1.0)
        assert iters <= 40
        assert delta <= 1e-3

    def test_spatial_coupling_strengthens_neighborhood_consensus(self):
        """A borderline voxel surrounded by active neighbors increases belief under beta > 0."""
        coords = np.array([
            [1, 1, 0],  # Center
            [0, 1, 0],
            [2, 1, 0],
            [1, 0, 0],
            [1, 2, 0],
        ], dtype=float)
        z = np.array([1.3, 3.8, 4.0, 3.6, 3.9])
        u0 = np.array([2.0, 6.0, 6.0, 6.0, 6.0])
        u1 = np.array([2.0, 0.5, 0.5, 0.5, 0.5])

        q1_no_beta, _, _ = run_dense_mean_field_inference(u0, u1, coords, z, beta=0.0)
        q1_with_beta, _, _ = run_dense_mean_field_inference(u0, u1, coords, z, beta=1.2)

        assert q1_with_beta[0] > q1_no_beta[0]


class TestLISAndFDRControl:
    """Verify LIS computation and Sun & Cai step-up FDR control guarantee."""

    def test_lis_inversion_formula(self):
        q1 = np.array([0.9, 0.1, 0.5, 1.0, 0.0])
        lis = compute_lis(q1)
        expected = np.array([0.1, 0.9, 0.5, 0.0, 1.0])
        assert np.allclose(lis, expected)

    def test_fdr_control_guarantee(self):
        lis = np.array([0.001, 0.004, 0.015, 0.03, 0.07, 0.20, 0.85])
        alpha = 0.05
        rejected, thresh, est_fdr = control_spatial_fdr_lis(lis, alpha=alpha)

        assert np.sum(rejected) > 0
        assert est_fdr <= alpha
        assert np.all(lis[rejected] <= thresh)
        assert np.mean(lis[rejected]) == pytest.approx(est_fdr)

    def test_empty_lis_raises_error(self):
        with pytest.raises(StatisticalInferenceError):
            control_spatial_fdr_lis(np.array([]), alpha=0.05)


class TestSyntheticGroundTruthValidation:
    """Validate spatial sensitivity (power) and empirical FDR control against known ground truth."""

    def test_synthetic_spatial_cluster_detection(self, tmp_path):
        np.random.seed(42)
        shape = (14, 14, 6)
        center = np.array([7, 7, 3])
        ground_truth = np.zeros(shape, dtype=bool)

        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if np.linalg.norm(np.array([x, y, z]) - center) <= 2.8:
                        ground_truth[x, y, z] = True

        n_active = int(np.sum(ground_truth))
        assert n_active >= 15

        # Null voxels ~ N(0, 1), Active cluster ~ N(3.6, 0.7)
        z_map = np.random.randn(*shape)
        z_map[ground_truth] = 3.6 + 0.7 * np.random.randn(n_active)

        custom_affine = np.array([
            [2.0, 0.0, 0.0, -20.0],
            [0.0, 2.0, 0.0, -25.0],
            [0.0, 0.0, 2.5, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        params = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            alpha=0.05,
            beta=0.8,
            max_iterations=25,
        )
        model = FCHMRFModel(params=params)
        result = model.fit_predict(z_scores=z_map, affine=custom_affine)

        assert isinstance(result, FCHMRFResult)
        assert result.lis_map.shape == shape
        assert result.significant_mask.shape == shape
        assert np.allclose(result.affine, custom_affine)

        # Power / Sensitivity (True Positive Rate)
        tp = np.sum(result.significant_mask & ground_truth)
        tpr = tp / n_active
        assert tpr >= 0.80, f"Expected TPR >= 0.80, got {tpr:.3f}"

        # False Discovery Proportion (FDP)
        fp = np.sum(result.significant_mask & (~ground_truth))
        total_discoveries = result.num_discoveries
        fdp = fp / max(1, total_discoveries)
        assert fdp <= 0.15, f"Expected FDP <= 0.15, got {fdp:.3f}"

        # Test NIfTI saving
        saved_paths = result.save_nifti_results(output_dir=tmp_path / "out_p5", prefix="p5_eval")
        assert saved_paths["lis_map"].exists()
        assert saved_paths["significant_mask"].exists()
        assert saved_paths["summary_json"].exists()

    def test_invalid_input_shapes_raise_error(self):
        model = FCHMRFModel()
        with pytest.raises(DataValidationError):
            # 1D z-scores without coordinates
            model.fit_predict(z_scores=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(DataValidationError):
            # 2D unsupported shape
            model.fit_predict(z_scores=np.ones((10, 10)))
