"""Unit and integration test suite for Phase 9: Robustness and Uncertainty Evaluation."""

from pathlib import Path
import pytest
import numpy as np

from src.robustness.metrics import (
    compute_pairwise_jaccard_and_dice,
    compute_detection_probability_map,
    compute_voxel_uncertainty_entropy,
    evaluate_repetition_stability,
    StabilityMetrics,
)
from src.robustness.perturbations import (
    apply_additive_voxel_perturbation,
    apply_prior_misalignment_perturbation,
    generate_snr_sweep_dataset,
)
from src.robustness.evaluator import (
    RobustnessEvaluator,
    RobustnessEvaluationReport,
)
from src.robustness.visualization import plot_robustness_analysis_summary
from src.evaluation.synthetic_benchmarks import generate_synthetic_benchmark_volume


class TestStabilityAndUncertaintyMetrics:
    """Test pairwise overlap metrics, probability maps, and Shannon entropy calculations."""

    def test_pairwise_jaccard_identical_masks(self):
        m1 = np.zeros((10, 10, 5), dtype=bool)
        m1[2:5, 2:5, 1:3] = True
        masks = [m1.copy(), m1.copy(), m1.copy()]

        j_mean, j_std, d_mean = compute_pairwise_jaccard_and_dice(masks)
        assert j_mean == 1.0
        assert j_std == 0.0
        assert d_mean == 1.0

    def test_pairwise_jaccard_disjoint_masks(self):
        m1 = np.zeros((10, 10, 5), dtype=bool)
        m2 = np.zeros((10, 10, 5), dtype=bool)
        m1[1:3, 1:3, :] = True
        m2[6:8, 6:8, :] = True

        j_mean, _, d_mean = compute_pairwise_jaccard_and_dice([m1, m2])
        assert j_mean == 0.0
        assert d_mean == 0.0

    def test_detection_probability_and_entropy(self):
        # 4 masks with predictable overlaps
        m1 = np.array([1, 1, 0, 0], dtype=bool)
        m2 = np.array([1, 0, 0, 0], dtype=bool)
        m3 = np.array([1, 1, 0, 0], dtype=bool)
        m4 = np.array([1, 0, 0, 0], dtype=bool)

        prob_map = compute_detection_probability_map([m1, m2, m3, m4])
        # Indices: [0]: p=1.0, [1]: p=0.5, [2]: p=0.0, [3]: p=0.0
        assert prob_map[0] == 1.0
        assert prob_map[1] == 0.5
        assert prob_map[2] == 0.0

        ent_map, mean_ent = compute_voxel_uncertainty_entropy(prob_map)
        # Shannon entropy at p=0.5 must be exactly 1.0 bit
        assert pytest.approx(ent_map[1], abs=1e-3) == 1.0
        # Entropy at p=1.0 and p=0.0 must be near zero
        assert ent_map[0] < 1e-4
        assert ent_map[2] < 1e-4


class TestPerturbations:
    """Test input noise jitter and prior misalignment transformations."""

    def test_additive_voxel_perturbation(self):
        vol = generate_synthetic_benchmark_volume(shape=(12, 12, 6), snr=3.0, seed=42)
        pert_vol = apply_additive_voxel_perturbation(volume=vol, noise_sigma=0.3, seed=99)

        assert pert_vol.shape == vol.shape
        # Differences should exist inside the spatial mask
        diff = np.abs(pert_vol.z_scores - vol.z_scores)
        assert np.max(diff[vol.spatial_mask]) > 0.05
        # Outside brain mask must remain untouched
        assert np.all(diff[~vol.spatial_mask] == 0.0)

    def test_prior_misalignment(self):
        vol = generate_synthetic_benchmark_volume(shape=(12, 12, 6), snr=3.0, seed=42)
        shifted_vol = apply_prior_misalignment_perturbation(vol, translation_voxels=(2, 0, 0))

        assert shifted_vol.dl_prior.shape == vol.dl_prior.shape
        assert not np.array_equal(shifted_vol.dl_prior, vol.dl_prior)


class TestRobustnessEvaluatorSuite:
    """Test execution of multi-dimensional robustness sweeps and artifact generation."""

    def test_fast_robustness_evaluator_run(self, tmp_path):
        evaluator = RobustnessEvaluator(
            nominal_alpha=0.05,
            default_beta=0.7,
            default_lambda_dl=0.5,
            max_iterations=4,
        )

        shape = (10, 10, 5)
        # Fast mini-sweeps
        report = evaluator.run_full_robustness_evaluation(
            snr_levels=[2.0, 3.5],
            perturbation_sigmas=[0.0, 0.2],
            spatial_sigmas=[0.0, 1.0],
            lambda_dl_values=[0.3, 0.7],
            num_seeds=3,
            shape=shape,
        )

        assert isinstance(report, RobustnessEvaluationReport)
        assert len(report.snr_sweep_metrics) == 2
        assert len(report.perturbation_sweep_metrics) == 2
        assert len(report.spatial_corr_metrics) == 2
        assert len(report.hyperparam_sweep_metrics) == 2
        assert report.multi_seed_metrics.num_repetitions == 3

        # Test CSV export
        csv_file = tmp_path / "robustness.csv"
        report.export_csv(csv_file)
        assert csv_file.exists()
        assert csv_file.stat().st_size > 500

        # Test Markdown rendering
        md_text = report.to_markdown_table()
        assert "SNR Sensitivity Sweep" in md_text
        assert "Input Data Perturbation Sweep" in md_text
        assert "Multi-Seed Uncertainty" in md_text

        # Test visualization
        fig_file = tmp_path / "robustness.png"
        out_fig = plot_robustness_analysis_summary(report, output_path=fig_file, dpi=80)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 10000
