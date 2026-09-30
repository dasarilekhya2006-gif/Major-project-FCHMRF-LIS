"""Unit and integration test suite for Phase 8: Rigorous Experimental Comparison."""

from pathlib import Path
import pytest
import numpy as np

from src.evaluation.metrics import (
    calculate_metrics,
    compute_centroid_distance,
    aggregate_metrics_across_trials,
    EvaluationMetrics,
)
from src.evaluation.baselines import (
    BenjaminiHochbergCorrection,
    LocalFDRCorrection,
    OriginalFCHMRFLISCorrection,
    ROIAssistedFCHMRFLISCorrection,
    AdaptiveROIAssistedFCHMRFLISCorrection,
)
from src.evaluation.synthetic_benchmarks import (
    generate_synthetic_benchmark_volume,
    generate_monte_carlo_dataset,
)
from src.evaluation.comparative_runner import (
    ComparativeExperimentRunner,
    BenchmarkComparisonReport,
)
from src.evaluation.visualization import plot_comparative_benchmark_summary


class TestMetricsCalculation:
    """Test performance, classification, overlap, and localization metric calculations."""

    def test_perfect_classification(self):
        """A perfect prediction must yield TPR=1, FDP=0, F1=1, IoU=1, MCC=1."""
        gt = np.zeros((10, 10, 5), dtype=bool)
        gt[3:6, 3:6, 1:3] = True
        pred = gt.copy()

        m = calculate_metrics(
            pred_mask=pred,
            ground_truth=gt,
            method_name="Perfect",
            nominal_alpha=0.05,
        )

        assert m.tpr == 1.0
        assert m.fpr == 0.0
        assert m.specificity == 1.0
        assert m.precision == 1.0
        assert m.fdp == 0.0
        assert m.f1_score == 1.0
        assert m.iou == 1.0
        assert pytest.approx(m.mcc, abs=1e-4) == 1.0
        assert m.fdr_controlled is True
        assert m.centroid_error_voxels == 0.0

    def test_empty_prediction(self):
        """An empty prediction (no discoveries) must handle division-by-zero safely."""
        gt = np.zeros((10, 10, 5), dtype=bool)
        gt[3:6, 3:6, 1:3] = True
        pred = np.zeros_like(gt, dtype=bool)

        m = calculate_metrics(
            pred_mask=pred,
            ground_truth=gt,
            method_name="Empty",
            nominal_alpha=0.05,
        )

        assert m.tpr == 0.0
        assert m.precision == 0.0
        assert m.fdp == 0.0
        assert m.f1_score == 0.0
        assert m.true_positives == 0

    def test_centroid_distance_computation(self):
        """Centroid distance between separated clusters should match Euclidean distance."""
        mask1 = np.zeros((20, 20, 10), dtype=bool)
        mask2 = np.zeros((20, 20, 10), dtype=bool)

        mask1[5, 5, 5] = True
        mask2[8, 9, 5] = True  # dx=3, dy=4, dz=0 => dist = 5.0

        dist = compute_centroid_distance(mask1, mask2)
        assert pytest.approx(dist, abs=1e-3) == 5.0


class TestBaselinesExecution:
    """Test standard and proposed multiple-testing correction methods."""

    @pytest.fixture
    def test_volume(self):
        return generate_synthetic_benchmark_volume(
            shape=(14, 14, 6),
            snr=3.5,
            seed=42,
        )

    def test_benjamini_hochberg(self, test_volume):
        bh = BenjaminiHochbergCorrection(alpha=0.05)
        out = bh.run(
            z_scores=test_volume.z_scores,
            p_values=test_volume.p_values,
            spatial_mask=test_volume.spatial_mask,
        )
        assert out.method_name == "Benjamini-Hochberg (BH-FDR)"
        assert out.significant_mask.shape == test_volume.shape
        assert out.runtime_seconds > 0.0
        assert out.peak_memory_mb >= 0.0
        assert "p_cutoff" in out.metadata

    def test_local_fdr(self, test_volume):
        lfdr = LocalFDRCorrection(alpha=0.05)
        out = lfdr.run(
            z_scores=test_volume.z_scores,
            spatial_mask=test_volume.spatial_mask,
        )
        assert out.method_name == "Local FDR (Efron lfdr)"
        assert out.significant_mask.shape == test_volume.shape
        assert "p0_estimated" in out.metadata
        assert "lfdr_cutoff" in out.metadata

    def test_original_fchmrf(self, test_volume):
        orig = OriginalFCHMRFLISCorrection(alpha=0.05, max_iterations=6)
        out = orig.run(
            z_scores=test_volume.z_scores,
            spatial_mask=test_volume.spatial_mask,
        )
        assert "Original fcHMRF-LIS" in out.method_name
        assert out.significant_mask.shape == test_volume.shape
        assert "lis_threshold" in out.metadata

    def test_roi_assisted_fchmrf(self, test_volume):
        roi_m = ROIAssistedFCHMRFLISCorrection(alpha=0.05, lambda_dl=0.5, max_iterations=6)
        out = roi_m.run(
            z_scores=test_volume.z_scores,
            dl_prior=test_volume.dl_prior,
            spatial_mask=test_volume.spatial_mask,
        )
        assert "ROI-Assisted fcHMRF-LIS" in out.method_name
        assert out.significant_mask.shape == test_volume.shape


class TestComparativeBenchmarkRunner:
    """Test full multi-method benchmark execution, aggregation, and artifact exports."""

    def test_benchmark_runner_and_artifacts(self, tmp_path):
        # Fast 2-trial dataset
        dataset = generate_monte_carlo_dataset(
            num_trials=2,
            shape=(12, 12, 6),
            snr=3.2,
            base_seed=42,
        )

        methods = [
            BenjaminiHochbergCorrection(alpha=0.05),
            LocalFDRCorrection(alpha=0.05),
            OriginalFCHMRFLISCorrection(alpha=0.05, max_iterations=5),
            ROIAssistedFCHMRFLISCorrection(alpha=0.05, lambda_dl=0.5, max_iterations=5),
            AdaptiveROIAssistedFCHMRFLISCorrection(
                alpha=0.05,
                optimized_parameters={"beta": 1.0, "sigma_spatial": 2.0, "sigma_intensity": 1.5, "lambda_dl": 0.6},
                max_iterations=5,
            ),
        ]

        runner = ComparativeExperimentRunner(nominal_alpha=0.05, methods=methods)
        report = runner.run_benchmark(dataset=dataset)

        assert isinstance(report, BenchmarkComparisonReport)
        assert report.num_trials == 2
        assert len(report.aggregated_metrics) == 5

        # Verify export to CSV
        csv_file = tmp_path / "metrics.csv"
        report.export_csv(csv_file)
        assert csv_file.exists()
        assert csv_file.stat().st_size > 500

        # Verify Markdown table
        md_table = report.to_markdown_table()
        assert "Benjamini-Hochberg" in md_table
        assert "Adaptive ROI-Assisted" in md_table

        # Verify publication summary plot
        fig_file = tmp_path / "benchmark_fig.png"
        out_fig = plot_comparative_benchmark_summary(report, output_path=fig_file, dpi=80)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 10000
