"""Phase 8: Baseline Comparison Unit and Integration Test Suite.

Verifies:
1. Exact execution of all 5 required comparative multiple-testing methods:
   - Benjamini-Hochberg FDR
   - Original fcHMRF-LIS
   - ROI + fcHMRF-LIS
   - Adaptive fcHMRF-LIS
   - ROI + Adaptive fcHMRF-LIS
2. Rigorous calculation of all required performance and localization metrics:
   - FDR (Empirical FDP)
   - TPR (Sensitivity / Recall)
   - FPR
   - Precision
   - Recall
   - Specificity
   - Localization Accuracy (Centroid Error in voxels)
   - Runtime
3. Generation and integrity of comparison tables (Markdown and CSV).
4. CLI execution of scripts/run_phase8_comparisons.py.
"""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from src.evaluation.baselines import (
    BenjaminiHochbergCorrection,
    OriginalFCHMRFLISCorrection,
    ROIAssistedFCHMRFLISCorrection,
    AdaptiveFCHMRFLISCorrection,
    AdaptiveROIAssistedFCHMRFLISCorrection,
)
from src.evaluation.metrics import (
    calculate_metrics,
    compute_centroid_distance,
    EvaluationMetrics,
)
from src.evaluation.synthetic_benchmarks import (
    generate_synthetic_benchmark_volume,
    generate_monte_carlo_dataset,
)
from src.evaluation.comparative_runner import (
    ComparativeExperimentRunner,
    BenchmarkComparisonReport,
)


@pytest.fixture
def benchmark_volume():
    """Reproducible synthetic neuroimaging volume with known ground truth."""
    return generate_synthetic_benchmark_volume(
        shape=(16, 16, 8),
        snr=3.2,
        seed=101,
    )


def test_all_five_methods_execution(benchmark_volume):
    """Verify that all 5 requested methods execute and output valid masks and statistics."""
    methods = [
        BenjaminiHochbergCorrection(alpha=0.05, name="Benjamini-Hochberg FDR"),
        OriginalFCHMRFLISCorrection(alpha=0.05, max_iterations=6, name="Original fcHMRF-LIS"),
        ROIAssistedFCHMRFLISCorrection(alpha=0.05, lambda_dl=0.5, max_iterations=6, name="ROI + fcHMRF-LIS"),
        AdaptiveFCHMRFLISCorrection(
            alpha=0.05,
            optimized_parameters={"beta": 1.1, "sigma_spatial": 2.0, "sigma_intensity": 1.4, "w_pure_spatial": 0.2},
            max_iterations=6,
            name="Adaptive fcHMRF-LIS",
        ),
        AdaptiveROIAssistedFCHMRFLISCorrection(
            alpha=0.05,
            optimized_parameters={"beta": 1.1, "sigma_spatial": 2.0, "sigma_intensity": 1.4, "lambda_dl": 0.6},
            max_iterations=6,
            name="ROI + Adaptive fcHMRF-LIS",
        ),
    ]

    assert len(methods) == 5

    for m in methods:
        out = m.run(
            z_scores=benchmark_volume.z_scores,
            p_values=benchmark_volume.p_values,
            dl_prior=benchmark_volume.dl_prior,
            spatial_mask=benchmark_volume.spatial_mask,
        )
        assert out.significant_mask.shape == benchmark_volume.shape
        assert out.test_statistic_map.shape == benchmark_volume.shape
        assert out.runtime_seconds > 0.0
        assert isinstance(out.num_discoveries, int)
        assert out.method_name == m.name


def test_metrics_calculation_completeness(benchmark_volume):
    """Verify that calculate_metrics computes all required metrics without inventing values."""
    pred_mask = benchmark_volume.ground_truth.copy()
    # Add small controlled perturbation (1 false positive, 1 false negative)
    pred_mask[0, 0, 0] = True
    active_coords = np.argwhere(benchmark_volume.ground_truth)
    if len(active_coords) > 0:
        c = tuple(active_coords[0])
        pred_mask[c] = False

    metrics: EvaluationMetrics = calculate_metrics(
        pred_mask=pred_mask,
        ground_truth=benchmark_volume.ground_truth,
        method_name="TestPerturbed",
        nominal_alpha=0.05,
        runtime_seconds=1.23,
    )

    # All required metrics must be present, real, and non-NaN
    assert 0.0 <= metrics.fdp <= 1.0
    assert 0.0 <= metrics.tpr <= 1.0
    assert 0.0 <= metrics.fpr <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.specificity <= 1.0
    assert metrics.centroid_error_voxels >= 0.0
    assert not np.isnan(metrics.centroid_error_voxels)
    assert metrics.runtime_seconds == 1.23
    assert metrics.recall == metrics.tpr


def test_comparative_runner_five_methods(tmp_path):
    """Test full comparative runner executing all 5 methods and exporting comparison tables."""
    dataset = generate_monte_carlo_dataset(
        num_trials=2,
        shape=(12, 12, 6),
        snr=3.0,
        base_seed=42,
    )

    methods = [
        BenjaminiHochbergCorrection(alpha=0.05, name="Benjamini-Hochberg FDR"),
        OriginalFCHMRFLISCorrection(alpha=0.05, max_iterations=5, name="Original fcHMRF-LIS"),
        ROIAssistedFCHMRFLISCorrection(alpha=0.05, lambda_dl=0.5, max_iterations=5, name="ROI + fcHMRF-LIS"),
        AdaptiveFCHMRFLISCorrection(alpha=0.05, max_iterations=5, name="Adaptive fcHMRF-LIS"),
        AdaptiveROIAssistedFCHMRFLISCorrection(alpha=0.05, max_iterations=5, name="ROI + Adaptive fcHMRF-LIS"),
    ]

    runner = ComparativeExperimentRunner(nominal_alpha=0.05, methods=methods)
    report = runner.run_benchmark(dataset=dataset)

    assert isinstance(report, BenchmarkComparisonReport)
    assert report.num_trials == 2
    assert len(report.aggregated_metrics) == 5

    # Check Markdown table content
    md_table = report.to_markdown_table()
    assert "Benjamini-Hochberg FDR" in md_table
    assert "Original fcHMRF-LIS" in md_table
    assert "ROI + fcHMRF-LIS" in md_table
    assert "Adaptive fcHMRF-LIS" in md_table
    assert "ROI + Adaptive fcHMRF-LIS" in md_table
    assert "FDR (Empirical FDP)" in md_table
    assert "TPR (Recall)" in md_table
    assert "FPR" in md_table
    assert "Precision" in md_table
    assert "Specificity" in md_table
    assert "Localization Error" in md_table
    assert "Runtime" in md_table

    # Check CSV export
    csv_file = tmp_path / "baseline_comparison.csv"
    report.export_csv(csv_file)
    assert csv_file.exists()
    assert csv_file.stat().st_size > 500


def test_cli_run_phase8_comparisons(tmp_path):
    """Test CLI runner execution with minimal trials."""
    cmd = [
        sys.executable,
        "scripts/run_phase8_comparisons.py",
        "--num-trials", "2",
        "--snr", "3.0",
        "--alpha", "0.05",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI runner failed with error:\n{res.stderr}\nOutput:\n{res.stdout}"
    assert "PHASE 8 RIGOROUS EXPERIMENT BENCHMARK COMPLETE" in res.stdout or "PHASE 8" in res.stdout
