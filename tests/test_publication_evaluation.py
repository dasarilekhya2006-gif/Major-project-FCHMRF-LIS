"""Unit and integration tests for Phase 14: Publication-Quality Evaluation & Ablation."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.core.seed import set_seed
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.evaluation.statistical_testing import (
    PairedTestResult,
    apply_holm_bonferroni,
    calculate_cliffs_delta,
    calculate_cohens_d,
    paired_statistical_comparison,
)
from src.evaluation.ablation import (
    AblationArm,
    AblationStudyResult,
    AblationStudyRunner,
    ComponentContribution,
)
from src.evaluation.publication_framework import (
    PublicationEvaluationSuite,
    ScientificProtocol,
)


class TestStatisticalTesting:
    """Tests for statistical comparison methods and effect size calculation."""

    def test_cohens_d_and_cliffs_delta(self) -> None:
        set_seed(42)
        x = np.array([0.5, 0.6, 0.55, 0.58, 0.52])
        y = np.array([0.7, 0.75, 0.72, 0.78, 0.71])

        d = calculate_cohens_d(x, y, paired=True)
        assert d > 0.8  # Large positive effect size

        delta = calculate_cliffs_delta(x, y)
        assert delta > 0.5  # Large non-parametric separation

    def test_paired_statistical_comparison(self) -> None:
        x = [0.1, 0.2, 0.15, 0.18, 0.12]
        y = [0.4, 0.45, 0.42, 0.48, 0.39]

        res = paired_statistical_comparison(
            baseline_values=x,
            candidate_values=y,
            metric_name="power",
            baseline_name="Arm A",
            candidate_name="Arm D",
        )
        assert isinstance(res, PairedTestResult)
        assert res.mean_difference > 0.0
        assert res.t_pvalue < 0.05
        assert res.is_statistically_significant is True
        assert res.ci_95_lower < res.mean_difference < res.ci_95_upper

    def test_holm_bonferroni_correction(self) -> None:
        tests = [
            paired_statistical_comparison([1, 2, 3], [10, 20, 30], "m1"),
            paired_statistical_comparison([1, 2, 3], [1.1, 2.1, 3.1], "m2"),
            paired_statistical_comparison([1, 2, 3], [1.01, 2.01, 3.01], "m3"),
        ]
        adjusted = apply_holm_bonferroni(tests, alpha=0.05)
        assert len(adjusted) == 3
        for t in adjusted:
            assert t.holm_bonferroni_pvalue is not None
            assert t.holm_bonferroni_pvalue >= t.t_pvalue


class TestComponentAttribution:
    """Tests for the additive component decomposition in ablation studies."""

    def test_attribution_decomposition_math(self) -> None:
        # Construct synthetic metric values
        # Baseline BH = 0.50
        # Arm A (fcHMRF alone) = 0.65 (Spatial gain = +0.15)
        # Arm B (ROI + fcHMRF) = 0.75 (ROI gain = +0.10)
        # Arm C (Adaptive) = 0.70 (Adaptive gain = +0.05)
        # Arm D (Full Proposed) = 0.83 (Synergy = 0.83 - (0.65 + 0.10 + 0.05) = +0.03)
        contrib = ComponentContribution(
            metric_name="tpr",
            baseline_bh=0.50,
            arm_a_fchmrf_alone=0.65,
            arm_b_roi_fchmrf=0.75,
            arm_c_adaptive_fchmrf=0.70,
            arm_d_roi_adaptive_fchmrf=0.83,
        )
        contrib.compute_decomposition()

        assert pytest.approx(contrib.delta_spatial_model, rel=1e-4) == 0.15
        assert pytest.approx(contrib.delta_roi_prior, rel=1e-4) == 0.10
        assert pytest.approx(contrib.delta_adaptive_tuning, rel=1e-4) == 0.05
        assert pytest.approx(contrib.delta_synergy_interaction, rel=1e-4) == 0.03
        assert pytest.approx(contrib.delta_total_improvement, rel=1e-4) == 0.18
        assert contrib.percent_improvement_over_original > 20.0


class TestAblationExecution:
    """Integration test verifying all 4 ablation arms and baseline execute cleanly."""

    def test_single_sample_ablation(self) -> None:
        set_seed(42)
        sample = generate_synthetic_spatial_dataset(
            grid_size=(16, 16, 8),
            snr=2.5,
            num_clusters=1,
            cluster_radius_range=(2, 3),
            seed=42,
        )

        runner = AblationStudyRunner(nominal_alpha=0.05, device="cpu")
        res_dict = runner.run_single_ablation_sample(sample=sample, seed=42)

        expected_arms = [
            "Baseline_BH",
            AblationArm.ARM_A_FCHMRF_ALONE.value,
            AblationArm.ARM_B_ROI_FCHMRF.value,
            AblationArm.ARM_C_ADAPTIVE_FCHMRF.value,
            AblationArm.ARM_D_ROI_ADAPTIVE_FCHMRF.value,
        ]

        for arm in expected_arms:
            assert arm in res_dict
            metrics = res_dict[arm]
            assert 0.0 <= metrics.tpr <= 1.0
            assert 0.0 <= metrics.fdp <= 1.0
            assert 0.0 <= metrics.f1_score <= 1.0
            assert metrics.runtime_seconds >= 0.0


class TestPublicationEvaluationSuite:
    """End-to-end integration test of the full publication framework."""

    def test_suite_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            suite = PublicationEvaluationSuite(
                nominal_alpha=0.05,
                output_dir=out_dir,
                device="cpu",
            )

            # Fast test run with 1 SNR and 2 seeds
            results = suite.run_multi_snr_ablation_evaluation(
                snr_levels=[2.5],
                seeds=[42, 101],
                grid_size=(16, 16, 8),
            )

            assert 2.5 in results
            res = results[2.5]
            assert res.n_seeds == 2

            # Check exported artifacts
            assert (out_dir / "publication_ablation_metrics.csv").exists()
            assert (out_dir / "publication_ablation_table.md").exists()
            assert (out_dir / "publication_ablation_table.tex").exists()
            assert (out_dir / "component_attribution_summary.md").exists()
            assert (out_dir / "scientific_evaluation_report.md").exists()
            assert (out_dir / "fig1_ablation_performance_bars.png").exists()
            assert (out_dir / "fig2_component_attribution_waterfall.png").exists()
            assert (out_dir / "fig3_snr_sensitivity_curves.png").exists()

            # Verify report content
            report_text = (out_dir / "scientific_evaluation_report.md").read_text(encoding="utf-8")
            assert "Rigorous Scientific Evaluation Report" in report_text
            assert "Ablation Study Breakdown" in report_text
            assert "Formal Statistical Comparisons" in report_text
