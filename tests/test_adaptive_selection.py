"""Unit and integration tests for Phase 6: Adaptive Parameter Selection."""

from pathlib import Path
import pytest
import numpy as np

from src.core.exceptions import DataValidationError
from src.adaptive_selection.search_space import (
    HyperparameterRange,
    ParameterType,
    SearchSpace,
    create_fchmrf_search_space,
)
from src.adaptive_selection.objectives import (
    ObjectiveResult,
    ObjectiveType,
    ValidationFDRBenchmarkObjective,
    VariationalEvidenceObjective,
)
from src.adaptive_selection.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
)
from src.adaptive_selection.random_search import RandomSearchOptimizer
from src.adaptive_selection.baseline_comparator import (
    BaselineComparator,
    ComparisonReport,
)
from src.adaptive_selection.visualization import plot_adaptive_optimization_summary


class TestSearchSpace:
    """Test search space mappings, bounds, and vectorization."""

    def test_search_space_roundtrip(self):
        """Converting parameters to normalized vector and back must preserve values."""
        space = create_fchmrf_search_space(include_lambda_dl=True)
        default_params = space.get_default_parameters()

        # Vectorize
        vec = space.to_normalized_vector(default_params)
        assert len(vec) == space.dimension
        assert np.all((vec >= 0.0) & (vec <= 1.0))

        # Reconstruct
        reconstructed = space.from_normalized_vector(vec)
        for k, v in default_params.items():
            assert pytest.approx(reconstructed[k], abs=1e-4) == v

    def test_clipping_bounds(self):
        """Values outside bounds should be safely clamped."""
        hp = HyperparameterRange(
            name="beta",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.2, 2.0),
            default_val=0.8,
        )
        assert hp.clip_value(5.0) == 2.0
        assert hp.clip_value(-1.0) == 0.2
        assert hp.clip_value(1.2) == 1.2


class TestObjectivesAndDataLeakage:
    """Test objective evaluation and strict validation data isolation."""

    def test_validation_objective_evaluates_metrics(self):
        """Verify supervised objective computes power, fdp, and score."""
        np.random.seed(42)
        shape = (10, 10, 5)
        gt = np.zeros(shape, dtype=bool)
        gt[3:7, 3:7, 2:4] = True

        z = np.random.randn(*shape)
        z[gt] += 3.5

        obj = ValidationFDRBenchmarkObjective(
            z_scores_val=z,
            ground_truth_val=gt,
            objective_type=ObjectiveType.F1_SCORE,
            max_iterations=10,
        )

        res = obj.evaluate({"beta": 0.8, "sigma_spatial": 2.5, "sigma_intensity": 1.5})
        assert isinstance(res, ObjectiveResult)
        assert 0.0 <= res.score <= 1.0
        assert "power" in res.metrics
        assert "fdp" in res.metrics
        assert res.elapsed_seconds > 0.0

    def test_empty_ground_truth_raises_error(self):
        """Objective must fail early if ground truth has zero active voxels."""
        shape = (10, 10, 5)
        gt = np.zeros(shape, dtype=bool)
        z = np.random.randn(*shape)

        with pytest.raises(DataValidationError):
            ValidationFDRBenchmarkObjective(
                z_scores_val=z,
                ground_truth_val=gt,
            )


class TestBayesianOptimization:
    """Test GP surrogate, acquisition functions, and optimization execution."""

    def test_bayesian_optimization_run(self, tmp_path):
        """Run a fast controlled Bayesian Optimization loop and verify trial logging."""
        np.random.seed(42)
        shape = (10, 10, 4)
        gt = np.zeros(shape, dtype=bool)
        gt[4:7, 4:7, 1:3] = True
        z = np.random.randn(*shape)
        z[gt] += 3.0

        space = create_fchmrf_search_space(include_lambda_dl=False, include_spatial_bandwidths=False)
        obj = ValidationFDRBenchmarkObjective(
            z_scores_val=z,
            ground_truth_val=gt,
            objective_type=ObjectiveType.PENALIZED_POWER,
            max_iterations=8,
        )

        opt = BayesianOptimizer(
            search_space=space,
            objective=obj,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            n_candidates=100,
            seed=42,
        )

        res = opt.optimize(n_iterations=3, n_initial=2)

        assert res.optimizer_name == "BayesianOptimization"
        # 1 baseline + 2 initial + 3 BO = 6 trials
        assert len(res.trials) == 6
        assert res.best_score >= res.baseline_score or res.score_improvement <= 0.0
        assert "beta" in res.best_parameters

        # Test result persistence
        saved = res.save_results(output_dir=tmp_path / "bo_out", prefix="test_bo")
        assert saved["selected_yaml"].exists()
        assert saved["history_json"].exists()


class TestBaselineComparatorAndVisualization:
    """Test head-to-head comparison on independent test partition and summary plotting."""

    def test_comparator_and_visualization(self, tmp_path):
        """Verify baseline comparator and diagnostic figure generation."""
        np.random.seed(42)
        shape = (12, 12, 6)
        gt = np.zeros(shape, dtype=bool)
        gt[4:8, 4:8, 2:4] = True
        z = np.random.randn(*shape)
        z[gt] += 3.2

        space = create_fchmrf_search_space(include_lambda_dl=False)
        obj = ValidationFDRBenchmarkObjective(
            z_scores_val=z,
            ground_truth_val=gt,
            max_iterations=8,
        )

        opt = RandomSearchOptimizer(search_space=space, objective=obj, seed=42)
        res = opt.optimize(n_iterations=2, n_initial=1)

        # Independent test volume with different random noise
        np.random.seed(99)
        z_test = np.random.randn(*shape)
        z_test[gt] += 3.0

        comparator = BaselineComparator(target_alpha=0.05)
        comp_report = comparator.compare(
            baseline_params=res.baseline_parameters,
            selected_params=res.best_parameters,
            z_scores_test=z_test,
            ground_truth_test=gt,
            optimization_cost_seconds=res.total_runtime_seconds,
        )

        assert isinstance(comp_report, ComparisonReport)
        assert comp_report.baseline_metrics.power is not None
        assert comp_report.selected_metrics.power is not None
        assert isinstance(comp_report.is_benefit_measurable, bool)

        # Visualization
        fig_path = tmp_path / "adaptive_summary.png"
        out_fig = plot_adaptive_optimization_summary(
            opt_result=res,
            comparison_report=comp_report,
            output_path=fig_path,
            dpi=100,
        )
        assert out_fig.exists()
        assert out_fig.stat().st_size > 5000
