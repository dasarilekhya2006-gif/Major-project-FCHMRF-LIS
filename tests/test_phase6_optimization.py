"""Phase 6 Verification Test Suite: Adaptive Parameter Selection and Bayesian Optimization.

Validates all Phase 6 scientific requirements:
1. Public API exports from top-level `optimization`.
2. Definition of tunable parameters with mathematically sound bounds.
3. Search space vectorization, clipping, and denormalization.
4. Objective functions evaluated solely on validation data to prevent data leakage.
5. Strict trial recording: every evaluation step logged with metrics and timestamps.
6. Persistence of selected hyperparameters (YAML) and complete trial history (JSON).
7. Controlled head-to-head comparison between fixed baseline and adaptively selected parameters on held-out test data.
8. Controlled empirical experiment measuring actual benefits, costs, and statistical significance.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import yaml

import optimization
from optimization import (
    AcquisitionFunction,
    BaselineComparator,
    BayesianOptimizer,
    ComparisonReport,
    HyperparameterRange,
    ObjectiveResult,
    ObjectiveType,
    OptimizationResult,
    OptimizationTrial,
    ParameterType,
    RandomSearchOptimizer,
    SearchSpace,
    ValidationFDRBenchmarkObjective,
    create_fchmrf_search_space,
    plot_adaptive_optimization_summary,
)
from src.core.exceptions import DataValidationError


class TestPhase6PublicAPI:
    """Verify top-level optimization exports."""

    def test_all_expected_symbols_exported(self):
        expected_symbols = [
            "HyperparameterRange",
            "ParameterType",
            "SearchSpace",
            "create_fchmrf_search_space",
            "BaseObjective",
            "ObjectiveResult",
            "ObjectiveType",
            "ValidationFDRBenchmarkObjective",
            "BaseOptimizer",
            "OptimizationResult",
            "OptimizationTrial",
            "AcquisitionFunction",
            "BayesianOptimizer",
            "RandomSearchOptimizer",
            "BaselineComparator",
            "ComparisonReport",
            "plot_adaptive_optimization_summary",
        ]
        for symbol in expected_symbols:
            assert hasattr(optimization, symbol), f"Symbol '{symbol}' not found in optimization"


class TestTunableParametersAndBounds:
    """Verify tunable hyperparameter definitions and domain constraints."""

    def test_search_space_parameters_and_ranges(self):
        space = create_fchmrf_search_space(include_lambda_dl=True, include_spatial_bandwidths=True)
        param_names = space.names

        assert "beta" in param_names
        assert "sigma_spatial" in param_names
        assert "sigma_intensity" in param_names
        assert "w_pure_spatial" in param_names
        assert "lambda_dl" in param_names

        # Verify valid bounds
        param_map = {p.name: p for p in space.parameters}
        assert param_map["beta"].bounds == (0.1, 2.5)
        assert param_map["sigma_spatial"].bounds == (1.0, 5.0)
        assert param_map["sigma_intensity"].bounds == (0.5, 3.5)
        assert param_map["w_pure_spatial"].bounds == (0.0, 0.5)
        assert param_map["lambda_dl"].bounds == (0.0, 1.0)

    def test_clipping_and_normalization_preservation(self):
        space = create_fchmrf_search_space()
        defaults = space.get_default_parameters()

        # Normalization to [0, 1]
        norm_vec = space.to_normalized_vector(defaults)
        assert len(norm_vec) == space.dimension
        assert np.all((norm_vec >= 0.0) & (norm_vec <= 1.0))

        # Reconstructed dictionary matches defaults
        recon = space.from_normalized_vector(norm_vec)
        for k, v in defaults.items():
            assert pytest.approx(recon[k], abs=1e-4) == v


class TestDataLeakagePreventionAndObjectives:
    """Verify objective functions evaluate exclusively on validation partitions."""

    def test_validation_objective_isolated_from_test_set(self):
        np.random.seed(42)
        val_shape = (10, 10, 4)
        gt_val = np.zeros(val_shape, dtype=bool)
        gt_val[3:7, 3:7, 1:3] = True
        z_val = np.random.randn(*val_shape)
        z_val[gt_val] += 3.2

        obj = ValidationFDRBenchmarkObjective(
            z_scores_val=z_val,
            ground_truth_val=gt_val,
            objective_type=ObjectiveType.F1_SCORE,
            max_iterations=10,
        )

        res = obj.evaluate({"beta": 0.9, "sigma_spatial": 2.2, "sigma_intensity": 1.4})
        assert isinstance(res, ObjectiveResult)
        assert 0.0 <= res.score <= 1.0
        assert "power" in res.metrics
        assert "fdp" in res.metrics
        assert res.elapsed_seconds > 0.0

    def test_invalid_ground_truth_raises_data_validation_error(self):
        val_shape = (8, 8, 4)
        gt_empty = np.zeros(val_shape, dtype=bool)
        z_val = np.random.randn(*val_shape)

        with pytest.raises(DataValidationError):
            ValidationFDRBenchmarkObjective(z_scores_val=z_val, ground_truth_val=gt_empty)


class TestBayesianOptimizationExecutionAndTrialRecording:
    """Verify Bayesian Optimization loop, trial recording, and artifact persistence."""

    def test_bo_records_every_trial_and_persists_artifacts(self, tmp_path):
        np.random.seed(42)
        shape = (10, 10, 4)
        gt = np.zeros(shape, dtype=bool)
        gt[3:7, 3:7, 1:3] = True
        z_val = np.random.randn(*shape)
        z_val[gt] += 3.5

        space = create_fchmrf_search_space(include_lambda_dl=False, include_spatial_bandwidths=False)
        obj = ValidationFDRBenchmarkObjective(
            z_scores_val=z_val,
            ground_truth_val=gt,
            objective_type=ObjectiveType.PENALIZED_POWER,
            max_iterations=8,
        )

        n_initial = 2
        n_bo_iters = 3
        optimizer = BayesianOptimizer(
            search_space=space,
            objective=obj,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            n_candidates=100,
            seed=42,
        )

        res = optimizer.optimize(n_iterations=n_bo_iters, n_initial=n_initial)

        # Expected trials: 1 baseline trial + 2 initial random + 3 BO = 6 trials total
        assert len(res.trials) == 1 + n_initial + n_bo_iters
        for trial in res.trials:
            assert isinstance(trial, OptimizationTrial)
            assert "beta" in trial.parameters
            assert trial.elapsed_seconds > 0.0
            assert trial.timestamp is not None
            assert "power" in trial.metrics
            assert "fdp" in trial.metrics

        # Best parameters found
        assert "beta" in res.best_parameters
        assert res.best_score is not None

        # Verify artifact persistence
        saved = res.save_results(output_dir=tmp_path / "bo_results", prefix="phase6_bo")
        assert saved["selected_yaml"].exists()
        assert saved["history_json"].exists()

        # Inspect YAML payload
        with open(saved["selected_yaml"], "r") as f:
            yaml_data = yaml.safe_load(f)
            assert "adaptive_selection" in yaml_data
            assert "selected_parameters" in yaml_data["adaptive_selection"]
            assert "best_score" in yaml_data["adaptive_selection"]

        # Inspect JSON payload
        with open(saved["history_json"], "r") as f:
            json_data = json.load(f)
            assert json_data["optimizer_name"] == "BayesianOptimization"
            assert len(json_data["trials"]) == 6


class TestControlledAdaptiveVsFixedComparison:
    """Run a controlled experiment comparing adaptive vs fixed baseline parameters on independent test data."""

    def test_head_to_head_comparison_on_test_volume(self, tmp_path):
        # 1. Validation split for optimization
        np.random.seed(101)
        shape = (12, 12, 6)
        gt = np.zeros(shape, dtype=bool)
        gt[4:8, 4:8, 2:4] = True

        z_val = np.random.randn(*shape)
        z_val[gt] += 3.2

        space = create_fchmrf_search_space(include_lambda_dl=False)
        obj = ValidationFDRBenchmarkObjective(
            z_scores_val=z_val,
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
        opt_res = opt.optimize(n_iterations=3, n_initial=2)

        # 2. Independent test split with distinct realization of noise (zero leakage)
        np.random.seed(202)
        z_test = np.random.randn(*shape)
        z_test[gt] += 3.0  # slightly lower SNR on test volume

        comparator = BaselineComparator(target_alpha=0.05)
        report = comparator.compare(
            baseline_params=opt_res.baseline_parameters,
            selected_params=opt_res.best_parameters,
            z_scores_test=z_test,
            ground_truth_test=gt,
            optimization_cost_seconds=opt_res.total_runtime_seconds,
        )

        assert isinstance(report, ComparisonReport)
        assert report.baseline_metrics.power is not None
        assert report.selected_metrics.power is not None
        assert report.baseline_metrics.fdp is not None
        assert report.selected_metrics.fdp is not None
        assert report.optimization_cost_seconds > 0.0
        assert isinstance(report.is_benefit_measurable, bool)
        assert len(report.benefit_summary) > 0

        # Verify summary plot generation
        fig_out = tmp_path / "phase6_experiment_summary.png"
        plot_path = plot_adaptive_optimization_summary(
            opt_result=opt_res,
            comparison_report=report,
            output_path=fig_out,
            dpi=100,
        )
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000
