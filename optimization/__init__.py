"""Phase 6: Adaptive Parameter Selection and Bayesian Optimization for fcHMRF-LIS.

This module provides the optimization framework for hyperparameter selection:
- SearchSpace and HyperparameterRange definitions with valid domain bounds.
- Objective functions (F1 score, MCC, penalized power) evaluated strictly on validation data to prevent test leakage.
- Bayesian Optimization with Gaussian Process surrogate and acquisition functions (EI, UCB, PI).
- Random Search optimization for benchmarking.
- Trial logging recording every evaluation step with elapsed time and metrics.
- Parameter persistence to YAML and trial history to JSON.
- BaselineComparator for rigorous head-to-head evaluation of adaptive vs. fixed parameters on held-out test data.
"""

from src.adaptive_selection.search_space import (
    HyperparameterRange,
    ParameterType,
    SearchSpace,
    create_fchmrf_search_space,
)
from src.adaptive_selection.objectives import (
    BaseObjective,
    ObjectiveResult,
    ObjectiveType,
    ValidationFDRBenchmarkObjective,
    VariationalEvidenceObjective,
)
from src.adaptive_selection.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
)
from src.adaptive_selection.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
)
from src.adaptive_selection.random_search import RandomSearchOptimizer
from src.adaptive_selection.baseline_comparator import (
    BaselineComparator,
    ComparisonReport,
    PerformanceMetrics,
)
from src.adaptive_selection.visualization import plot_adaptive_optimization_summary

__all__ = [
    "HyperparameterRange",
    "ParameterType",
    "SearchSpace",
    "create_fchmrf_search_space",
    "BaseObjective",
    "ObjectiveResult",
    "ObjectiveType",
    "ValidationFDRBenchmarkObjective",
    "VariationalEvidenceObjective",
    "BaseOptimizer",
    "OptimizationResult",
    "OptimizationTrial",
    "AcquisitionFunction",
    "BayesianOptimizer",
    "RandomSearchOptimizer",
    "BaselineComparator",
    "ComparisonReport",
    "PerformanceMetrics",
    "plot_adaptive_optimization_summary",
]
