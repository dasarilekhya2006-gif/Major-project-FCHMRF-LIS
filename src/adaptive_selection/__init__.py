"""Adaptive parameter selection module for fcHMRF-LIS spatial inference.

Provides:
- Search space abstractions with mathematical bound checking.
- Objective functions with data leakage prevention (supervised benchmarks and unsupervised ELBO).
- Bayesian Optimization with Gaussian Process surrogates and acquisition functions (EI, UCB, PI).
- Modular Random Search baseline optimizer.
- Independent baseline comparator on held-out test data.
- Diagnostic and publication-grade visualization.
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
    "ParameterType",
    "HyperparameterRange",
    "SearchSpace",
    "create_fchmrf_search_space",
    "ObjectiveType",
    "ObjectiveResult",
    "BaseObjective",
    "ValidationFDRBenchmarkObjective",
    "VariationalEvidenceObjective",
    "OptimizationTrial",
    "OptimizationResult",
    "BaseOptimizer",
    "AcquisitionFunction",
    "BayesianOptimizer",
    "RandomSearchOptimizer",
    "PerformanceMetrics",
    "ComparisonReport",
    "BaselineComparator",
    "plot_adaptive_optimization_summary",
]
