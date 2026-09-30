"""Automated research experiment runner and reproducibility package."""

from src.experiments.schemas import (
    AdaptiveConfig,
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    FCHMRFConfig,
    PreprocessingConfig,
    StatisticalTestConfig,
    UNetConfig,
)
from src.experiments.runner import ExperimentRunner
from src.experiments.reproducibility import reproduce_experiment
from src.experiments.matrix import build_experiment_matrix, load_suite_from_yaml
from src.experiments.tables import (
    export_suite_summary_csv,
    generate_comparison_latex_table,
    generate_comparison_markdown_table,
)
from src.experiments.plots import (
    plot_single_experiment_diagnostic,
    plot_suite_summary_figures,
)

__all__ = [
    "AdaptiveConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ExperimentResult",
    "FCHMRFConfig",
    "PreprocessingConfig",
    "StatisticalTestConfig",
    "UNetConfig",
    "ExperimentRunner",
    "reproduce_experiment",
    "build_experiment_matrix",
    "load_suite_from_yaml",
    "export_suite_summary_csv",
    "generate_comparison_latex_table",
    "generate_comparison_markdown_table",
    "plot_single_experiment_diagnostic",
    "plot_suite_summary_figures",
]
