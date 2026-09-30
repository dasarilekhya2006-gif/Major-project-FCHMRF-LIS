"""Evaluation and comparative benchmarking module for Phase 8.

Provides:
- Rigorous performance metrics (TPR, FPR, Specificity, Precision, FDP, Dice, IoU, MCC, Centroid Error).
- Wall-clock runtime and peak RAM profiling via tracemalloc.
- Baseline multiple-testing corrections:
  1. Benjamini-Hochberg FDR (BH-FDR)
  2. Efron's Local False Discovery Rate (lfdr)
  3. Original Published fcHMRF-LIS (lambda_dl = 0.0)
  4. ROI-Assisted fcHMRF-LIS (fixed lambda_dl = 0.5)
  5. Adaptive ROI-Assisted fcHMRF-LIS (optimized hyperparameters)
- Reproducible synthetic neuroimaging benchmark generator with ground truth.
- Multi-method comparative experiment runner with CSV, ASCII, and Markdown report export.
- Publication-grade 4-panel diagnostic figure.
"""

from src.evaluation.metrics import (
    AggregatedMetrics,
    EvaluationMetrics,
    aggregate_metrics_across_trials,
    calculate_metrics,
    compute_centroid_distance,
)
from src.evaluation.baselines import (
    AdaptiveFCHMRFLISCorrection,
    AdaptiveROIAssistedFCHMRFLISCorrection,
    BaseCorrectionMethod,
    BenjaminiHochbergCorrection,
    LocalFDRCorrection,
    MethodOutput,
    OriginalFCHMRFLISCorrection,
    ROIAssistedFCHMRFLISCorrection,
)
from src.evaluation.synthetic_benchmarks import (
    SyntheticBenchmarkVolume,
    generate_monte_carlo_dataset,
    generate_synthetic_benchmark_volume,
)
from src.evaluation.comparative_runner import (
    BenchmarkComparisonReport,
    ComparativeExperimentRunner,
    TrialResult,
)
from src.evaluation.visualization import plot_comparative_benchmark_summary

__all__ = [
    "EvaluationMetrics",
    "AggregatedMetrics",
    "calculate_metrics",
    "compute_centroid_distance",
    "aggregate_metrics_across_trials",
    "BaseCorrectionMethod",
    "MethodOutput",
    "BenjaminiHochbergCorrection",
    "LocalFDRCorrection",
    "OriginalFCHMRFLISCorrection",
    "ROIAssistedFCHMRFLISCorrection",
    "AdaptiveFCHMRFLISCorrection",
    "AdaptiveROIAssistedFCHMRFLISCorrection",
    "SyntheticBenchmarkVolume",
    "generate_synthetic_benchmark_volume",
    "generate_monte_carlo_dataset",
    "TrialResult",
    "BenchmarkComparisonReport",
    "ComparativeExperimentRunner",
    "plot_comparative_benchmark_summary",
    # Phase 14: Publication Evaluation Framework & Ablation
    "ScientificProtocol",
    "PublicationEvaluationSuite",
    "AblationArm",
    "AblationStudyResult",
    "AblationStudyRunner",
    "ComponentContribution",
    "PairedTestResult",
    "paired_statistical_comparison",
    "calculate_cohens_d",
    "calculate_cliffs_delta",
    "apply_holm_bonferroni",
]

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
