"""Robustness and uncertainty evaluation module for Phase 9.

Provides:
- Stability metrics (pairwise Jaccard/Dice, discovery count CV, boundary Shannon entropy).
- Multi-dimensional stress-testing generators (SNR sweep, additive voxel jitter, spatial correlation, ROI scaling).
- RobustnessEvaluator orchestrator for controlled Monte Carlo sensitivity analysis.
- Publication-grade diagnostic sensitivity curves and spatial uncertainty entropy visualization.
"""

from src.robustness.metrics import (
    StabilityMetrics,
    compute_detection_probability_map,
    compute_pairwise_jaccard_and_dice,
    compute_voxel_uncertainty_entropy,
    evaluate_repetition_stability,
)
from src.robustness.perturbations import (
    apply_additive_voxel_perturbation,
    apply_prior_misalignment_perturbation,
    generate_snr_sweep_dataset,
    generate_spatial_correlation_sweep_dataset,
    generate_roi_size_sweep_dataset,
)
from src.robustness.evaluator import (
    RobustnessEvaluationReport,
    RobustnessEvaluator,
)
from src.robustness.visualization import plot_robustness_analysis_summary

__all__ = [
    "StabilityMetrics",
    "compute_pairwise_jaccard_and_dice",
    "compute_detection_probability_map",
    "compute_voxel_uncertainty_entropy",
    "evaluate_repetition_stability",
    "apply_additive_voxel_perturbation",
    "apply_prior_misalignment_perturbation",
    "generate_snr_sweep_dataset",
    "generate_spatial_correlation_sweep_dataset",
    "generate_roi_size_sweep_dataset",
    "RobustnessEvaluationReport",
    "RobustnessEvaluator",
    "plot_robustness_analysis_summary",
]
