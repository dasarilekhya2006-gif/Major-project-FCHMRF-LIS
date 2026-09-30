"""Diagnostic and publication visualization utilities for adaptive parameter selection."""

from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.logging import get_logger
from src.adaptive_selection.base_optimizer import OptimizationResult
from src.adaptive_selection.baseline_comparator import ComparisonReport

logger = get_logger(__name__)


def plot_adaptive_optimization_summary(
    opt_result: OptimizationResult,
    comparison_report: Optional[ComparisonReport] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Generate a multi-panel publication-grade summary of hyperparameter optimization.

    Panels:
      1. Objective Score Trajectory over Trials with Running Best
      2. Computational Cost (Per-Trial Latency & Cumulative Search Time)
      3. Selected vs Baseline Hyperparameter Comparison
      4. Held-Out Test Evaluation: Baseline vs Adaptive Performance

    Args:
        opt_result: OptimizationResult from optimizer.optimize().
        comparison_report: Optional test set evaluation report.
        output_path: Target PNG file path.
        dpi: Resolution.

    Returns:
        Path to generated figure.
    """
    if output_path is None:
        output_path = Path("outputs/figures/adaptive_optimization_summary.png")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(
        f"Adaptive Hyperparameter Optimization: {opt_result.optimizer_name} "
        f"(Total Search Time: {opt_result.total_runtime_seconds:.1f}s)",
        fontsize=15,
        fontweight="bold",
    )

    trials = opt_result.trials
    trial_indices = [t.trial_idx for t in trials]
    scores = [t.score for t in trials]
    times = [t.elapsed_seconds for t in trials]

    # Running maximum
    running_best = np.maximum.accumulate(scores)

    # -------------------------------------------------------------
    # Panel 1: Objective Trajectory
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(trial_indices, scores, "o-", color="#1f77b4", alpha=0.6, label="Trial Score")
    ax1.plot(trial_indices, running_best, "r-", linewidth=2.5, label="Running Best")
    ax1.axhline(
        opt_result.baseline_score,
        color="k",
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline Fixed ({opt_result.baseline_score:.3f})",
    )
    ax1.set_xlabel("Trial Index", fontsize=11)
    ax1.set_ylabel("Validation Objective Score", fontsize=11)
    ax1.set_title("Optimization Objective Trajectory", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: Computational Cost per Trial
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.bar(trial_indices, times, color="#2ca02c", alpha=0.7, width=0.6, label="Trial Latency (s)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(trial_indices, np.cumsum(times), color="#d62728", linestyle="-", linewidth=2, label="Cumulative Time (s)")
    ax2.set_xlabel("Trial Index", fontsize=11)
    ax2.set_ylabel("Trial Runtime (s)", fontsize=11, color="#2ca02c")
    ax2_twin.set_ylabel("Cumulative Search Time (s)", fontsize=11, color="#d62728")
    ax2.set_title("Computational Cost per Trial", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 3: Hyperparameter Values (Baseline vs Selected)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    param_names = list(opt_result.baseline_parameters.keys())
    base_vals = [float(opt_result.baseline_parameters[k]) for k in param_names]
    sel_vals = [float(opt_result.best_parameters[k]) for k in param_names]

    x = np.arange(len(param_names))
    width = 0.35
    ax3.bar(x - width / 2, base_vals, width, label="Baseline (Fixed)", color="#7f7f7f", alpha=0.8)
    ax3.bar(x + width / 2, sel_vals, width, label="Adaptive (Selected)", color="#ff7f0e", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names, rotation=20, ha="right", fontsize=10)
    ax3.set_ylabel("Parameter Value", fontsize=11)
    ax3.set_title("Hyperparameter Comparison", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 4: Test Set Benchmark Performance
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    if comparison_report is not None and comparison_report.baseline_metrics.f1_score is not None:
        metrics_labels = ["Power", "1 - FDP", "F1-Score", "MCC"]
        b_m = comparison_report.baseline_metrics
        s_m = comparison_report.selected_metrics
        b_scores = [b_m.power or 0.0, 1.0 - (b_m.fdp or 0.0), b_m.f1_score or 0.0, b_m.mcc or 0.0]
        s_scores = [s_m.power or 0.0, 1.0 - (s_m.fdp or 0.0), s_m.f1_score or 0.0, s_m.mcc or 0.0]

        x_m = np.arange(len(metrics_labels))
        ax4.bar(x_m - width / 2, b_scores, width, label="Baseline", color="#7f7f7f", alpha=0.8)
        ax4.bar(x_m + width / 2, s_scores, width, label="Adaptive", color="#1f77b4", alpha=0.85)
        ax4.set_xticks(x_m)
        ax4.set_xticklabels(metrics_labels, fontsize=10)
        ax4.set_ylim(0.0, 1.1)
        ax4.set_ylabel("Score [0, 1]", fontsize=11)
        ax4.set_title(
            f"Held-Out Test Set Performance (Measurable Benefit: {comparison_report.is_benefit_measurable})",
            fontsize=11,
            fontweight="bold",
        )
        ax4.legend(loc="lower right", fontsize=10)
        ax4.grid(True, linestyle=":", alpha=0.6)
    else:
        # Fallback to trial objective score distribution
        ax4.hist(scores, bins=15, color="#17becf", alpha=0.7, edgecolor="black")
        ax4.axvline(opt_result.best_score, color="r", linestyle="-", linewidth=2, label=f"Best ({opt_result.best_score:.3f})")
        ax4.axvline(opt_result.baseline_score, color="k", linestyle="--", linewidth=2, label=f"Baseline ({opt_result.baseline_score:.3f})")
        ax4.set_xlabel("Objective Score", fontsize=11)
        ax4.set_ylabel("Trial Count", fontsize=11)
        ax4.set_title("Objective Score Distribution", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper left", fontsize=10)
        ax4.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved adaptive optimization summary figure to {out_p}")
    return out_p
