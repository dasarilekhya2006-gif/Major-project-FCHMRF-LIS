"""Publication and diagnostic visualization of comparative benchmark results."""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.logging import get_logger
from src.evaluation.comparative_runner import BenchmarkComparisonReport, TrialResult

logger = get_logger(__name__)


def plot_comparative_benchmark_summary(
    report: BenchmarkComparisonReport,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Generate 4-panel publication-grade summary figure of comparative benchmarks.

    Panels:
      1. Power (TPR / Sensitivity) vs. Empirical FDP with nominal alpha threshold.
      2. Spatial Localization Fidelity (Dice/F1, IoU, MCC).
      3. Computational Cost: Wall-Clock Runtime (s) & Peak Memory (MB).
      4. Representative Slice Visual Overlay (Ground Truth vs Methods).

    Args:
        report: BenchmarkComparisonReport from runner.
        output_path: Target PNG path.
        dpi: Image resolution.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = Path("outputs/figures/phase8_comparative_summary.png")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    methods = list(report.aggregated_metrics.keys())
    short_names = [
        m.replace("Benjamini-Hochberg (BH-FDR)", "BH-FDR")
         .replace("Local FDR (Efron lfdr)", "Local FDR")
         .replace("Original fcHMRF-LIS (lambda_dl=0)", "Original fcHMRF")
         .replace("ROI-Assisted fcHMRF-LIS (Fixed)", "ROI-fcHMRF (Fixed)")
         .replace("Adaptive ROI-Assisted fcHMRF-LIS (Optimized)", "Adaptive ROI-fcHMRF")
        for m in methods
    ]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    # --------------------------------------------------------------------------
    # Panel 1: Power (TPR) vs Empirical FDP
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.35

    tprs = [report.aggregated_metrics[m].mean_tpr for m in methods]
    tpr_sems = [report.aggregated_metrics[m].sem_tpr for m in methods]
    fdps = [report.aggregated_metrics[m].mean_fdp for m in methods]
    fdp_sems = [report.aggregated_metrics[m].sem_fdp for m in methods]

    rects1 = ax1.bar(x - width / 2, tprs, width, yerr=tpr_sems, label="Power (TPR / Sensitivity)", color="#1f77b4", alpha=0.85, capsize=4)
    rects2 = ax1.bar(x + width / 2, fdps, width, yerr=fdp_sems, label="Empirical FDP", color="#d62728", alpha=0.85, capsize=4)

    ax1.axhline(
        report.nominal_alpha,
        color="darkred",
        linestyle="--",
        linewidth=1.8,
        label=f"Nominal FDR (alpha = {report.nominal_alpha})",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Rate [0.0, 1.0]", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("Statistical Power vs. Empirical False Discovery Proportion", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel 2: Spatial Localization Fidelity (Dice, IoU, MCC)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    f1s = [report.aggregated_metrics[m].mean_f1 for m in methods]
    f1_sems = [report.aggregated_metrics[m].sem_f1 for m in methods]
    ious = [report.aggregated_metrics[m].mean_iou for m in methods]
    iou_sems = [report.aggregated_metrics[m].sem_iou for m in methods]
    mccs = [report.aggregated_metrics[m].mean_mcc for m in methods]
    mcc_sems = [report.aggregated_metrics[m].sem_mcc for m in methods]

    w_sub = 0.25
    ax2.bar(x - w_sub, f1s, w_sub, yerr=f1_sems, label="Dice (F1)", color="#2ca02c", alpha=0.85, capsize=3)
    ax2.bar(x, ious, w_sub, yerr=iou_sems, label="Jaccard (IoU)", color="#ff7f0e", alpha=0.85, capsize=3)
    ax2.bar(x + w_sub, mccs, w_sub, yerr=mcc_sems, label="MCC", color="#9467bd", alpha=0.85, capsize=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Score [0.0, 1.0]", fontsize=11)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Spatial Localization Accuracy & Overlap", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel 3: Computational Efficiency (Runtime & Peak RAM)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    runtimes = [report.aggregated_metrics[m].mean_runtime_seconds for m in methods]
    time_sems = [report.aggregated_metrics[m].sem_runtime_seconds for m in methods]
    mems = [report.aggregated_metrics[m].mean_peak_memory_mb for m in methods]
    mem_sems = [report.aggregated_metrics[m].sem_peak_memory_mb for m in methods]

    ax3.bar(x - width / 2, runtimes, width, yerr=time_sems, label="Runtime (s)", color="#8c564b", alpha=0.85, capsize=4)
    ax3.set_ylabel("Runtime (seconds)", color="#8c564b", fontsize=11)
    ax3.tick_params(axis="y", labelcolor="#8c564b")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width / 2, mems, width, yerr=mem_sems, label="Peak RAM (MB)", color="#e377c2", alpha=0.85, capsize=4)
    ax3_twin.set_ylabel("Peak RAM (MB)", color="#e377c2", fontsize=11)
    ax3_twin.tick_params(axis="y", labelcolor="#e377c2")
    ax3.set_title("Computational Cost: Wall-Clock Time & Peak Memory", fontsize=12, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel 4: Representative Slice Spatial Discovery Maps
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    if report.trial_results:
        rep_trial = report.trial_results[0]
        slice_z = rep_trial.volume.shape[2] // 2
        gt_slice = rep_trial.volume.ground_truth[:, :, slice_z]
        z_slice = rep_trial.volume.z_scores[:, :, slice_z]

        # Multi-slice visualization in a grid
        ax4.axis("off")
        inner_gs = gs[1, 1].subgridspec(2, 3, wspace=0.15, hspace=0.25)

        # Plot Ground Truth
        ax_gt = fig.add_subplot(inner_gs[0, 0])
        ax_gt.imshow(z_slice.T, cmap="gray", origin="lower")
        ax_gt.contour(gt_slice.T, levels=[0.5], colors=["lime"], linewidths=1.8)
        ax_gt.set_title("Ground Truth", fontsize=10, fontweight="bold")
        ax_gt.axis("off")

        # Plot each method's detections
        plot_positions = [
            (0, 1),  # BH-FDR
            (0, 2),  # Local FDR
            (1, 0),  # Original fcHMRF
            (1, 1),  # ROI fcHMRF
            (1, 2),  # Adaptive fcHMRF
        ]

        for method_idx, (r, c) in enumerate(plot_positions):
            if method_idx < len(methods):
                m_name = methods[method_idx]
                s_name = short_names[method_idx]
                sig_slice = rep_trial.method_outputs[m_name].significant_mask[:, :, slice_z]

                ax_m = fig.add_subplot(inner_gs[r, c])
                ax_m.imshow(z_slice.T, cmap="gray", origin="lower")
                # True positives: Green, False positives: Red, Missed: Blue
                tp_slice = sig_slice & gt_slice
                fp_slice = sig_slice & ~gt_slice

                if np.any(tp_slice):
                    ax_m.contour(tp_slice.T, levels=[0.5], colors=["lime"], linewidths=1.5)
                if np.any(fp_slice):
                    ax_m.contour(fp_slice.T, levels=[0.5], colors=["red"], linewidths=1.5)

                ax_m.set_title(s_name, fontsize=9, fontweight="bold")
                ax_m.axis("off")
    else:
        ax4.text(0.5, 0.5, "No trial data available", ha="center", va="center")
        ax4.axis("off")

    fig.suptitle(
        f"Phase 8 Benchmark: Rigorous Multi-Method Comparison (N={report.num_trials} Trials, alpha={report.nominal_alpha})",
        fontsize=15,
        fontweight="bold",
    )

    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved publication benchmark summary figure to {out_p}")
    return out_p
