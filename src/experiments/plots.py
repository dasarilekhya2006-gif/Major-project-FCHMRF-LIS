"""Automated publication-quality visualization and diagnostic plotting for experiments."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.schemas import ExperimentResult


def plot_single_experiment_diagnostic(
    result: ExperimentResult,
    mri_volume: np.ndarray,
    roi_mask: Optional[np.ndarray],
    z_scores: np.ndarray,
    lis_map: Optional[np.ndarray],
    sig_mask: np.ndarray,
    ground_truth: Optional[np.ndarray],
    output_path: Union[str, Path],
    slice_idx: Optional[int] = None,
) -> Path:
    """Generate a multi-panel publication diagnostic figure for a single experiment run.

    Panels:
      1. Structural / Synthetic Brain Volume with Ground Truth Overlay
      2. 3D U-Net Prior / ROI Localization Mask
      3. Voxel-Level Statistical Test Map (z-scores)
      4. Local Index of Significance (LIS) Map
      5. FDR-Controlled Significant Discoveries in Temperature Colormap
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if slice_idx is None:
        slice_idx = mri_volume.shape[2] // 2

    # Extract axial slice for each volume
    mri_slice = mri_volume[:, :, slice_idx]
    z_slice = z_scores[:, :, slice_idx]
    sig_slice = sig_mask[:, :, slice_idx]
    gt_slice = ground_truth[:, :, slice_idx] if ground_truth is not None else None
    roi_slice = roi_mask[:, :, slice_idx] if roi_mask is not None else None
    lis_slice = lis_map[:, :, slice_idx] if lis_map is not None else None

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor="#0f172a")

    for ax in axes:
        ax.set_facecolor("#020617")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Panel 1: MRI + Ground Truth
    im1 = axes[0].imshow(mri_slice.T, cmap="gray", origin="lower")
    if gt_slice is not None and np.any(gt_slice > 0):
        axes[0].contour(gt_slice.T, levels=[0.5], colors=["#10b981"], linewidths=1.8)
    axes[0].set_title("1. Structural MRI + GT", color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8", labelsize=7)

    # Panel 2: 3D U-Net ROI Prior
    if roi_slice is not None:
        im2 = axes[1].imshow(roi_slice.T, cmap="Blues", origin="lower", vmin=0, vmax=1)
        axes[1].set_title("2. 3D U-Net Prior", color="#38bdf8", fontsize=10, fontweight="bold", pad=8)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8", labelsize=7)
    else:
        axes[1].text(0.5, 0.5, "No ROI Prior", color="#64748b", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("2. 3D U-Net Prior (N/A)", color="#64748b", fontsize=10, pad=8)

    # Panel 3: Statistical Map (z-scores)
    vmax_z = max(3.5, float(np.max(np.abs(z_slice)))) if np.max(np.abs(z_slice)) > 0 else 3.5
    im3 = axes[2].imshow(z_slice.T, cmap="coolwarm", origin="lower", vmin=-vmax_z, vmax=vmax_z)
    axes[2].set_title("3. Voxel Statistics (Z)", color="#f59e0b", fontsize=10, fontweight="bold", pad=8)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8", labelsize=7)

    # Panel 4: Local Index of Significance (LIS)
    if lis_slice is not None:
        im4 = axes[3].imshow(lis_slice.T, cmap="viridis_r", origin="lower", vmin=0.0, vmax=1.0)
        axes[3].set_title("4. LIS Map P(H0|Z)", color="#a855f7", fontsize=10, fontweight="bold", pad=8)
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8", labelsize=7)
    else:
        axes[3].text(0.5, 0.5, "No LIS Map", color="#64748b", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title("4. LIS Map (N/A)", color="#64748b", fontsize=10, pad=8)

    # Panel 5: FDR-Controlled Significant Discoveries
    axes[4].imshow(mri_slice.T, cmap="gray", origin="lower", alpha=0.6)
    if np.any(sig_slice > 0):
        # Masked overlay with temperature colormap
        masked_sig = np.ma.masked_where(sig_slice.T == 0, sig_slice.T)
        im5 = axes[4].imshow(masked_sig, cmap="YlOrRd", origin="lower", vmin=0.5, vmax=1.5)
        if gt_slice is not None:
            axes[4].contour(gt_slice.T, levels=[0.5], colors=["#10b981"], linewidths=1.2, linestyles="--")
    axes[4].set_title(
        f"5. FDR Discoveries (k={int(np.sum(sig_mask))})",
        color="#fb923c",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    # Global Super Title with Metrics
    m = result.metrics
    tpr_str = f"TPR: {m.get('tpr', 0.0):.3f}" if "tpr" in m else ""
    fdr_str = f"FDR: {m.get('empirical_fdr', 0.0):.3f}" if "empirical_fdr" in m else ""
    dice_str = f"Dice: {m.get('dice', 0.0):.3f}" if "dice" in m else ""
    metrics_line = " | ".join(filter(None, [tpr_str, fdr_str, dice_str, f"Runtime: {result.total_runtime:.2f}s"]))

    plt.suptitle(
        f"Experiment: {result.experiment_id} ({result.config.name})\n[{metrics_line}]",
        color="#ffffff",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_suite_summary_figures(
    results: List[ExperimentResult],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Generate aggregated comparison plots across all experiments in a suite."""
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    if not results:
        return generated_plots

    # Filter successful runs
    success_runs = [r for r in results if r.status == "completed" and r.metrics]
    if not success_runs:
        return generated_plots

    exp_ids = [r.experiment_id for r in success_runs]
    method_labels = [r.config.fchmrf.method_mode for r in success_runs]

    # 1. Metrics Comparison Bar Chart (TPR / Power, Empirical FDR, Dice, Precision)
    tprs = [r.metrics.get("tpr", 0.0) for r in success_runs]
    fdrs = [r.metrics.get("empirical_fdr", 0.0) for r in success_runs]
    dices = [r.metrics.get("dice", 0.0) for r in success_runs]
    precisions = [r.metrics.get("precision", 0.0) for r in success_runs]
    nominal_alpha = success_runs[0].config.fchmrf.alpha

    fig, ax = plt.subplots(figsize=(max(8, len(success_runs) * 1.5), 5), facecolor="#0f172a")
    ax.set_facecolor("#020617")

    x = np.arange(len(success_runs))
    width = 0.2

    rects1 = ax.bar(x - 1.5 * width, tprs, width, label="Power (TPR)", color="#38bdf8")
    rects2 = ax.bar(x - 0.5 * width, fdrs, width, label="Empirical FDR", color="#f43f5e")
    rects3 = ax.bar(x + 0.5 * width, dices, width, label="Dice Similarity", color="#10b981")
    rects4 = ax.bar(x + 1.5 * width, precisions, width, label="Precision (PPV)", color="#fbbf24")

    # Add nominal alpha reference line
    ax.axhline(
        nominal_alpha,
        color="#f43f5e",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Nominal FDR α = {nominal_alpha}",
    )

    ax.set_ylabel("Score / Rate", color="#e2e8f0", fontsize=10)
    ax.set_title("Experimental Comparison: Detection Power, FDR Control, & Spatial Fidelity", color="#ffffff", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=35, ha="right", color="#94a3b8", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(axis="y", color="#1e293b", linestyle=":", alpha=0.7)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    bar_path = out_d / "suite_metrics_comparison_bar.png"
    fig.savefig(bar_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    generated_plots["metrics_bar"] = bar_path

    # 2. Runtime Breakdown Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(max(8, len(success_runs) * 1.5), 4.5), facecolor="#0f172a")
    ax.set_facecolor("#020617")

    stage_keys = ["preprocessing", "unet_inference", "statistical_testing", "adaptive_optimization", "fchmrf_inference"]
    stage_labels = ["Preprocessing", "3D U-Net", "Statistics", "Adaptive Tuning", "fcHMRF-LIS"]
    stage_colors = ["#64748b", "#06b6d4", "#f59e0b", "#ec4899", "#f97316"]

    bottoms = np.zeros(len(success_runs))
    for key, label, color in zip(stage_keys, stage_labels, stage_colors):
        values = np.array([r.runtime_breakdown.get(key, 0.0) for r in success_runs])
        ax.bar(x, values, 0.45, bottom=bottoms, label=label, color=color)
        bottoms += values

    ax.set_ylabel("Execution Time (seconds)", color="#e2e8f0", fontsize=10)
    ax.set_title("Stage-by-Stage Computational Latency Breakdown", color="#ffffff", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=35, ha="right", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(axis="y", color="#1e293b", linestyle=":", alpha=0.7)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    runtime_path = out_d / "suite_runtime_breakdown_bar.png"
    fig.savefig(runtime_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    generated_plots["runtime_bar"] = runtime_path

    # 3. FDR vs Power (TPR) Scatter / Calibration Plot
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor="#0f172a")
    ax.set_facecolor("#020617")

    for r in success_runs:
        fdr_val = r.metrics.get("empirical_fdr", 0.0)
        tpr_val = r.metrics.get("tpr", 0.0)
        mode = r.config.fchmrf.method_mode
        color = "#f97316" if "proposed" in mode else "#3b82f6"
        marker = "o" if "adaptive" not in mode else "*"
        ax.scatter(fdr_val, tpr_val, color=color, s=120, marker=marker, edgecolors="#ffffff", linewidth=0.8, label=r.experiment_id)
        ax.annotate(r.experiment_id, (fdr_val + 0.005, tpr_val), color="#cbd5e1", fontsize=7)

    ax.axvline(nominal_alpha, color="#ef4444", linestyle="--", linewidth=1.2, label=f"Target α = {nominal_alpha}")
    ax.set_xlabel("Empirical False Discovery Rate (FDR)", color="#e2e8f0", fontsize=10)
    ax.set_ylabel("True Positive Rate / Power (TPR)", color="#e2e8f0", fontsize=10)
    ax.set_title("FDR Control vs Detection Power Calibration", color="#ffffff", fontsize=11, pad=12)
    ax.set_xlim(-0.02, max(0.2, max(fdrs) + 0.05))
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(color="#1e293b", linestyle=":", alpha=0.7)

    plt.tight_layout()
    fdr_scatter_path = out_d / "suite_fdr_power_calibration.png"
    fig.savefig(fdr_scatter_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    generated_plots["fdr_power_scatter"] = fdr_scatter_path

    return generated_plots
