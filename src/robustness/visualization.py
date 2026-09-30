"""Diagnostic and publication-grade visualization of robustness curves and uncertainty maps."""

from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.logging import get_logger
from src.robustness.evaluator import RobustnessEvaluationReport

logger = get_logger(__name__)


def plot_robustness_analysis_summary(
    report: RobustnessEvaluationReport,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Generate 4-panel publication-grade robustness and uncertainty figure.

    Panels:
      1. SNR Sensitivity Curves: Power (TPR) & Empirical FDP vs. SNR.
      2. Input Data Perturbation Decay: Stability (Pairwise Jaccard) vs. Additive Noise.
      3. Spatial Correlation Resilience: Stability & Power vs. Smoothing Bandwidth.
      4. Spatial Uncertainty Maps: Empirical Detection Probability & Boundary Entropy.

    Args:
        report: RobustnessEvaluationReport instance.
        output_path: Target PNG path.
        dpi: Resolution.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        output_path = Path("outputs/figures/phase9_robustness_summary.png")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # --------------------------------------------------------------------------
    # Panel 1: SNR Sensitivity Curves (Power & FDP vs SNR)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    snrs = [m.condition_value for m in report.snr_sweep_metrics]
    tprs = [m.mean_tpr for m in report.snr_sweep_metrics]
    fdps = [m.mean_fdp for m in report.snr_sweep_metrics]
    jaccs_snr = [m.mean_pairwise_jaccard for m in report.snr_sweep_metrics]

    line1 = ax1.plot(snrs, tprs, marker="o", linewidth=2.2, color="#1f77b4", label="Power (TPR / Sensitivity)")
    line2 = ax1.plot(snrs, jaccs_snr, marker="s", linewidth=2.0, linestyle="--", color="#2ca02c", label="Pairwise Jaccard Stability")
    line3 = ax1.plot(snrs, fdps, marker="^", linewidth=2.0, color="#d62728", label="Empirical FDP")

    ax1.axhline(
        report.nominal_alpha,
        color="darkred",
        linestyle=":",
        linewidth=1.8,
        label=f"Nominal FDR (alpha = {report.nominal_alpha})",
    )
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Metric Value [0.0, 1.0]", fontsize=11)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_title("Detection Sensitivity & FDR Control vs. Signal Strength (SNR)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="center right", fontsize=10)

    # --------------------------------------------------------------------------
    # Panel 2: Input Perturbation Decay (Stability vs Additive Noise)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    sigmas_pert = [m.condition_value for m in report.perturbation_sweep_metrics]
    jaccs_pert = [m.mean_pairwise_jaccard for m in report.perturbation_sweep_metrics]
    jacc_stds = [m.std_pairwise_jaccard for m in report.perturbation_sweep_metrics]
    cvs = [m.discovery_count_cv * 100.0 for m in report.perturbation_sweep_metrics]

    ax2.errorbar(
        sigmas_pert,
        jaccs_pert,
        yerr=jacc_stds,
        marker="o",
        linewidth=2.2,
        color="#9467bd",
        capsize=4,
        label="Pairwise Jaccard Stability",
    )
    ax2.set_xlabel("Input Perturbation Noise Sigma", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Pairwise Jaccard [0.0, 1.0]", color="#9467bd", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(sigmas_pert, cvs, marker="d", linewidth=2.0, linestyle="--", color="#e377c2", label="Discovery Count CV (%)")
    ax2_twin.set_ylabel("Discovery Count CV (%)", color="#e377c2", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#e377c2")
    ax2.set_title("Stability Degradation Under Additive Input Jitter", fontsize=12, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel 3: Spatial Autocorrelation Robustness
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    sigmas_corr = [m.condition_value for m in report.spatial_corr_metrics]
    jaccs_corr = [m.mean_pairwise_jaccard for m in report.spatial_corr_metrics]
    tprs_corr = [m.mean_tpr for m in report.spatial_corr_metrics]
    fdps_corr = [m.mean_fdp for m in report.spatial_corr_metrics]

    ax3.plot(sigmas_corr, jaccs_corr, marker="s", linewidth=2.2, color="#2ca02c", label="Pairwise Jaccard Stability")
    ax3.plot(sigmas_corr, tprs_corr, marker="o", linewidth=2.2, color="#1f77b4", label="Power (TPR)")
    ax3.plot(sigmas_corr, fdps_corr, marker="^", linewidth=2.0, color="#d62728", label="Empirical FDP")
    ax3.axhline(
        report.nominal_alpha,
        color="darkred",
        linestyle=":",
        linewidth=1.8,
        label=f"Nominal FDR (alpha = {report.nominal_alpha})",
    )
    ax3.set_xlabel("Background Spatial Smoothing Sigma (voxels)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Rate / Overlap [0.0, 1.0]", fontsize=11)
    ax3.set_ylim(-0.02, 1.05)
    ax3.set_title("Performance vs. Spatial Autocorrelation Bandwidth", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=10)

    # --------------------------------------------------------------------------
    # Panel 4: Spatial Uncertainty & Boundary Entropy Maps
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    slice_z = report.detection_probability_map.shape[2] // 2
    prob_slice = report.detection_probability_map[:, :, slice_z]
    ent_slice = report.uncertainty_entropy_map[:, :, slice_z]

    inner_gs = gs[1, 1].subgridspec(1, 2, wspace=0.25)

    # 4A: Empirical Detection Probability
    ax4a = fig.add_subplot(inner_gs[0, 0])
    im1 = ax4a.imshow(prob_slice.T, cmap="viridis", origin="lower", vmin=0.0, vmax=1.0)
    ax4a.set_title("Detection Probability P_detect", fontsize=11, fontweight="bold")
    ax4a.axis("off")
    cbar1 = plt.colorbar(im1, ax=ax4a, fraction=0.046, pad=0.04)
    cbar1.set_label("Probability", fontsize=9)

    # 4B: Boundary Shannon Entropy
    ax4b = fig.add_subplot(inner_gs[0, 1])
    im2 = ax4b.imshow(ent_slice.T, cmap="inferno", origin="lower", vmin=0.0, vmax=1.0)
    ax4b.set_title("Boundary Uncertainty Entropy H", fontsize=11, fontweight="bold")
    ax4b.axis("off")
    cbar2 = plt.colorbar(im2, ax=ax4b, fraction=0.046, pad=0.04)
    cbar2.set_label("Entropy (bits)", fontsize=9)

    fig.suptitle(
        f"Phase 9: Comprehensive Robustness and Uncertainty Evaluation (alpha = {report.nominal_alpha})",
        fontsize=15,
        fontweight="bold",
    )

    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved publication robustness summary figure to {out_p}")
    return out_p
