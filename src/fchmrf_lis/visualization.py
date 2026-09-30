"""Diagnostic and publication visualization utilities for fcHMRF-LIS spatial inference."""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.core.logging import get_logger
from src.fchmrf_lis.fchmrf_model import FCHMRFResult

logger = get_logger(__name__)


def plot_fchmrf_summary(
    result: FCHMRFResult,
    z_scores: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    slice_idx: Optional[int] = None,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """Generate a comprehensive 6-panel diagnostic and publication figure for fcHMRF-LIS.

    Panels:
      1. Input Test Statistics (z-scores) slice
      2. Fitted Two-Group Emission Mixture Density f(z) vs Empirical Histogram
      3. Posterior Activation Probability Q(theta_i = 1) Map
      4. Local Index of Significance (LIS) Map
      5. FDR-Controlled Significant Discoveries Overlay
      6. Ranked LIS Curve with Spatial FDR Cutoff Threshold

    Args:
        result: FCHMRFResult container from FCHMRFModel.fit_predict().
        z_scores: 3D array of input statistical z-scores.
        output_path: Destination PNG image path.
        slice_idx: Optional axial slice index (defaults to slice with peak discoveries).
        title: Figure suptitle.
        dpi: Image resolution.

    Returns:
        Path: Path to saved figure.
    """
    if output_path is None:
        output_path = Path("outputs/figures/fchmrf_lis_summary.png")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    shape = result.lis_map.shape
    # Determine best axial slice: peak discoveries or volume center
    if slice_idx is None:
        axial_discoveries = np.sum(result.significant_mask, axis=(0, 1))
        if np.any(axial_discoveries > 0):
            z_slice = int(np.argmax(axial_discoveries))
        else:
            z_slice = shape[2] // 2
    else:
        z_slice = np.clip(slice_idx, 0, shape[2] - 1)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    else:
        mode_str = "Original fcHMRF-LIS" if result.method_mode == "original_fchmrf" else "Adaptive DL-fcHMRF-LIS"
        fig.suptitle(
            f"{mode_str} | Discoveries: {result.num_discoveries:,} / {result.num_analyzed_voxels:,} "
            f"({100.0 * result.num_discoveries / max(1, result.num_analyzed_voxels):.1f}%) | "
            f"Nominal FDR: {result.fdr_results['nominal_alpha']}",
            fontsize=15,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Panel 1: Input z-scores
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    z_slice_img = np.rot90(z_scores[:, :, z_slice])
    vmax = float(np.percentile(np.abs(z_scores[result.spatial_mask]), 99)) if np.any(result.spatial_mask) else 3.0
    im1 = ax1.imshow(z_slice_img, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax1.set_title(f"Input z-Score Map (Axial z={z_slice})", fontsize=12, fontweight="bold")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)

    # -------------------------------------------------------------
    # Panel 2: Two-Group Mixture Model Fit
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    active_z = z_scores[result.spatial_mask]
    bins = np.linspace(float(np.min(active_z)), float(np.max(active_z)), 60)
    ax2.hist(active_z, bins=bins, density=True, alpha=0.35, color="gray", label="Empirical z")

    mp = result.model_parameters
    x_axis = np.linspace(bins[0], bins[-1], 300)
    f0 = mp["p0"] * stats.norm.pdf(x_axis, loc=mp["mu0"], scale=mp["sigma0"])
    f1 = (1.0 - mp["p0"]) * stats.norm.pdf(x_axis, loc=mp["mu1"], scale=mp["sigma1"])
    f_total = f0 + f1

    ax2.plot(x_axis, f0, "b--", linewidth=1.8, label=r"Null $p_0 f_0(z)$")
    ax2.plot(x_axis, f1, "r--", linewidth=1.8, label=r"Active $(1-p_0) f_1(z)$")
    ax2.plot(x_axis, f_total, "k-", linewidth=2.0, label=r"Total $f(z)$")
    ax2.set_title("Emission Mixture Density Fit", fontsize=12, fontweight="bold")
    ax2.set_xlabel("z-score", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 3: Posterior Activation Probability Q(theta_i = 1)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    prob_slice_img = np.rot90(result.posterior_prob[:, :, z_slice])
    im3 = ax3.imshow(prob_slice_img, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax3.set_title(r"Posterior Active Marginal $Q(\theta_i=1)$", fontsize=12, fontweight="bold")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, shrink=0.7, pad=0.02)

    # -------------------------------------------------------------
    # Panel 4: Local Index of Significance (LIS) Map
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    lis_slice_img = np.rot90(result.lis_map[:, :, z_slice])
    im4 = ax4.imshow(lis_slice_img, cmap="plasma_r", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax4.set_title(r"Local Index of Significance (LIS)", fontsize=12, fontweight="bold")
    ax4.axis("off")
    fig.colorbar(im4, ax=ax4, shrink=0.7, pad=0.02)

    # -------------------------------------------------------------
    # Panel 5: Significant Discoveries Overlay
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    sig_slice_img = np.rot90(result.significant_mask[:, :, z_slice])
    # Background anatomical / contrast grayscale
    ax5.imshow(z_slice_img, cmap="gray", alpha=0.7, interpolation="nearest")
    if np.any(sig_slice_img > 0):
        # Red discovery overlay with yellow boundary contour
        ax5.imshow(np.ma.masked_where(sig_slice_img == 0, sig_slice_img), cmap="autumn", alpha=0.85, interpolation="nearest")
        ax5.contour(sig_slice_img, levels=[0.5], colors=["yellow"], linewidths=1.5)
    ax5.set_title(f"Discoveries (Spatial FDR < {result.fdr_results['nominal_alpha']})", fontsize=12, fontweight="bold")
    ax5.axis("off")

    # -------------------------------------------------------------
    # Panel 6: Ranked LIS and Step-Up Cutoff
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_lis = np.sort(result.lis_vector)
    cum_mean_lis = np.cumsum(sorted_lis) / np.arange(1, len(sorted_lis) + 1)
    k_discoveries = result.num_discoveries
    alpha = result.fdr_results["nominal_alpha"]

    ranks_to_plot = min(len(sorted_lis), max(50, int(k_discoveries * 2.5)))
    ax6.plot(np.arange(1, ranks_to_plot + 1), sorted_lis[:ranks_to_plot], "b-", label=r"Sorted $LIS_{(i)}$")
    ax6.plot(np.arange(1, ranks_to_plot + 1), cum_mean_lis[:ranks_to_plot], "g--", label=r"Cumulative Mean $\widehat{FDR}$")
    ax6.axhline(alpha, color="r", linestyle=":", linewidth=1.5, label=rf"Target $\alpha = {alpha}$")

    if k_discoveries > 0:
        ax6.axvline(k_discoveries, color="m", linestyle="-.", linewidth=1.5, label=f"Cutoff $k = {k_discoveries}$")

    ax6.set_title("LIS Step-Up Ranking & Cutoff", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Voxel Rank (Ascending LIS)", fontsize=10)
    ax6.set_ylabel("Probability", fontsize=10)
    ax6.legend(fontsize=9, loc="lower right")
    ax6.grid(True, linestyle=":", alpha=0.5)

    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved fcHMRF-LIS summary visualization to {out_p}")
    return out_p
