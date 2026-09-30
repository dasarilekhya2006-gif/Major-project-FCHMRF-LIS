"""Statistical plots for evaluating spatial FDR control and power calibration."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.core.logging import get_logger

logger = get_logger(__name__)


def plot_fdr_power_calibration(
    nominal_alphas: List[float],
    empirical_fdrs: Dict[str, List[float]],
    powers: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Nominal Alpha vs Empirical FDR calibration and Statistical Power.

    Validates whether methods strictly respect the nominal control line (FDR <= alpha).
    """
    fig, (ax_fdr, ax_pwr) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. FDR Control Calibration
    ax_fdr.plot(
        nominal_alphas,
        nominal_alphas,
        "k--",
        label="Nominal Control (y = x)",
        linewidth=1.5,
    )
    for method, fdrs in empirical_fdrs.items():
        ax_fdr.plot(nominal_alphas, fdrs, marker="o", label=method, linewidth=2)

    ax_fdr.set_xlabel("Nominal Alpha Level", fontsize=11)
    ax_fdr.set_ylabel("Empirical FDR", fontsize=11)
    ax_fdr.set_title("FDR Control Calibration", fontsize=12, fontweight="bold")
    ax_fdr.grid(True, linestyle=":", alpha=0.6)
    ax_fdr.legend(fontsize=9)

    # 2. Statistical Power (TPR)
    for method, pwrs in powers.items():
        ax_pwr.plot(nominal_alphas, pwrs, marker="s", label=method, linewidth=2)

    ax_pwr.set_xlabel("Nominal Alpha Level", fontsize=11)
    ax_pwr.set_ylabel("True Positive Rate (Power)", fontsize=11)
    ax_pwr.set_title("Statistical Power Comparison", fontsize=12, fontweight="bold")
    ax_pwr.grid(True, linestyle=":", alpha=0.6)
    ax_pwr.legend(fontsize=9)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        logger.info(f"Saved FDR/power calibration plot to {output_path}")

    return fig


def plot_lis_distribution(
    lis_values: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot distribution of Local Index of Significance (LIS) values."""
    fig, ax = plt.subplots(figsize=(6, 4))

    if ground_truth is not None:
        lis_null = lis_values[~ground_truth]
        lis_active = lis_values[ground_truth]
        ax.hist(
            lis_null,
            bins=40,
            density=True,
            alpha=0.6,
            color="steelblue",
            label="True Null (theta=0)",
        )
        ax.hist(
            lis_active,
            bins=40,
            density=True,
            alpha=0.6,
            color="crimson",
            label="True Active (theta=1)",
        )
    else:
        ax.hist(lis_values, bins=40, density=True, color="purple", alpha=0.7)

    ax.set_xlabel("Local Index of Significance (LIS)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of LIS Statistic", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig
