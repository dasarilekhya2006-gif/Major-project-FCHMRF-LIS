"""Spatial map overlays comparing side-by-side method discoveries."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.core.logging import get_logger

logger = get_logger(__name__)


def plot_side_by_side_discoveries(
    z_map: np.ndarray,
    ground_truth: np.ndarray,
    method_discoveries: Dict[str, np.ndarray],
    slice_z: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot an axial slice comparing ground truth vs multiple FDR controlling methods.

    Args:
        z_map: 3D z-score volume.
        ground_truth: 3D binary ground truth mask.
        method_discoveries: Dict mapping method name to 3D boolean discovery mask.
        slice_z: Axial slice index (default: center slice).
        output_path: Optional output path.

    Returns:
        plt.Figure: Created figure.
    """
    if slice_z is None:
        slice_z = z_map.shape[2] // 2

    num_panels = 2 + len(method_discoveries)
    fig, axes = plt.subplots(1, num_panels, figsize=(3.5 * num_panels, 3.5))

    # 1. Raw z-score map
    im0 = axes[0].imshow(np.rot90(z_map[:, :, slice_z]), cmap="cold_hot" if "cold_hot" in plt.colormaps() else "bwr")
    axes[0].set_title(f"Raw Z-map (z={slice_z})", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Ground truth
    axes[1].imshow(np.rot90(ground_truth[:, :, slice_z]), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Active", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # 3. Method discovery panels
    for idx, (method_name, disc_mask) in enumerate(method_discoveries.items(), start=2):
        ax = axes[idx]
        ax.imshow(np.rot90(disc_mask[:, :, slice_z]), cmap="Reds", vmin=0, vmax=1)
        ax.set_title(method_name, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        logger.info(f"Saved side-by-side discoveries map to {output_path}")

    return fig
