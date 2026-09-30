"""Orthogonal slice visualization (axial, sagittal, coronal) for neuroimaging volumes."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.core.logging import get_logger

logger = get_logger(__name__)


def plot_orthogonal_slices(
    volume: np.ndarray,
    overlay_mask: Optional[np.ndarray] = None,
    cut_coords: Optional[Tuple[int, int, int]] = None,
    title: str = "Neuroimaging Slice Orthoview",
    output_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "gray",
    overlay_color: str = "red",
    overlay_alpha: float = 0.5,
) -> plt.Figure:
    """Generate a 3-panel orthogonal slice viewer (Sagittal, Coronal, Axial).

    Args:
        volume: 3D numpy array of image or z-scores.
        overlay_mask: Optional 3D boolean mask (e.g. thresholded active discoveries).
        cut_coords: (x, y, z) voxel indices for slicing. If None, center slices are used.
        title: Figure title.
        output_path: Optional path to save the generated image.
        vmin, vmax: Color scaling limits.
        cmap: Base colormap.
        overlay_color: Color for discoveries overlay.
        overlay_alpha: Transparency for overlay.

    Returns:
        plt.Figure: The created Matplotlib figure.
    """
    nx, ny, nz = volume.shape
    if cut_coords is None:
        cut_coords = (nx // 2, ny // 2, nz // 2)

    cx, cy, cz = cut_coords

    # Slices
    # Sagittal (YZ at x = cx)
    sag_vol = np.rot90(volume[cx, :, :])
    # Coronal (XZ at y = cy)
    cor_vol = np.rot90(volume[:, cy, :])
    # Axial (XY at z = cz)
    axi_vol = np.rot90(volume[:, :, cz])

    if vmin is None:
        vmin = float(np.percentile(volume, 1))
    if vmax is None:
        vmax = float(np.percentile(volume, 99))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    views = [
        (sag_vol, f"Sagittal (x={cx})", overlay_mask[cx, :, :] if overlay_mask is not None else None),
        (cor_vol, f"Coronal (y={cy})", overlay_mask[:, cy, :] if overlay_mask is not None else None),
        (axi_vol, f"Axial (z={cz})", overlay_mask[:, :, cz] if overlay_mask is not None else None),
    ]

    for ax, (slc, view_name, msk) in zip(axes, views):
        im = ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax)
        if msk is not None:
            msk_rot = np.rot90(msk)
            # Create colored overlay where mask is True
            overlay_rgba = np.zeros((*msk_rot.shape, 4), dtype=np.float32)
            if overlay_color == "red":
                overlay_rgba[msk_rot] = [1.0, 0.0, 0.0, overlay_alpha]
            elif overlay_color == "yellow":
                overlay_rgba[msk_rot] = [1.0, 1.0, 0.0, overlay_alpha]
            else:
                overlay_rgba[msk_rot] = [0.0, 1.0, 0.0, overlay_alpha]
            ax.imshow(overlay_rgba)

        ax.set_title(view_name, fontsize=11)
        ax.axis("off")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        logger.info(f"Saved orthogonal slice figure to {output_path}")

    return fig
