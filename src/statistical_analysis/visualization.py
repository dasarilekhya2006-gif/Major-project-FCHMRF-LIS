"""Publication-grade visualization utilities for voxel-level neuroimaging statistics."""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.logging import get_logger
from src.statistical_analysis.voxel_testing import StatisticalMaps

logger = get_logger(__name__)


def plot_statistical_maps(
    stat_maps: StatisticalMaps,
    output_path: Optional[Union[str, Path]] = None,
    roi_mask: Optional[np.ndarray] = None,
    slices: Optional[Tuple[int, int, int]] = None,
    p_threshold: float = 0.05,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """Render a comprehensive 3-row orthogonal slice figure displaying voxel-level statistics.

    Row 1: Test Statistic Map (t, F, or z-statistic)
    Row 2: -log10(p-value) Significance Map (with p < p_threshold contour)
    Row 3: Effect Size Map (Cohen's d or contrast beta) with ROI Boundary Overlay

    Args:
        stat_maps: StatisticalMaps container from compute_statistics.
        output_path: Destination PNG image file path.
        roi_mask: Optional 3D binary array showing ROI boundary.
        slices: Optional (sagittal, coronal, axial) slice indices.
        p_threshold: Nominal significance threshold line (default 0.05 -> -log10(p) = 1.301).
        title: Overall figure title.
        dpi: Figure resolution.

    Returns:
        Path: Path to the generated figure.
    """
    if output_path is None:
        output_path = Path("outputs/figures/statistical_maps_summary.png")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    shape = stat_maps.spatial_shape

    # Determine slice indices: use peak significance if available, otherwise volume center
    if slices is None:
        if np.any(stat_maps.neg_log10_p > 0.0):
            max_idx = np.unravel_index(np.argmax(stat_maps.neg_log10_p), shape)
            s_idx, c_idx, a_idx = int(max_idx[0]), int(max_idx[1]), int(max_idx[2])
        else:
            s_idx, c_idx, a_idx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    else:
        s_idx, c_idx, a_idx = slices

    s_idx = np.clip(s_idx, 0, shape[0] - 1)
    c_idx = np.clip(c_idx, 0, shape[1] - 1)
    a_idx = np.clip(a_idx, 0, shape[2] - 1)

    # Prepare data rows
    rows_data = [
        ("Test Statistic (t / z)", stat_maps.test_statistic, "coolwarm", True),
        (r"Significance: $-\log_{10}(p)$", stat_maps.neg_log10_p, "inferno", False),
        ("Effect Size (Cohen's d / Contrast)", stat_maps.effect_sizes, "viridis", True),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    else:
        fig.suptitle(
            f"Voxel-Level Statistical Testing ({stat_maps.test_type.upper()}) | Active Voxels: {stat_maps.num_active_voxels:,}",
            fontsize=15,
            fontweight="bold",
        )

    # Reference ROI mask for overlay
    active_roi = roi_mask if roi_mask is not None else stat_maps.mask

    for r_idx, (label, data_field, cmap_name, is_diverging) in enumerate(rows_data):
        # Determine color limits
        valid_vals = data_field[stat_maps.mask]
        if len(valid_vals) > 0 and np.any(valid_vals != 0):
            if is_diverging:
                vmax = float(np.percentile(np.abs(valid_vals), 99.5))
                vmin = -vmax
            else:
                vmin = 0.0
                vmax = float(np.percentile(valid_vals, 99.5))
        else:
            vmin, vmax = -1.0, 1.0

        # Slices across orthogonal planes
        slice_sag = np.rot90(data_field[s_idx, :, :])
        slice_cor = np.rot90(data_field[:, c_idx, :])
        slice_ax = np.rot90(data_field[:, :, a_idx])

        # ROI mask slices
        roi_sag = np.rot90(active_roi[s_idx, :, :])
        roi_cor = np.rot90(active_roi[:, c_idx, :])
        roi_ax = np.rot90(active_roi[:, :, a_idx])

        # Plot 3 planes
        planes = [
            (slice_sag, roi_sag, f"Sagittal (x={s_idx})"),
            (slice_cor, roi_cor, f"Coronal (y={c_idx})"),
            (slice_ax, roi_ax, f"Axial (z={a_idx})"),
        ]

        for c_col, (slice_img, roi_img, plane_name) in enumerate(planes):
            ax = axes[r_idx, c_col]
            im = ax.imshow(slice_img, cmap=cmap_name, vmin=vmin, vmax=vmax, interpolation="nearest")

            # Overlay ROI boundary contour on Row 3
            if r_idx == 2 and np.any(roi_img > 0):
                ax.contour(roi_img, levels=[0.5], colors=["lime"], linewidths=1.5)

            # Overlay p < 0.05 threshold contour on Row 2
            if r_idx == 1:
                thresh_val = -np.log10(p_threshold)
                if np.any(slice_img >= thresh_val):
                    ax.contour(slice_img, levels=[thresh_val], colors=["cyan"], linewidths=1.5, linestyles="dashed")

            if r_idx == 0:
                ax.set_title(plane_name, fontsize=12, fontweight="bold")
            if c_col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")

            ax.axis("off")

        # Row colorbar
        cbar = fig.colorbar(im, ax=axes[r_idx, :], shrink=0.6, aspect=25, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Statistical maps visualization saved to {out_p}")
    return out_p


plot_statistical_maps_ortho = plot_statistical_maps
