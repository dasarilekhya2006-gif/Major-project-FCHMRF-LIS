"""Visualization utilities for 3D U-Net ROI segmentation and training curves."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.logging import get_logger

logger = get_logger(__name__)


def visualize_roi_segmentation(
    original_volume: np.ndarray,
    predicted_mask: np.ndarray,
    probability_map: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """Generate a multi-panel visual comparison: Original MRI, Predicted ROI, and Composite Overlay.

    Rows:
      1. Original MRI / Statistical Map (Sagittal, Coronal, Axial)
      2. Predicted ROI Probability Heatmap (or binary mask)
      3. Composite Overlay (Original background with semi-transparent red ROI and boundary contour)

    Args:
        original_volume: 3D numpy array of image or z-scores.
        predicted_mask: 3D binary numpy array {0, 1}.
        probability_map: Optional 3D continuous probability array in [0, 1].
        slice_indices: Optional (x, y, z) slice coordinates. If None, uses center of mass of predicted ROI.
        output_path: Destination PNG path.
        title: Optional figure title.
        dpi: Image resolution.

    Returns:
        Path: Path to saved visualization figure.
    """
    orig_3d = original_volume[..., 0] if original_volume.ndim == 4 else original_volume
    mask_3d = predicted_mask[..., 0] if predicted_mask.ndim == 4 else predicted_mask
    prob_3d = (probability_map[..., 0] if probability_map.ndim == 4 else probability_map) if probability_map is not None else mask_3d

    # Determine slice indices
    if slice_indices is None:
        if np.any(mask_3d > 0.5):
            # Centroid of active ROI
            coords = np.argwhere(mask_3d > 0.5)
            x_sl, y_sl, z_sl = coords.mean(axis=0).astype(int)
        else:
            # Fallback to volume geometric center
            x_sl, y_sl, z_sl = [s // 2 for s in orig_3d.shape]
    else:
        x_sl, y_sl, z_sl = slice_indices

    # Clamp slice indices
    x_sl = max(0, min(x_sl, orig_3d.shape[0] - 1))
    y_sl = max(0, min(y_sl, orig_3d.shape[1] - 1))
    z_sl = max(0, min(z_sl, orig_3d.shape[2] - 1))

    out_p = Path(output_path or "outputs/figures/roi_segmentation_overview.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 11), dpi=dpi)
    fig.patch.set_facecolor("#0b0f19")

    fig_title = title or "3D U-Net Region of Interest (ROI) Localization Overview"
    fig.suptitle(fig_title, color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)

    # --- Slices Definition ---
    # Sagittal (slice along X), Coronal (along Y), Axial (along Z)
    slices_orig = [np.rot90(orig_3d[x_sl, :, :]), np.rot90(orig_3d[:, y_sl, :]), np.rot90(orig_3d[:, :, z_sl])]
    slices_prob = [np.rot90(prob_3d[x_sl, :, :]), np.rot90(prob_3d[:, y_sl, :]), np.rot90(prob_3d[:, :, z_sl])]
    slices_mask = [np.rot90(mask_3d[x_sl, :, :]), np.rot90(mask_3d[:, y_sl, :]), np.rot90(mask_3d[:, :, z_sl])]

    view_names = [f"Sagittal (X={x_sl})", f"Coronal (Y={y_sl})", f"Axial (Z={z_sl})"]

    for col in range(3):
        # Row 1: Original MRI
        ax1 = plt.subplot2grid((3, 3), (0, col))
        im1 = ax1.imshow(slices_orig[col], cmap="gray")
        ax1.set_title(f"Original: {view_names[col]}", color="#94a3b8", fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Row 2: Predicted ROI Probability Heatmap
        ax2 = plt.subplot2grid((3, 3), (1, col))
        im2 = ax2.imshow(slices_prob[col], cmap="magma", vmin=0.0, vmax=1.0)
        ax2.set_title(f"Predicted ROI Prior: {view_names[col]}", color="#f43f5e", fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Row 3: Composite Overlay
        ax3 = plt.subplot2grid((3, 3), (2, col))
        ax3.imshow(slices_orig[col], cmap="gray")
        # Overlay mask with alpha
        overlay = np.zeros((*slices_mask[col].shape, 4), dtype=np.float32)
        active = slices_mask[col] > 0.5
        overlay[active] = [0.93, 0.27, 0.37, 0.45]  # Red translucent
        ax3.imshow(overlay)
        if np.any(active):
            ax3.contour(slices_mask[col], levels=[0.5], colors="#ef4444", linewidths=1.5)
        ax3.set_title(f"Overlay: {view_names[col]}", color="#38bdf8", fontsize=10)

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#1e293b")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_p, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated ROI segmentation visualization at: {out_p}")
    return out_p


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "3D U-Net Training Dynamics",
    dpi: int = 300,
) -> Path:
    """Generate dual-panel training dynamics curves: Loss progression and Validation metrics.

    Args:
        history: Dictionary containing 'train_loss', and optionally 'val_loss', 'val_dice', 'val_iou', etc.
        output_path: Destination PNG file path.
        title: Main figure title.
        dpi: Image resolution.

    Returns:
        Path: Path to saved figure.
    """
    out_p = Path(output_path or "outputs/figures/unet_training_curves.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)
    fig.patch.set_facecolor("#0b0f19")

    fig.suptitle(title, color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Loss
    ax_loss.set_facecolor("#111827")
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="#38bdf8", linewidth=2.0)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        ax_loss.plot(epochs, history["val_loss"], label="Val Loss", color="#f43f5e", linewidth=2.0, linestyle="--")
    ax_loss.set_title("Optimization Loss", color="#f8fafc", fontsize=11)
    ax_loss.set_xlabel("Epoch", color="#94a3b8")
    ax_loss.set_ylabel("Loss", color="#94a3b8")
    ax_loss.tick_params(colors="#94a3b8")
    ax_loss.grid(True, linestyle=":", alpha=0.3, color="#475569")
    ax_loss.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax_loss.spines.values():
        spine.set_color("#334155")

    # Panel 2: Segmentation Metrics (Dice, IoU, Precision, Recall)
    ax_metrics.set_facecolor("#111827")
    has_metrics = False
    if "val_dice" in history and len(history["val_dice"]) > 0:
        ax_metrics.plot(epochs, history["val_dice"], label="Dice Score", color="#10b981", linewidth=2.2)
        has_metrics = True
    if "val_iou" in history and len(history["val_iou"]) > 0:
        ax_metrics.plot(epochs, history["val_iou"], label="IoU (Jaccard)", color="#06b6d4", linewidth=1.8, linestyle="-.")
        has_metrics = True
    if "val_precision" in history and len(history["val_precision"]) > 0:
        ax_metrics.plot(epochs, history["val_precision"], label="Precision", color="#fbbf24", linewidth=1.5, linestyle=":")
        has_metrics = True
    if "val_recall" in history and len(history["val_recall"]) > 0:
        ax_metrics.plot(epochs, history["val_recall"], label="Recall (TPR)", color="#a855f7", linewidth=1.5, linestyle=":")
        has_metrics = True

    if has_metrics:
        ax_metrics.set_ylim(-0.05, 1.05)
        ax_metrics.set_title("Validation Segmentation Metrics", color="#f8fafc", fontsize=11)
        ax_metrics.set_xlabel("Epoch", color="#94a3b8")
        ax_metrics.set_ylabel("Score [0, 1]", color="#94a3b8")
        ax_metrics.tick_params(colors="#94a3b8")
        ax_metrics.grid(True, linestyle=":", alpha=0.3, color="#475569")
        ax_metrics.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")
    else:
        ax_metrics.text(
            0.5, 0.5, "No validation metrics logged", color="#64748b", ha="center", va="center"
        )

    for spine in ax_metrics.spines.values():
        spine.set_color("#334155")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_p, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved training curves to {out_p}")
    return out_p
