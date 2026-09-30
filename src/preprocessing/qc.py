"""Visual quality-control (QC) generator comparing neuroimaging volumes before and after preprocessing."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.logging import get_logger
from src.data.schemas import NeuroimagingVolume
from src.preprocessing.pipeline import PreprocessedVolume

logger = get_logger(__name__)


def generate_preprocessing_qc_plot(
    raw_volume: NeuroimagingVolume,
    preprocessed_volume: PreprocessedVolume,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Generate a comprehensive multi-panel visual QC report comparing volume before and after preprocessing.

    Panels included:
      - Orthogonal slices (Axial, Coronal, Sagittal) for Raw Volume.
      - Orthogonal slices for Preprocessed Volume with brain mask boundary contours.
      - Voxel intensity distribution histograms before vs. after normalization.
      - Tabular summary card displaying spatial metadata, affine shifts, and execution parameters.

    Args:
        raw_volume: Input volume prior to preprocessing.
        preprocessed_volume: Preprocessed volume with attached audit metadata.
        output_path: Destination path (.png). If None, defaults to outputs/qc/<subject_id>_qc.png.
        dpi: Resolution of exported QC image.

    Returns:
        Path: Path to the generated visual report.
    """
    if output_path is None:
        sub_id = preprocessed_volume.subject_id or "volume"
        out_p = Path(f"outputs/qc/{sub_id}_preprocessing_qc.png")
    else:
        out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Extract 3D spatial arrays (first timepoint if 4D)
    raw_data = raw_volume.data[..., 0] if raw_volume.is_4d else raw_volume.data
    proc_data = preprocessed_volume.data[..., 0] if preprocessed_volume.is_4d else preprocessed_volume.data
    mask = preprocessed_volume.mask

    # Find central slice indices
    raw_shape = raw_data.shape
    proc_shape = proc_data.shape

    raw_slices = (raw_shape[0] // 2, raw_shape[1] // 2, raw_shape[2] // 2)
    proc_slices = (proc_shape[0] // 2, proc_shape[1] // 2, proc_shape[2] // 2)

    # Configure 3x3 grid figure
    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")

    title_text = f"Quality Control Report: Preprocessing Pipeline\nSubject: {preprocessed_volume.subject_id}"
    fig.suptitle(title_text, color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)

    # Colormap
    cmap = "viridis"

    # --- Row 1: Raw Volume Slices ---
    ax_raw_x = plt.subplot2grid((3, 3), (0, 0))
    ax_raw_y = plt.subplot2grid((3, 3), (0, 1))
    ax_raw_z = plt.subplot2grid((3, 3), (0, 2))

    im1 = ax_raw_x.imshow(np.rot90(raw_data[raw_slices[0], :, :]), cmap=cmap)
    ax_raw_x.set_title(f"Raw Sagittal (X={raw_slices[0]})", color="#94a3b8", fontsize=10)
    plt.colorbar(im1, ax=ax_raw_x, fraction=0.046, pad=0.04)

    im2 = ax_raw_y.imshow(np.rot90(raw_data[:, raw_slices[1], :]), cmap=cmap)
    ax_raw_y.set_title(f"Raw Coronal (Y={raw_slices[1]})", color="#94a3b8", fontsize=10)
    plt.colorbar(im2, ax=ax_raw_y, fraction=0.046, pad=0.04)

    im3 = ax_raw_z.imshow(np.rot90(raw_data[:, :, raw_slices[2]]), cmap=cmap)
    ax_raw_z.set_title(f"Raw Axial (Z={raw_slices[2]})", color="#94a3b8", fontsize=10)
    plt.colorbar(im3, ax=ax_raw_z, fraction=0.046, pad=0.04)

    # --- Row 2: Preprocessed Volume Slices + Mask Contours ---
    ax_pr_x = plt.subplot2grid((3, 3), (1, 0))
    ax_pr_y = plt.subplot2grid((3, 3), (1, 1))
    ax_pr_z = plt.subplot2grid((3, 3), (1, 2))

    im4 = ax_pr_x.imshow(np.rot90(proc_data[proc_slices[0], :, :]), cmap=cmap)
    if mask is not None:
        ax_pr_x.contour(np.rot90(mask[proc_slices[0], :, :]), levels=[0.5], colors="#ef4444", linewidths=1.2)
    ax_pr_x.set_title(f"Preproc Sagittal (X={proc_slices[0]})", color="#38bdf8", fontsize=10)
    plt.colorbar(im4, ax=ax_pr_x, fraction=0.046, pad=0.04)

    im5 = ax_pr_y.imshow(np.rot90(proc_data[:, proc_slices[1], :]), cmap=cmap)
    if mask is not None:
        ax_pr_y.contour(np.rot90(mask[:, proc_slices[1], :]), levels=[0.5], colors="#ef4444", linewidths=1.2)
    ax_pr_y.set_title(f"Preproc Coronal (Y={proc_slices[1]})", color="#38bdf8", fontsize=10)
    plt.colorbar(im5, ax=ax_pr_y, fraction=0.046, pad=0.04)

    im6 = ax_pr_z.imshow(np.rot90(proc_data[:, :, proc_slices[2]]), cmap=cmap)
    if mask is not None:
        ax_pr_z.contour(np.rot90(mask[:, :, proc_slices[2]]), levels=[0.5], colors="#ef4444", linewidths=1.2)
    ax_pr_z.set_title(f"Preproc Axial (Z={proc_slices[2]})", color="#38bdf8", fontsize=10)
    plt.colorbar(im6, ax=ax_pr_z, fraction=0.046, pad=0.04)

    for ax in [ax_raw_x, ax_raw_y, ax_raw_z, ax_pr_x, ax_pr_y, ax_pr_z]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # --- Row 3: Histograms and Metadata Summary ---
    ax_hist_raw = plt.subplot2grid((3, 3), (2, 0))
    ax_hist_pr = plt.subplot2grid((3, 3), (2, 1))
    ax_card = plt.subplot2grid((3, 3), (2, 2))

    # Raw Histogram
    fin_raw = raw_data[np.isfinite(raw_data)]
    ax_hist_raw.hist(fin_raw.ravel(), bins=50, color="#94a3b8", alpha=0.75, edgecolor="#475569")
    ax_hist_raw.set_title("Raw Intensity Distribution", color="#f8fafc", fontsize=10)
    ax_hist_raw.set_facecolor("#1e293b")
    ax_hist_raw.tick_params(colors="#94a3b8")
    for spine in ax_hist_raw.spines.values():
        spine.set_color("#334155")

    # Preprocessed Histogram (in-mask)
    fin_pr = proc_data[mask] if mask is not None else proc_data[np.isfinite(proc_data)]
    ax_hist_pr.hist(fin_pr.ravel(), bins=50, color="#38bdf8", alpha=0.75, edgecolor="#0284c7")
    ax_hist_pr.set_title("Preprocessed In-Mask Intensities", color="#f8fafc", fontsize=10)
    ax_hist_pr.set_facecolor("#1e293b")
    ax_hist_pr.tick_params(colors="#94a3b8")
    for spine in ax_hist_pr.spines.values():
        spine.set_color("#334155")

    # Metadata Card
    ax_card.set_facecolor("#1e293b")
    ax_card.set_xticks([])
    ax_card.set_yticks([])
    for spine in ax_card.spines.values():
        spine.set_color("#334155")

    meta = preprocessed_volume.preprocessing_metadata
    origin_shift = np.linalg.norm(meta.preprocessed_affine[:3, 3] - meta.original_affine[:3, 3])

    card_text = (
        f"PREPROCESSING AUDIT TRAIL\n"
        f"-----------------------------------------\n"
        f"Raw Dimensions:        {meta.original_shape}\n"
        f"Target Dimensions:     {meta.preprocessed_shape}\n"
        f"Original Voxel Size:   {meta.original_voxel_dimensions[:3]} mm\n"
        f"Preproc Voxel Size:    {meta.preprocessed_voxel_dimensions[:3]} mm\n"
        f"Affine Origin Shift:   {origin_shift:.3f} mm\n"
        f"Normalization Center:  {meta.normalization_center:.4f}\n"
        f"Normalization Scale:   {meta.normalization_scale:.4f}\n"
        f"In-Mask Brain Voxels:  {preprocessed_volume.num_brain_voxels:,}\n"
        f"Executed Pipeline Steps:\n"
    )
    for s in meta.history:
        card_text += f"  • {s.step_name}\n"

    ax_card.text(
        0.05,
        0.95,
        card_text,
        color="#f1f5f9",
        fontsize=8.5,
        family="monospace",
        verticalalignment="top",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_p, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated preprocessing QC plot at: {out_p}")
    return out_p


plot_preprocessing_qc = generate_preprocessing_qc_plot
