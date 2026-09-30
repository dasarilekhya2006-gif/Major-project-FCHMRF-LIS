"""Interactive 2D Tri-Planar Slice Viewer with Grayscale Anatomical MRI and Temperature Heatmaps.

Preserves raw scientific values without artificial exaggeration or cosmetic manipulation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

from src.core.logging import get_logger

logger = get_logger(__name__)


def voxel_to_world_coordinates(
    voxel_coords: Tuple[int, int, int],
    affine: np.ndarray,
) -> Tuple[float, float, float]:
    """Convert (x, y, z) voxel indices to anatomical world coordinates (X, Y, Z) in mm."""
    v = np.array([voxel_coords[0], voxel_coords[1], voxel_coords[2], 1.0], dtype=np.float64)
    world = affine @ v
    return float(world[0]), float(world[1]), float(world[2])


def extract_slice(
    volume: np.ndarray,
    orientation: str,
    slice_idx: int,
) -> np.ndarray:
    """Extract a 2D slice from a 3D volume according to anatomical orientation.

    Args:
        volume: 3D numpy array (X, Y, Z).
        orientation: 'axial' (Z-plane), 'coronal' (Y-plane), or 'sagittal' (X-plane).
        slice_idx: 0-indexed slice index along the corresponding dimension.

    Returns:
        2D numpy array oriented for radiological display (transposed/flipped appropriately).
    """
    orientation = orientation.lower()
    nx, ny, nz = volume.shape

    if orientation == "axial":
        idx = max(0, min(slice_idx, nz - 1))
        # Axial: X is Left-Right, Y is Anterior-Posterior. Display with Anterior at top.
        slice_2d = np.rot90(volume[:, :, idx])
    elif orientation == "coronal":
        idx = max(0, min(slice_idx, ny - 1))
        # Coronal: X is Left-Right, Z is Superior-Inferior. Display with Superior at top.
        slice_2d = np.rot90(volume[:, idx, :])
    elif orientation == "sagittal":
        idx = max(0, min(slice_idx, nx - 1))
        # Sagittal: Y is Anterior-Posterior, Z is Superior-Inferior.
        slice_2d = np.rot90(volume[idx, :, :])
    else:
        raise ValueError(f"Unknown orientation '{orientation}'. Choose 'axial', 'coronal', or 'sagittal'.")

    return slice_2d


def build_temperature_colormap() -> List[List[Union[float, str]]]:
    """Intuitive scientific temperature-style colormap for abnormality heatmaps:

    Dark/Low Values -> Rich Orange -> Bright Yellow/High Values.
    """
    return [
        [0.0, "rgb(180, 50, 0)"],    # Dark reddish-orange
        [0.35, "rgb(230, 95, 0)"],   # Deep orange
        [0.70, "rgb(255, 165, 0)"],  # Warm orange
        [0.90, "rgb(255, 215, 0)"],  # Bright gold-yellow
        [1.0, "rgb(255, 255, 120)"], # Intense light yellow
    ]


def build_interactive_slice_figure(
    mri_volume: np.ndarray,
    orientation: str = "axial",
    slice_idx: Optional[int] = None,
    preprocessed_volume: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    statistical_map: Optional[np.ndarray] = None,
    lis_map: Optional[np.ndarray] = None,
    fdr_significant_mask: Optional[np.ndarray] = None,
    ground_truth_mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    # Visibility toggles
    show_original_mri: bool = True,
    show_preprocessed: bool = False,
    show_roi: bool = True,
    show_statistical_map: bool = False,
    show_lis_map: bool = False,
    show_fdr_significant: bool = True,
    show_ground_truth: bool = False,
    # Thresholds and opacities
    stat_threshold: float = 2.0,
    lis_threshold: float = 0.5,
    mri_opacity: float = 1.0,
    heatmap_opacity: float = 0.85,
    roi_opacity: float = 0.40,
    gt_opacity: float = 0.60,
    heatmap_colorscale: str = "temperature",
    figure_title: Optional[str] = None,
) -> go.Figure:
    """Construct an interactive Plotly figure composing anatomical grayscale MRI with scientific overlays.

    Anatomical Layer:
      - Grayscale MRI (Original or Preprocessed).

    Abnormality Heatmap Layer:
      - Statistical or LIS values filtered strictly by significance/threshold.
      - Rendered in a temperature-style palette (Dark Orange -> Orange -> Bright Yellow).
      - Non-significant voxels remain completely transparent to maintain anatomical brain visibility.

    Structural Overlays:
      - 3D U-Net ROI: Semi-transparent cyan boundary/mask.
      - Ground Truth: Outlined/semi-transparent magenta overlay.
    """
    if affine is None:
        affine = np.eye(4, dtype=np.float64)

    nx, ny, nz = mri_volume.shape
    if slice_idx is None:
        if orientation.lower() == "axial":
            slice_idx = nz // 2
        elif orientation.lower() == "coronal":
            slice_idx = ny // 2
        else:
            slice_idx = nx // 2

    # 1. Base Anatomical Image
    if show_preprocessed and preprocessed_volume is not None:
        base_slice = extract_slice(preprocessed_volume, orientation, slice_idx)
        base_name = "Preprocessed MRI"
    else:
        base_slice = extract_slice(mri_volume, orientation, slice_idx)
        base_name = "Original MRI"

    h, w = base_slice.shape
    fig = go.Figure()

    # 1A. Add Base Grayscale Anatomical MRI
    if show_original_mri or show_preprocessed:
        # Determine robust grayscale display window (percentile-based without mutating raw values)
        vmin_disp = float(np.percentile(base_slice, 2))
        vmax_disp = float(np.percentile(base_slice, 98))
        if vmax_disp <= vmin_disp:
            vmax_disp = vmin_disp + 1.0

        fig.add_trace(
            go.Heatmap(
                z=base_slice,
                colorscale="gray",
                zmin=vmin_disp,
                zmax=vmax_disp,
                showscale=False,
                opacity=mri_opacity,
                hoverinfo="none",
                name=base_name,
            )
        )

    # 2. Add 3D U-Net ROI Mask Layer (Translucent Cyan)
    if show_roi and roi_mask is not None:
        roi_slice = extract_slice(roi_mask, orientation, slice_idx) > 0
        if np.any(roi_slice):
            # Create masked RGBA overlay: Cyan (0, 220, 220)
            roi_disp = np.where(roi_slice, 1.0, np.nan)
            cyan_colorscale = [[0.0, "rgba(0, 230, 230, 0.8)"], [1.0, "rgba(0, 230, 230, 0.8)"]]
            fig.add_trace(
                go.Heatmap(
                    z=roi_disp,
                    colorscale=cyan_colorscale,
                    showscale=False,
                    opacity=roi_opacity,
                    hoverinfo="none",
                    name="3D U-Net ROI",
                )
            )

    # 3. Add Ground Truth Layer (Magenta Contour/Fill)
    if show_ground_truth and ground_truth_mask is not None:
        gt_slice = extract_slice(ground_truth_mask, orientation, slice_idx) > 0
        if np.any(gt_slice):
            gt_disp = np.where(gt_slice, 1.0, np.nan)
            magenta_colorscale = [[0.0, "rgba(230, 0, 230, 0.8)"], [1.0, "rgba(230, 0, 230, 0.8)"]]
            fig.add_trace(
                go.Heatmap(
                    z=gt_disp,
                    colorscale=magenta_colorscale,
                    showscale=False,
                    opacity=gt_opacity,
                    hoverinfo="none",
                    name="Ground Truth",
                )
            )

    # 4. Abnormality Heatmap Layer (Temperature Colormap: Dark Orange -> Orange -> Bright Yellow)
    # Priority: FDR Significant Masked Map > LIS Map > Raw Statistical Map
    heatmap_data = None
    heatmap_label = ""
    cmin, cmax = 0.0, 1.0
    active_cscale: Any = build_temperature_colormap() if heatmap_colorscale == "temperature" else heatmap_colorscale

    if show_fdr_significant and fdr_significant_mask is not None:
        fdr_slice = extract_slice(fdr_significant_mask, orientation, slice_idx) > 0
        if statistical_map is not None:
            raw_stat_slice = extract_slice(statistical_map, orientation, slice_idx)
            # Only display statistical values on significant voxels
            heatmap_data = np.where(fdr_slice, raw_stat_slice, np.nan)
            cmin = float(max(stat_threshold, np.nanmin(heatmap_data) if np.any(fdr_slice) else 0.0))
            cmax = float(np.nanmax(heatmap_data) if np.any(fdr_slice) else cmin + 1.0)
            heatmap_label = "FDR-Significant Statistic (|z|)"
        else:
            heatmap_data = np.where(fdr_slice, 1.0, np.nan)
            heatmap_label = "FDR-Significant Region"

    elif show_lis_map and lis_map is not None:
        raw_lis_slice = extract_slice(lis_map, orientation, slice_idx)
        # LIS: low values = high statistical significance. Evidence is often plotted as 1 - LIS
        evidence_slice = 1.0 - raw_lis_slice
        cutoff_evidence = 1.0 - lis_threshold
        heatmap_data = np.where(raw_lis_slice <= lis_threshold, evidence_slice, np.nan)
        cmin = float(cutoff_evidence)
        cmax = float(np.nanmax(heatmap_data) if np.nanmax(heatmap_data) > cmin else 1.0)
        heatmap_label = "Local Significance (1 - LIS)"

    elif show_statistical_map and statistical_map is not None:
        raw_stat_slice = extract_slice(statistical_map, orientation, slice_idx)
        heatmap_data = np.where(np.abs(raw_stat_slice) >= stat_threshold, np.abs(raw_stat_slice), np.nan)
        cmin = float(stat_threshold)
        cmax = float(np.nanmax(heatmap_data) if np.nanmax(heatmap_data) > cmin else cmin + 1.0)
        heatmap_label = f"Statistic (|z| >= {stat_threshold:.1f})"

    if heatmap_data is not None and np.any(~np.isnan(heatmap_data)):
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                colorscale=active_cscale,
                zmin=cmin,
                zmax=cmax,
                opacity=heatmap_opacity,
                colorbar=dict(
                    title=dict(text=heatmap_label, side="right", font=dict(size=11)),
                    thickness=14,
                    len=0.75,
                ),
                name=heatmap_label,
            )
        )

    # 5. Interactive Hover Readout (Preserving Raw Values)
    # Build informative hover text array
    hover_texts = []
    stat_slice = extract_slice(statistical_map, orientation, slice_idx) if statistical_map is not None else None
    lis_s = extract_slice(lis_map, orientation, slice_idx) if lis_map is not None else None
    fdr_s = extract_slice(fdr_significant_mask, orientation, slice_idx) if fdr_significant_mask is not None else None
    roi_s = extract_slice(roi_mask, orientation, slice_idx) if roi_mask is not None else None
    gt_s = extract_slice(ground_truth_mask, orientation, slice_idx) if ground_truth_mask is not None else None

    for r in range(h):
        row_txt = []
        for c in range(w):
            raw_val = base_slice[r, c]
            stat_val = f"{stat_slice[r, c]:.2f}" if stat_slice is not None else "N/A"
            lis_val = f"{lis_s[r, c]:.4f}" if lis_s is not None else "N/A"
            sig_txt = "Yes" if (fdr_s is not None and fdr_s[r, c] > 0) else "No"
            roi_txt = "Inside ROI" if (roi_s is not None and roi_s[r, c] > 0) else "Outside"
            gt_txt = "Active" if (gt_s is not None and gt_s[r, c] > 0) else "Null"

            txt = (
                f"<b>Pixel</b>: ({r}, {c})<br>"
                f"<b>MRI Intensity</b>: {raw_val:.2f}<br>"
                f"<b>Statistic (z)</b>: {stat_val}<br>"
                f"<b>LIS Value</b>: {lis_val}<br>"
                f"<b>FDR Significant</b>: {sig_txt}<br>"
                f"<b>3D U-Net ROI</b>: {roi_txt}"
            )
            if ground_truth_mask is not None:
                txt += f"<br><b>Ground Truth</b>: {gt_txt}"
            row_txt.append(txt)
        hover_texts.append(row_txt)

    # Invisible overlay for hover information across entire plane
    fig.add_trace(
        go.Heatmap(
            z=np.zeros_like(base_slice),
            text=hover_texts,
            hoverinfo="text",
            showscale=False,
            opacity=0.001,
            name="Voxel Inspector",
        )
    )

    # 6. Layout Configuration
    total_slices = nz if orientation == "axial" else (ny if orientation == "coronal" else nx)
    title = figure_title or f"{orientation.capitalize()} View — Slice {slice_idx + 1} / {total_slices}"

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color="#e0e0e0")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",
        hoverlabel=dict(bgcolor="rgba(30, 30, 30, 0.95)", font_size=11, font_color="#ffffff"),
    )

    return fig
