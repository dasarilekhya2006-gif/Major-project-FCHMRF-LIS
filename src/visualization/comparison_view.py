"""Comparative visualization module: Baseline Method vs. Proposed Method.

Supports:
- Synchronized side-by-side tri-planar slice inspection.
- Difference discovery classification map (Consensus, Proposed-Only, Baseline-Only).
- Quantitative slice-level metrics without cosmetic distortion.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.logging import get_logger
from src.visualization.interactive_2d import extract_slice

logger = get_logger(__name__)


def build_method_comparison_figure(
    mri_volume: np.ndarray,
    baseline_significant_mask: np.ndarray,
    proposed_significant_mask: np.ndarray,
    orientation: str = "axial",
    slice_idx: Optional[int] = None,
    baseline_name: str = "Baseline (BH-FDR)",
    proposed_name: str = "Proposed (Adaptive fcHMRF-LIS)",
    mri_opacity: float = 0.95,
    overlay_opacity: float = 0.85,
) -> go.Figure:
    """Generate a synchronized 2-panel side-by-side figure comparing Baseline vs. Proposed methods.

    Both panels render the exact same anatomical slice and spatial orientation.
    """
    nx, ny, nz = mri_volume.shape
    if slice_idx is None:
        slice_idx = nz // 2 if orientation == "axial" else (ny // 2 if orientation == "coronal" else nx // 2)

    base_slice = extract_slice(mri_volume, orientation, slice_idx)
    base_sig_slice = extract_slice(baseline_significant_mask, orientation, slice_idx) > 0
    prop_sig_slice = extract_slice(proposed_significant_mask, orientation, slice_idx) > 0

    vmin_disp = float(np.percentile(base_slice, 2))
    vmax_disp = float(np.percentile(base_slice, 98))
    if vmax_disp <= vmin_disp:
        vmax_disp = vmin_disp + 1.0

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{baseline_name} (Discoveries: {int(np.sum(base_sig_slice))})",
            f"{proposed_name} (Discoveries: {int(np.sum(prop_sig_slice))})",
        ],
        horizontal_spacing=0.06,
    )

    # 1. Left Subplot: Baseline Anatomical MRI
    fig.add_trace(
        go.Heatmap(
            z=base_slice,
            colorscale="gray",
            zmin=vmin_disp,
            zmax=vmax_disp,
            showscale=False,
            opacity=mri_opacity,
            hoverinfo="none",
        ),
        row=1,
        col=1,
    )

    # Left Subplot: Baseline Significant Overlay (Amber/Orange)
    if np.any(base_sig_slice):
        fig.add_trace(
            go.Heatmap(
                z=np.where(base_sig_slice, 1.0, np.nan),
                colorscale=[[0.0, "rgba(255, 140, 0, 0.85)"], [1.0, "rgba(255, 140, 0, 0.85)"]],
                showscale=False,
                opacity=overlay_opacity,
                name=f"{baseline_name} Detections",
            ),
            row=1,
            col=1,
        )

    # 2. Right Subplot: Proposed Anatomical MRI
    fig.add_trace(
        go.Heatmap(
            z=base_slice,
            colorscale="gray",
            zmin=vmin_disp,
            zmax=vmax_disp,
            showscale=False,
            opacity=mri_opacity,
            hoverinfo="none",
        ),
        row=1,
        col=2,
    )

    # Right Subplot: Proposed Significant Overlay (Warm Amber/Yellow)
    if np.any(prop_sig_slice):
        fig.add_trace(
            go.Heatmap(
                z=np.where(prop_sig_slice, 1.0, np.nan),
                colorscale=[[0.0, "rgba(255, 215, 0, 0.9)"], [1.0, "rgba(255, 215, 0, 0.9)"]],
                showscale=False,
                opacity=overlay_opacity,
                name=f"{proposed_name} Detections",
            ),
            row=1,
            col=2,
        )

    # Layout styling
    fig.update_layout(
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(color="#e0e0e0"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)

    return fig


def build_difference_map_figure(
    mri_volume: np.ndarray,
    baseline_significant_mask: np.ndarray,
    proposed_significant_mask: np.ndarray,
    orientation: str = "axial",
    slice_idx: Optional[int] = None,
    baseline_name: str = "Baseline",
    proposed_name: str = "Proposed",
) -> go.Figure:
    """Generate a 3-category difference classification map:

    1. Consensus Discoveries (Detected by BOTH methods): Green.
    2. Proposed-Only Discoveries (Detected ONLY by Proposed): Orange/Yellow.
    3. Baseline-Only Discoveries (Detected ONLY by Baseline): Purple/Blue.
    """
    nx, ny, nz = mri_volume.shape
    if slice_idx is None:
        slice_idx = nz // 2 if orientation == "axial" else (ny // 2 if orientation == "coronal" else nx // 2)

    base_slice = extract_slice(mri_volume, orientation, slice_idx)
    b_sig = extract_slice(baseline_significant_mask, orientation, slice_idx) > 0
    p_sig = extract_slice(proposed_significant_mask, orientation, slice_idx) > 0

    consensus = b_sig & p_sig
    prop_only = p_sig & ~b_sig
    base_only = b_sig & ~p_sig

    vmin_disp = float(np.percentile(base_slice, 2))
    vmax_disp = float(np.percentile(base_slice, 98))
    if vmax_disp <= vmin_disp:
        vmax_disp = vmin_disp + 1.0

    fig = go.Figure()

    # Base MRI
    fig.add_trace(
        go.Heatmap(
            z=base_slice,
            colorscale="gray",
            zmin=vmin_disp,
            zmax=vmax_disp,
            showscale=False,
            opacity=0.9,
            hoverinfo="none",
            name="Anatomical MRI",
        )
    )

    # Category 1: Consensus (Both) -> Green
    if np.any(consensus):
        fig.add_trace(
            go.Heatmap(
                z=np.where(consensus, 1.0, np.nan),
                colorscale=[[0.0, "rgba(46, 204, 113, 0.85)"], [1.0, "rgba(46, 204, 113, 0.85)"]],
                showscale=False,
                name=f"Consensus (Both: {int(np.sum(consensus))} voxels)",
            )
        )

    # Category 2: Proposed Only -> Warm Gold
    if np.any(prop_only):
        fig.add_trace(
            go.Heatmap(
                z=np.where(prop_only, 1.0, np.nan),
                colorscale=[[0.0, "rgba(241, 196, 15, 0.85)"], [1.0, "rgba(241, 196, 15, 0.85)"]],
                showscale=False,
                name=f"{proposed_name} Only ({int(np.sum(prop_only))} voxels)",
            )
        )

    # Category 3: Baseline Only -> Violet
    if np.any(base_only):
        fig.add_trace(
            go.Heatmap(
                z=np.where(base_only, 1.0, np.nan),
                colorscale=[[0.0, "rgba(155, 89, 182, 0.85)"], [1.0, "rgba(155, 89, 182, 0.85)"]],
                showscale=False,
                name=f"{baseline_name} Only ({int(np.sum(base_only))} voxels)",
            )
        )

    total_slices = nz if orientation == "axial" else (ny if orientation == "coronal" else nx)
    fig.update_layout(
        title=dict(
            text=f"Method Discrepancy Map ({orientation.capitalize()} Slice {slice_idx + 1}/{total_slices})",
            x=0.5,
            font=dict(size=14, color="#e0e0e0"),
        ),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            font=dict(color="#cccccc", size=10),
            bgcolor="rgba(25, 25, 25, 0.85)",
            x=0.02,
            y=0.98,
        ),
    )

    return fig
