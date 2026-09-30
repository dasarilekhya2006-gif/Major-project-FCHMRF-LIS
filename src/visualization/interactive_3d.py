"""Interactive 3D Brain and Abnormality Visualization using Marching Cubes and Plotly 3D.

Renders real 3D anatomical brain structures and semi-transparent orange/yellow abnormality heatmaps.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage
from skimage import measure

from src.core.logging import get_logger

logger = get_logger(__name__)


def extract_surface_mesh(
    volume: np.ndarray,
    level: float = 0.5,
    step_size: int = 1,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract 3D triangulated isosurface using Marching Cubes algorithm.

    Returns:
        (x, y, z, i, j, k) where (x, y, z) are vertex coordinates and (i, j, k) are triangle indices.
        Returns None if volume has no values crossing the specified level.
    """
    try:
        # Add a zero padding border to ensure closed isosurfaces at volume edges
        padded = np.pad(volume, 1, mode="constant", constant_values=0)
        verts, faces, normals, values = measure.marching_cubes(
            padded,
            level=level,
            step_size=step_size,
            spacing=spacing,
        )
        # Shift back coordinates by 1 due to padding
        verts = verts - np.array(spacing, dtype=np.float64)

        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        return x, y, z, i, j, k
    except Exception as e:
        logger.debug(f"Marching cubes surface extraction skipped: {e}")
        return None


def build_interactive_3d_brain_figure(
    mri_volume: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    significant_mask: Optional[np.ndarray] = None,
    ground_truth_mask: Optional[np.ndarray] = None,
    statistical_map: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    # Visibility toggles
    show_brain_shell: bool = True,
    show_roi_mesh: bool = True,
    show_significant_mesh: bool = True,
    show_ground_truth_mesh: bool = False,
    # Opacities
    brain_shell_opacity: float = 0.12,
    roi_mesh_opacity: float = 0.35,
    significant_mesh_opacity: float = 0.85,
    gt_mesh_opacity: float = 0.55,
    # Voxel downsampling factor for real-time 3D rendering responsiveness
    step_size: int = 1,
    figure_title: str = "Interactive 3D Anatomical Brain & Abnormality Visualization",
) -> go.Figure:
    """Construct an interactive 3D Plotly figure.

    Visual Layers:
      1. Brain Anatomy: Semi-transparent neutral gray outer shell revealing internal structure.
      2. 3D U-Net ROI: Translucent cyan/teal 3D surface mesh.
      3. Statistical Abnormalities: Rich semi-transparent orange/yellow 3D surface mesh.
      4. Ground Truth: Semi-transparent magenta 3D surface mesh.

    Features:
      - 360-degree free rotation, zoom, pan.
      - Preserves anatomical spatial coordinates.
    """
    if affine is None:
        voxel_spacing = (1.0, 1.0, 1.0)
    else:
        # Compute voxel spacing from affine norm of column vectors
        voxel_spacing = (
            float(np.linalg.norm(affine[:3, 0])),
            float(np.linalg.norm(affine[:3, 1])),
            float(np.linalg.norm(affine[:3, 2])),
        )

    fig = go.Figure()

    # 1. Brain Anatomy Shell (Semi-transparent Neutral Gray)
    if show_brain_shell:
        if brain_mask is not None and np.any(brain_mask > 0):
            shell_vol = (brain_mask > 0).astype(np.float32)
        else:
            # Approximate anatomical brain boundary using robust intensity thresholding
            thresh = float(np.percentile(mri_volume[mri_volume > 0], 25)) if np.any(mri_volume > 0) else 0.5
            shell_vol = ndimage.binary_fill_holes(mri_volume > thresh).astype(np.float32)

        # Smooth slightly to create smooth anatomical cortical surface
        smooth_shell = ndimage.gaussian_filter(shell_vol, sigma=0.8)
        mesh_shell = extract_surface_mesh(smooth_shell, level=0.4, step_size=max(1, step_size), spacing=voxel_spacing)

        if mesh_shell is not None:
            sx, sy, sz, si, sj, sk = mesh_shell
            fig.add_trace(
                go.Mesh3d(
                    x=sx,
                    y=sy,
                    z=sz,
                    i=si,
                    j=sj,
                    k=sk,
                    color="#aaaaaa",
                    opacity=brain_shell_opacity,
                    name="Brain Anatomical Shell",
                    hoverinfo="name",
                    flatshading=False,
                    lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.5, specular=0.2),
                )
            )

    # 2. 3D U-Net ROI Mesh (Semi-transparent Cyan)
    if show_roi_mesh and roi_mask is not None and np.any(roi_mask > 0):
        roi_vol = ndimage.gaussian_filter((roi_mask > 0).astype(np.float32), sigma=0.6)
        mesh_roi = extract_surface_mesh(roi_vol, level=0.35, step_size=max(1, step_size), spacing=voxel_spacing)

        if mesh_roi is not None:
            rx, ry, rz, ri, rj, rk = mesh_roi
            fig.add_trace(
                go.Mesh3d(
                    x=rx,
                    y=ry,
                    z=rz,
                    i=ri,
                    j=rj,
                    k=rk,
                    color="#00e5ff",  # Vibrant cyan
                    opacity=roi_mesh_opacity,
                    name="3D U-Net ROI",
                    hoverinfo="name",
                    flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.7, roughness=0.3, specular=0.4),
                )
            )

    # 3. Ground Truth Surface Mesh (Magenta)
    if show_ground_truth_mesh and ground_truth_mask is not None and np.any(ground_truth_mask > 0):
        gt_vol = ndimage.gaussian_filter((ground_truth_mask > 0).astype(np.float32), sigma=0.6)
        mesh_gt = extract_surface_mesh(gt_vol, level=0.35, step_size=max(1, step_size), spacing=voxel_spacing)

        if mesh_gt is not None:
            gx, gy, gz, gi, gj, gk = mesh_gt
            fig.add_trace(
                go.Mesh3d(
                    x=gx,
                    y=gy,
                    z=gz,
                    i=gi,
                    j=gj,
                    k=gk,
                    color="#e040fb",  # Magenta
                    opacity=gt_mesh_opacity,
                    name="Ground Truth ROI",
                    hoverinfo="name",
                    flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.6, roughness=0.4, specular=0.3),
                )
            )

    # 4. Significant Abnormalities Mesh (Semi-transparent Temperature Orange-to-Yellow)
    if show_significant_mesh and significant_mask is not None and np.any(significant_mask > 0):
        sig_vol = ndimage.gaussian_filter((significant_mask > 0).astype(np.float32), sigma=0.5)
        mesh_sig = extract_surface_mesh(sig_vol, level=0.35, step_size=max(1, step_size), spacing=voxel_spacing)

        if mesh_sig is not None:
            ax, ay, az, ai, aj, ak = mesh_sig
            fig.add_trace(
                go.Mesh3d(
                    x=ax,
                    y=ay,
                    z=az,
                    i=ai,
                    j=aj,
                    k=ak,
                    color="#ff9100",  # Intense warm amber/orange
                    opacity=significant_mesh_opacity,
                    name="FDR-Significant Region",
                    hoverinfo="name",
                    flatshading=False,
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.2, specular=0.6),
                )
            )

    # Configure 3D Scene Camera and Dark Theme
    fig.update_layout(
        title=dict(text=figure_title, x=0.5, font=dict(size=14, color="#e0e0e0")),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, title="X (mm)", color="#888888"),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, title="Y (mm)", color="#888888"),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, title="Z (mm)", color="#888888"),
            bgcolor="#121212",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.3),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#121212",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            font=dict(color="#cccccc", size=10),
            bgcolor="rgba(25, 25, 25, 0.8)",
            x=0.02,
            y=0.98,
        ),
    )

    return fig
