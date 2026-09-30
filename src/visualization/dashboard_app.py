"""Interactive Neuroimaging Research Visualization & Analysis Dashboard.

Built with Streamlit and Plotly to support research-grade exploration of MRI/fMRI volumes,
preprocessing outputs, 3D U-Net ROIs, statistical maps, LIS maps, FDR discoveries,
and method comparisons while preserving raw scientific values without cosmetic distortion.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
import streamlit as st

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.logging import setup_logging, get_logger
from src.visualization.interactive_2d import (
    build_interactive_slice_figure,
    voxel_to_world_coordinates,
)
from src.visualization.interactive_3d import build_interactive_3d_brain_figure
from src.visualization.comparison_view import (
    build_method_comparison_figure,
    build_difference_map_figure,
)
from src.visualization.pipeline_bridge import (
    VisualizationPipelineSession,
    load_research_benchmark_session,
    process_uploaded_nifti_bytes,
)

logger = get_logger("dashboard_app")


def configure_page() -> None:
    """Set up wide research layout and dark scientific theme."""
    st.set_page_config(
        page_title="Adaptive fcHMRF-LIS Neuroimaging Research Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 20px;
            font-weight: 700;
            color: #eceff1;
            margin-bottom: 2px;
        }
        .sub-header {
            font-size: 13px;
            color: #90a4ae;
            margin-bottom: 15px;
        }
        .metric-card {
            background-color: #1e1e1e;
            border-left: 3px solid #ff9100;
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 4px;
        }
        .metric-title { font-size: 11px; color: #b0bec5; text-transform: uppercase; }
        .metric-value { font-size: 16px; font-weight: 700; color: #ffffff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Ensure pipeline session exists in Streamlit state."""
    if "session" not in st.session_state:
        # Load synthetic research benchmark as default initial data
        st.session_state.session = load_research_benchmark_session(
            shape=(24, 24, 12),
            snr=3.2,
            nominal_alpha=0.05,
            seed=42,
        )


def main() -> None:
    setup_logging(log_level="INFO")
    configure_page()
    initialize_session_state()

    session: VisualizationPipelineSession = st.session_state.session
    nx, ny, nz = session.shape

    # ==========================================================================
    # 1. Left Panel (Sidebar): Data Ingestion & Controls
    # ==========================================================================
    with st.sidebar:
        st.markdown("<div class='main-header'>Neuroimaging FDR Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Adaptive Deep Learning fcHMRF-LIS</div>", unsafe_allow_html=True)

        # 1A. Data Ingestion
        st.subheader("1. Data Ingestion")
        uploaded_file = st.file_uploader(
            "Upload NIfTI Volume (.nii, .nii.gz)",
            type=["nii", "gz"],
            help="Upload raw anatomical MRI or fMRI volume.",
        )

        col_load_demo, col_reset = st.columns(2)
        with col_load_demo:
            if st.button("Load Benchmark", use_container_width=True, help="Load benchmark volume with known ground truth"):
                st.session_state.session = load_research_benchmark_session(
                    shape=(24, 24, 12),
                    snr=3.2,
                    nominal_alpha=0.05,
                    seed=42,
                )
                st.rerun()

        if uploaded_file is not None:
            # Check if newly uploaded
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                with st.spinner("Processing volume through research pipeline..."):
                    bytes_data = uploaded_file.read()
                    st.session_state.session = process_uploaded_nifti_bytes(
                        file_bytes=bytes_data,
                        filename=uploaded_file.name,
                        nominal_alpha=0.05,
                    )
                    st.session_state.last_uploaded_name = uploaded_file.name
                st.success(f"Loaded {uploaded_file.name}")
                st.rerun()

        st.divider()

        # 1B. Visualization Mode
        st.subheader("2. View Selection")
        view_mode = st.radio(
            "Display Mode",
            options=["2D Tri-Planar Slices", "Interactive 3D Brain", "Baseline vs Proposed Comparison"],
            index=0,
        )

        orientation = "axial"
        if view_mode in ["2D Tri-Planar Slices", "Baseline vs Proposed Comparison"]:
            orientation = st.selectbox(
                "Anatomical Plane",
                options=["axial", "coronal", "sagittal"],
                index=0,
                format_func=lambda x: f"{x.capitalize()} View",
            )

        # Max slice determination
        max_slice = nz if orientation == "axial" else (ny if orientation == "coronal" else nx)
        slice_idx = st.slider(
            f"Slice Index (1 to {max_slice})",
            min_value=1,
            max_value=max_slice,
            value=max_slice // 2,
            step=1,
        ) - 1

        st.divider()

        # 1C. Layer Visibility Toggles
        st.subheader("3. Layers")
        show_mri = st.checkbox("Original MRI", value=True)
        show_prep = st.checkbox("Preprocessed MRI", value=False)
        show_roi = st.checkbox("3D U-Net ROI", value=True)
        show_stat = st.checkbox("Statistical Map (|z|)", value=False)
        show_lis = st.checkbox("LIS Map (1 - LIS)", value=False)
        show_fdr = st.checkbox("FDR-Significant Regions", value=True)
        show_gt = st.checkbox("Ground Truth (if available)", value=(session.ground_truth_mask is not None))

        st.divider()

        # 1D. Sliders & Opacities
        st.subheader("4. Visualization Controls")
        heatmap_opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.85, 0.05)
        mri_opacity = st.slider("MRI Opacity", 0.1, 1.0, 1.0, 0.05)
        roi_opacity = st.slider("ROI Opacity", 0.1, 1.0, 0.40, 0.05)

        stat_threshold = st.slider("Statistical Threshold (|z|)", 0.0, 5.0, 2.0, 0.1)
        lis_threshold = st.slider("LIS Cutoff Threshold", 0.01, 1.0, 0.5, 0.02)
        heatmap_cscale = st.selectbox(
            "Heatmap Palette",
            options=["temperature", "hot", "inferno", "viridis", "plasma"],
            index=0,
            format_func=lambda x: "Scientific Temperature (Orange/Yellow)" if x == "temperature" else x.capitalize(),
        )

    # ==========================================================================
    # 2. Main Canvas (Center Panel) & Analysis Information (Right Panel)
    # ==========================================================================
    col_main, col_info = st.columns([7, 3])

    with col_main:
        if view_mode == "2D Tri-Planar Slices":
            fig_2d = build_interactive_slice_figure(
                mri_volume=session.mri_volume,
                orientation=orientation,
                slice_idx=slice_idx,
                preprocessed_volume=session.preprocessed_volume,
                roi_mask=session.roi_mask,
                statistical_map=session.statistical_map,
                lis_map=session.lis_map,
                fdr_significant_mask=session.proposed_significant_mask,
                ground_truth_mask=session.ground_truth_mask,
                affine=session.affine,
                show_original_mri=show_mri,
                show_preprocessed=show_prep,
                show_roi=show_roi,
                show_statistical_map=show_stat,
                show_lis_map=show_lis,
                show_fdr_significant=show_fdr,
                show_ground_truth=show_gt,
                stat_threshold=stat_threshold,
                lis_threshold=lis_threshold,
                mri_opacity=mri_opacity,
                heatmap_opacity=heatmap_opacity,
                roi_opacity=roi_opacity,
                heatmap_colorscale=heatmap_cscale,
            )
            st.plotly_chart(fig_2d, use_container_width=True, height=680)

        elif view_mode == "Interactive 3D Brain":
            with st.spinner("Extracting 3D Marching Cubes isosurfaces..."):
                fig_3d = build_interactive_3d_brain_figure(
                    mri_volume=session.mri_volume,
                    brain_mask=session.brain_mask,
                    roi_mask=session.roi_mask,
                    significant_mask=session.proposed_significant_mask,
                    ground_truth_mask=session.ground_truth_mask,
                    affine=session.affine,
                    show_brain_shell=True,
                    show_roi_mesh=show_roi,
                    show_significant_mesh=show_fdr,
                    show_ground_truth_mesh=show_gt,
                    brain_shell_opacity=0.12,
                    roi_mesh_opacity=roi_opacity,
                    significant_mesh_opacity=heatmap_opacity,
                    step_size=1,
                )
            st.plotly_chart(fig_3d, use_container_width=True, height=680)

        elif view_mode == "Baseline vs Proposed Comparison":
            comp_type = st.radio(
                "Comparison Type",
                options=["Synchronized Side-by-Side", "Difference Discrepancy Map"],
                horizontal=True,
            )
            if comp_type == "Synchronized Side-by-Side":
                fig_comp = build_method_comparison_figure(
                    mri_volume=session.mri_volume,
                    baseline_significant_mask=session.baseline_significant_mask or np.zeros_like(session.mri_volume, bool),
                    proposed_significant_mask=session.proposed_significant_mask or np.zeros_like(session.mri_volume, bool),
                    orientation=orientation,
                    slice_idx=slice_idx,
                    mri_opacity=mri_opacity,
                    overlay_opacity=heatmap_opacity,
                )
                st.plotly_chart(fig_comp, use_container_width=True, height=650)
            else:
                fig_diff = build_difference_map_figure(
                    mri_volume=session.mri_volume,
                    baseline_significant_mask=session.baseline_significant_mask or np.zeros_like(session.mri_volume, bool),
                    proposed_significant_mask=session.proposed_significant_mask or np.zeros_like(session.mri_volume, bool),
                    orientation=orientation,
                    slice_idx=slice_idx,
                )
                st.plotly_chart(fig_diff, use_container_width=True, height=650)

    # ==========================================================================
    # 3. Right Panel: Analysis Information & Voxel Inspector
    # ==========================================================================
    with col_info:
        st.markdown("### Analysis Information")

        # 3A. Selected Slice Coordinates
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Plane & Slice</div>
                <div class='metric-value'>{orientation.capitalize()} — {slice_idx + 1} / {max_slice}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3B. Quantitative Discovery Metrics
        prop_disc = int(np.sum(session.proposed_significant_mask)) if session.proposed_significant_mask is not None else 0
        base_disc = int(np.sum(session.baseline_significant_mask)) if session.baseline_significant_mask is not None else 0
        total_voxels = int(np.prod(session.shape))
        prop_pct = (prop_disc / total_voxels) * 100.0

        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Proposed Method Discoveries</div>
                <div class='metric-value'>{prop_disc:,} voxels ({prop_pct:.2f}%)</div>
            </div>
            <div class='metric-card'>
                <div class='metric-title'>Baseline (BH-FDR) Discoveries</div>
                <div class='metric-value'>{base_disc:,} voxels ({(base_disc / total_voxels) * 100.0:.2f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if session.ground_truth_mask is not None:
            gt_vox = int(np.sum(session.ground_truth_mask))
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-title'>Ground Truth Target</div>
                    <div class='metric-value'>{gt_vox:,} voxels</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3C. Spatial & Acquisition Metadata
        st.markdown("#### Volume Metadata")
        st.write(f"- **Dimensions**: `{session.shape[0]} × {session.shape[1]} × {session.shape[2]}`")
        st.write(f"- **Voxel Spacing**: `{session.voxel_spacing[0]:.1f} × {session.voxel_spacing[1]:.1f} × {session.voxel_spacing[2]:.1f} mm`")
        st.write(f"- **Nominal FDR Target**: `alpha = {session.metadata.get('nominal_alpha', 0.05)}`")

        # 3D. Voxel Coordinate Inspector
        st.markdown("#### Voxel Inspector")
        ix = st.number_input("Voxel X", 0, nx - 1, nx // 2)
        iy = st.number_input("Voxel Y", 0, ny - 1, ny // 2)
        iz = st.number_input("Voxel Z", 0, nz - 1, nz // 2)

        world_x, world_y, world_z = voxel_to_world_coordinates((ix, iy, iz), session.affine)
        raw_val = float(session.mri_volume[ix, iy, iz])
        stat_val = float(session.statistical_map[ix, iy, iz]) if session.statistical_map is not None else 0.0
        lis_val = float(session.lis_map[ix, iy, iz]) if session.lis_map is not None else 1.0
        is_sig = bool(session.proposed_significant_mask[ix, iy, iz]) if session.proposed_significant_mask is not None else False
        in_roi = bool(session.roi_mask[ix, iy, iz]) if session.roi_mask is not None else False

        st.markdown(
            f"""
            - **World Coordinates**: `({world_x:.1f}, {world_y:.1f}, {world_z:.1f}) mm`
            - **MRI Intensity**: `{raw_val:.2f}`
            - **Statistic (z-score)**: `{stat_val:.2f}`
            - **Local Index of Significance (LIS)**: `{lis_val:.4f}`
            - **FDR Significance**: `{'Significant' if is_sig else 'Not Significant'}`
            - **3D U-Net ROI**: `{'Inside' if in_roi else 'Outside'}`
            """
        )

        st.divider()
        st.caption(
            "Scientific Integrity Notice: All voxel values, affine matrices, and statistical quantities "
            "are preserved from raw data without cosmetic manipulation or artificial enhancement."
        )


if __name__ == "__main__":
    main()
