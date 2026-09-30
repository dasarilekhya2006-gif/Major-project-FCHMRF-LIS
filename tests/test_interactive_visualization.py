"""Unit and integration tests for Phase 10 interactive neuroimaging visualization."""

import numpy as np
import pytest

from src.visualization.interactive_2d import (
    build_interactive_slice_figure,
    build_temperature_colormap,
    extract_slice,
    voxel_to_world_coordinates,
)
from src.visualization.interactive_3d import (
    build_interactive_3d_brain_figure,
    extract_surface_mesh,
)
from src.visualization.comparison_view import (
    build_method_comparison_figure,
    build_difference_map_figure,
)
from src.visualization.pipeline_bridge import (
    VisualizationPipelineSession,
    load_research_benchmark_session,
)


class TestInteractive2DViewer:
    """Tests for 2D tri-planar slice extraction and interactive figure building."""

    def test_voxel_to_world_coordinates(self) -> None:
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        affine[0, 3] = 10.0  # X offset
        affine[1, 3] = -20.0 # Y offset

        wx, wy, wz = voxel_to_world_coordinates((5, 10, 15), affine)
        assert wx == pytest.approx(5 * 2.0 + 10.0)
        assert wy == pytest.approx(10 * 3.0 - 20.0)
        assert wz == pytest.approx(15 * 4.0)

    def test_extract_slice_orientations(self) -> None:
        vol = np.zeros((20, 30, 40), dtype=np.float32)
        vol[5, 10, 15] = 42.0

        axial = extract_slice(vol, "axial", 15)
        coronal = extract_slice(vol, "coronal", 10)
        sagittal = extract_slice(vol, "sagittal", 5)

        assert axial.ndim == 2
        assert coronal.ndim == 2
        assert sagittal.ndim == 2

        with pytest.raises(ValueError):
            extract_slice(vol, "invalid_orientation", 0)

    def test_build_interactive_slice_figure(self) -> None:
        shape = (16, 16, 8)
        mri = np.random.randn(*shape).astype(np.float32)
        roi = np.zeros(shape, bool)
        roi[4:8, 4:8, 2:5] = True
        stat = np.random.randn(*shape).astype(np.float32)
        stat[roi] += 3.0
        lis = np.ones(shape, dtype=np.float32)
        lis[roi] = 0.01
        sig = roi.copy()

        fig = build_interactive_slice_figure(
            mri_volume=mri,
            orientation="axial",
            slice_idx=3,
            roi_mask=roi,
            statistical_map=stat,
            lis_map=lis,
            fdr_significant_mask=sig,
            show_fdr_significant=True,
            show_roi=True,
        )

        assert fig is not None
        assert len(fig.data) >= 2  # base mri + roi + fdr + hover
        assert fig.layout.paper_bgcolor == "#121212"

    def test_build_temperature_colormap(self) -> None:
        cscale = build_temperature_colormap()
        assert len(cscale) >= 3
        assert cscale[0][0] == 0.0
        assert cscale[-1][0] == 1.0


class TestInteractive3DBrain:
    """Tests for 3D marching cubes surface extraction and 3D figure building."""

    def test_extract_surface_mesh_sphere(self) -> None:
        shape = (20, 20, 20)
        vol = np.zeros(shape, dtype=np.float32)
        # Create solid sphere
        x, y, z = np.ogrid[:20, :20, :20]
        dist = np.sqrt((x - 10) ** 2 + (y - 10) ** 2 + (z - 10) ** 2)
        vol[dist <= 6] = 1.0

        mesh = extract_surface_mesh(vol, level=0.5, step_size=1)
        assert mesh is not None
        vx, vy, vz, fi, fj, fk = mesh
        assert len(vx) > 0
        assert len(fi) > 0

    def test_build_interactive_3d_brain_figure(self) -> None:
        shape = (16, 16, 16)
        mri = np.zeros(shape, dtype=np.float32)
        mri[4:12, 4:12, 4:12] = 1.0
        roi = np.zeros(shape, bool)
        roi[6:10, 6:10, 6:10] = True

        fig = build_interactive_3d_brain_figure(
            mri_volume=mri,
            roi_mask=roi,
            significant_mask=roi,
            show_brain_shell=True,
            show_roi_mesh=True,
            show_significant_mesh=True,
            step_size=1,
        )
        assert fig is not None
        assert fig.layout.scene is not None


class TestComparisonView:
    """Tests for baseline vs proposed synchronized comparison and difference map."""

    def test_build_method_comparison_figure(self) -> None:
        shape = (16, 16, 8)
        mri = np.ones(shape, dtype=np.float32)
        base_sig = np.zeros(shape, bool)
        base_sig[4:7, 4:7, 3] = True
        prop_sig = np.zeros(shape, bool)
        prop_sig[4:9, 4:9, 3] = True

        fig = build_method_comparison_figure(
            mri_volume=mri,
            baseline_significant_mask=base_sig,
            proposed_significant_mask=prop_sig,
            orientation="axial",
            slice_idx=3,
        )
        assert fig is not None
        assert len(fig.data) >= 2

    def test_build_difference_map_figure(self) -> None:
        shape = (16, 16, 8)
        mri = np.ones(shape, dtype=np.float32)
        base_sig = np.zeros(shape, bool)
        base_sig[4:6, 4:6, 3] = True
        prop_sig = np.zeros(shape, bool)
        prop_sig[5:8, 5:8, 3] = True

        fig = build_difference_map_figure(
            mri_volume=mri,
            baseline_significant_mask=base_sig,
            proposed_significant_mask=prop_sig,
            orientation="axial",
            slice_idx=3,
        )
        assert fig is not None
        assert len(fig.data) >= 1


class TestPipelineBridge:
    """Tests for pipeline bridge session orchestration and benchmark loading."""

    def test_load_research_benchmark_session(self) -> None:
        session = load_research_benchmark_session(
            shape=(16, 16, 8),
            snr=3.0,
            nominal_alpha=0.05,
            seed=42,
        )
        assert isinstance(session, VisualizationPipelineSession)
        assert session.mri_volume.shape == (16, 16, 8)
        assert session.proposed_significant_mask is not None
        assert session.baseline_significant_mask is not None
        assert session.lis_map is not None
        assert session.ground_truth_mask is not None
        assert "proposed_discoveries" in session.metadata
