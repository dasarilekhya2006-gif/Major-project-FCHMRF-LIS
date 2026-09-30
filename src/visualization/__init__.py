"""Visualization module for neuroimaging slice viewing, spatial maps, and interactive dashboard analytics."""

from src.visualization.slice_viewer import plot_orthogonal_slices
from src.visualization.statistical_plots import (
    plot_fdr_power_calibration,
    plot_lis_distribution,
)
from src.visualization.spatial_maps import plot_side_by_side_discoveries
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
    process_uploaded_nifti_bytes,
)
from src.visualization.api_client import NeuroimagingAPIClient

__all__ = [
    "plot_orthogonal_slices",
    "plot_fdr_power_calibration",
    "plot_lis_distribution",
    "plot_side_by_side_discoveries",
    "build_interactive_slice_figure",
    "build_temperature_colormap",
    "extract_slice",
    "voxel_to_world_coordinates",
    "build_interactive_3d_brain_figure",
    "extract_surface_mesh",
    "build_method_comparison_figure",
    "build_difference_map_figure",
    "VisualizationPipelineSession",
    "load_research_benchmark_session",
    "process_uploaded_nifti_bytes",
    "NeuroimagingAPIClient",
]

