"""Modular neuroimaging preprocessing package.

Provides modular tools for missing voxel imputation, spatial masking, intensity normalization,
denoising, spatial resampling, dimension standardization, pipeline orchestration, and QC reporting.
"""

from src.preprocessing.schemas import (
    PreprocessingConfig,
    PreprocessingMetadata,
    PreprocessingStepRecord,
)
from src.preprocessing.pipeline import (
    PreprocessedVolume,
    PreprocessingPipeline,
    preprocess_neuroimaging_volume,
)
from src.preprocessing.imputation import handle_missing_voxels
from src.preprocessing.normalization import (
    denormalize_intensities,
    normalize_intensities,
    standardize_zscores,
    convert_pvalues_to_zscores,
)
from src.preprocessing.denoising import denoise_volume
from src.preprocessing.resampling import resample_to_spacing
from src.preprocessing.resizing import standardize_spatial_dimensions
from src.preprocessing.masking import (
    apply_mask,
    compute_brain_mask,
    extract_bounding_box,
    load_or_compute_mask,
    unmask,
)
from src.preprocessing.smoothing import smooth_volume
from src.preprocessing.spatial_graph import (
    extract_masked_coordinates,
    build_knn_spatial_adjacency,
)
from src.preprocessing.qc import generate_preprocessing_qc_plot

__all__ = [
    "PreprocessingConfig",
    "PreprocessingMetadata",
    "PreprocessingStepRecord",
    "PreprocessedVolume",
    "PreprocessingPipeline",
    "preprocess_neuroimaging_volume",
    "handle_missing_voxels",
    "normalize_intensities",
    "denormalize_intensities",
    "standardize_zscores",
    "convert_pvalues_to_zscores",
    "denoise_volume",
    "resample_to_spacing",
    "standardize_spatial_dimensions",
    "apply_mask",
    "compute_brain_mask",
    "extract_bounding_box",
    "load_or_compute_mask",
    "unmask",
    "smooth_volume",
    "extract_masked_coordinates",
    "build_knn_spatial_adjacency",
    "generate_preprocessing_qc_plot",
]
