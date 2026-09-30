"""Phase 4: Voxel-Level Statistical Testing and Inference Module.

Provides:
  - Experimental design matrix construction and contrast specifications (One-Sample t, Two-Sample t, Paired t, GLM).
  - Voxel-wise hypothesis testing computing test statistics, p-values, z-scores, and effect sizes.
  - Optional spatial ROI masking (supporting masks from Phase 3 or reference atlases).
  - Validation routines for design rank, sample dimensions, and NaN/Inf handling.
  - Coordinate and affine-preserving NIfTI statistical volume outputs.
"""

from src.statistical_analysis.design import (
    AlternativeHypothesis,
    ExperimentalDesign,
    TestType,
    create_glm_design,
    create_one_sample_design,
    create_paired_design,
    create_two_sample_design,
)
from src.statistical_analysis.validation import validate_statistical_inputs
from src.statistical_analysis.voxel_testing import (
    StatisticalMaps,
    compute_statistics,
    compute_voxel_statistics,
    glm_voxelwise,
    one_sample_t_test,
    two_sample_t_test,
)
from src.statistical_analysis.visualization import plot_statistical_maps

__all__ = [
    "AlternativeHypothesis",
    "ExperimentalDesign",
    "TestType",
    "create_glm_design",
    "create_one_sample_design",
    "create_paired_design",
    "create_two_sample_design",
    "validate_statistical_inputs",
    "StatisticalMaps",
    "compute_statistics",
    "compute_voxel_statistics",
    "glm_voxelwise",
    "one_sample_t_test",
    "two_sample_t_test",
    "plot_statistical_maps",
]
