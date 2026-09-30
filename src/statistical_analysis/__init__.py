"""Statistical analysis module for voxel-level testing, FDR control, q-values, locfdr, and mixture modeling."""

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
    glm_voxelwise,
    one_sample_t_test,
    two_sample_t_test,
)
from src.statistical_analysis.visualization import plot_statistical_maps
from src.statistical_analysis.classical_fdr import benjamini_hochberg, benjamini_yekutieli
from src.statistical_analysis.storey_qvalue import estimate_pi0, storey_qvalues
from src.statistical_analysis.local_fdr import (
    estimate_empirical_null_central_matching,
    efron_locfdr,
)
from src.statistical_analysis.mixture_models import (
    TwoGroupMixtureParameters,
    fit_two_group_mixture_em,
)

__all__ = [
    # Phase 4 Voxel-Level Testing
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
    "glm_voxelwise",
    "one_sample_t_test",
    "two_sample_t_test",
    "plot_statistical_maps",
    # Classical & Spatial FDR
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "estimate_pi0",
    "storey_qvalues",
    "estimate_empirical_null_central_matching",
    "efron_locfdr",
    "TwoGroupMixtureParameters",
    "fit_two_group_mixture_em",
]
