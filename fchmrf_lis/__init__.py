"""Phase 5: Fully Connected Hidden Markov Random Field with Local Index of Significance (fcHMRF-LIS).

This module provides the core spatial statistical inference engine for false discovery rate (FDR) control:
- Dense pairwise bilateral Gaussian affinity modeling.
- Coordinate-ascent Mean-Field variational inference.
- Local Index of Significance (LIS) computation (Sun & Cai 2009).
- Step-up spatial False Discovery Rate control.
- Clear methodological separation between published base paper method and adaptive DL prior extensions.
"""

from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.dense_kernel import (
    compute_dense_bilateral_matrix,
    dense_bilateral_message_passing,
)
from src.fchmrf_lis.graph_potentials import (
    compute_bilateral_weights,
    compute_unary_potentials,
)
from src.fchmrf_lis.mean_field import (
    run_dense_mean_field_inference,
    run_mean_field_inference,
)
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis, control_spatial_fdr
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult
from src.fchmrf_lis.visualization import plot_fchmrf_summary

__all__ = [
    "FCHMRFModel",
    "FCHMRFParameters",
    "MethodMode",
    "FCHMRFResult",
    "compute_unary_potentials",
    "compute_bilateral_weights",
    "compute_dense_bilateral_matrix",
    "dense_bilateral_message_passing",
    "run_dense_mean_field_inference",
    "run_mean_field_inference",
    "compute_lis",
    "control_spatial_fdr_lis",
    "control_spatial_fdr",
    "plot_fchmrf_summary",
]
