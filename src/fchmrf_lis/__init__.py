"""fcHMRF-LIS spatial statistical modeling module.

Provides:
- Fully connected Hidden Markov Random Field potentials and bilateral kernels.
- Memory-bounded sparse local lattice neighborhood graph construction.
- Sparse Mean-Field variational inference engine (O(26·N) memory, no N×N matrix).
- Local Index of Significance (LIS) computation.
- Step-up spatial False Discovery Rate control.
- Clear distinction between original published fcHMRF-LIS and adaptive DL extensions.
- Publication-grade diagnostic visualization.
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
from src.fchmrf_lis.sparse_graph import (
    build_lattice_neighbor_graph,
    SparseNeighborGraph,
)
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult
from src.fchmrf_lis.visualization import plot_fchmrf_summary

__all__ = [
    "MethodMode",
    "FCHMRFParameters",
    "compute_dense_bilateral_matrix",
    "dense_bilateral_message_passing",
    "compute_unary_potentials",
    "compute_bilateral_weights",
    "run_dense_mean_field_inference",
    "run_mean_field_inference",
    "build_lattice_neighbor_graph",
    "SparseNeighborGraph",
    "compute_lis",
    "control_spatial_fdr_lis",
    "FCHMRFModel",
    "FCHMRFResult",
    "plot_fchmrf_summary",
]
