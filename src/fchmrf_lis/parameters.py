"""Parameter definitions and method distinction modes for fcHMRF-LIS spatial modeling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MethodMode(str, Enum):
    """Execution mode distinguishing original published fcHMRF-LIS vs our adaptive DL extension.

    - ORIGINAL_FCHMRF: Original published fcHMRF-LIS methodology.
      Uses spatially uniform prior P(theta_i = 1) = 1 - p0 derived purely from emission mixture,
      dense pairwise bilateral potential, mean-field variational inference, and step-up LIS FDR control.

    - ADAPTIVE_DL_FCHMRF: Project modification introducing deep-learning ROI prior integration.
      Blends emission-based mixture prior with continuous 3D U-Net prior map pi_i^(DL) via weighting
      hyperparameter lambda_dl in [0.0, 1.0]. Setting lambda_dl = 0 identically recovers original fcHMRF-LIS.
    """

    ORIGINAL_FCHMRF = "original_fchmrf"
    ADAPTIVE_DL_FCHMRF = "adaptive_dl_fchmrf"


@dataclass
class FCHMRFParameters:
    """Hyperparameters for fcHMRF graph construction, mean-field inference, and spatial FDR control.

    Attributes:
        method_mode: Execution mode ('original_fchmrf' or 'adaptive_dl_fchmrf').
        beta: Spatial coupling parameter (inverse temperature) controlling neighborhood smoothness.
        sigma_spatial: Bandwidth for spatial Euclidean distance kernel (in mm or voxel units).
        sigma_intensity: Bandwidth for radiometric / statistical contrast difference kernel.
        w_spatial: Weight of spatial bilateral kernel component.
        w_pure_spatial: Weight of pure spatial proximity kernel component.
        sigma_spatial_pure: Bandwidth for pure spatial kernel component.
        lambda_dl: Weight in [0.0, 1.0] for deep-learning ROI prior in unary energy.
                   (Ignored and enforced as 0.0 under ORIGINAL_FCHMRF mode).
        alpha: Nominal spatial False Discovery Rate control level (e.g. 0.05).
        max_iterations: Maximum number of mean-field variational inference iterations.
        tolerance: Convergence threshold on maximum absolute change in marginal beliefs.
        damping: Damping factor in (0.0, 1.0] for fixed-point mean-field updates.
        fix_null: Whether to fix emission null distribution to standard normal N(0, 1).
        neighbor_connectivity: Local lattice neighborhood connectivity for the sparse
            interaction graph. Must be 6 (face-adjacent), 18 (edge-adjacent), or
            26 (vertex-adjacent). This replaces the previous radius-based k-NN approach
            and guarantees memory usage of O(connectivity * N) rather than O(N^2).
            Default 26 (full 3x3x3 cube minus center) is the standard MRF connectivity
            for 3D neuroimaging data and captures all immediately adjacent interactions.
        max_valid_voxels: Hard upper limit on number of valid voxels to process.
            With lattice connectivity, typical whole-brain volumes (N~500K, connectivity=26)
            require ~115 MB for the sparse graph. Set to 2M as a conservative ceiling.
        max_graph_edges: Hard upper limit on total directed edges (nnz of K). With lattice
            connectivity, max edges = connectivity * N, so 26 * 2M = 52M < 100M default.
    """

    method_mode: MethodMode = MethodMode.ORIGINAL_FCHMRF
    beta: float = 0.8
    sigma_spatial: float = 2.5
    sigma_intensity: float = 1.5
    w_spatial: float = 1.0
    w_pure_spatial: float = 0.2
    sigma_spatial_pure: float = 4.0
    lambda_dl: float = 0.5
    alpha: float = 0.05
    max_iterations: int = 40
    tolerance: float = 1e-4
    damping: float = 0.5
    fix_null: bool = True
    neighbor_connectivity: int = 26
    max_valid_voxels: int = 2000000
    max_graph_edges: int = 100000000

    def __post_init__(self) -> None:
        if isinstance(self.method_mode, str):
            self.method_mode = MethodMode(self.method_mode.lower())

        if self.method_mode == MethodMode.ORIGINAL_FCHMRF:
            # Under original methodology, DL prior weighting is strictly 0.0
            self.lambda_dl = 0.0
        else:
            self.lambda_dl = max(0.0, min(1.0, float(self.lambda_dl)))

        self.beta = max(0.0, float(self.beta))
        self.sigma_spatial = max(1e-4, float(self.sigma_spatial))
        self.sigma_intensity = max(1e-4, float(self.sigma_intensity))
        self.alpha = max(1e-6, min(0.5, float(self.alpha)))
        self.damping = max(0.05, min(1.0, float(self.damping)))

        if int(self.neighbor_connectivity) not in (6, 18, 26):
            raise ValueError(
                f"neighbor_connectivity must be 6, 18, or 26; got {self.neighbor_connectivity}. "
                "These correspond to face-adjacent, edge-adjacent, and vertex-adjacent "
                "3D lattice neighborhoods."
            )
        self.neighbor_connectivity = int(self.neighbor_connectivity)
        self.max_valid_voxels = max(100, int(self.max_valid_voxels))
        self.max_graph_edges = max(100, int(self.max_graph_edges))
