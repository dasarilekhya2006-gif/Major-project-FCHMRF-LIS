"""Graph potentials for fully connected Hidden Markov Random Field (fcHMRF).

Defines:
1. Unary energy potentials distinguishing original fcHMRF-LIS and adaptive DL extensions.
2. Pairwise bilateral potentials combining spatial Euclidean proximity and feature affinity.
"""

from typing import Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger
from src.fchmrf_lis.parameters import MethodMode
from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters

logger = get_logger(__name__)


def compute_unary_potentials(
    z_scores: np.ndarray,
    mixture_params: TwoGroupMixtureParameters,
    dl_prior_p1: Optional[np.ndarray] = None,
    lambda_dl: float = 0.5,
    method_mode: Union[MethodMode, str] = MethodMode.ORIGINAL_FCHMRF,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unary energy potentials psi_i(0) and psi_i(1) for null and active states.

    =============================================================================
    MATHEMATICAL FORMULATION:
    =============================================================================
    Under the Gibbs random field:
        psi_i(theta_i) = -log f_{theta_i}(z_i) - log P(theta_i)

    1. ORIGINAL fcHMRF-LIS METHODOLOGY (method_mode='original_fchmrf'):
       Uses a spatially homogeneous prior derived purely from the two-group mixture:
           P(theta_i = 0) = p_0
           P(theta_i = 1) = 1 - p_0
       giving:
           psi_i(0) = -log f_0(z_i) - log p_0
           psi_i(1) = -log f_1(z_i) - log (1 - p_0)

    2. ADAPTIVE DL-ASSISTED MODIFICATION (method_mode='adaptive_dl_fchmrf'):
       Introduces a spatially varying prior informed by the 3D U-Net localization map:
           P(theta_i = 1) = lambda_dl * pi_i^(DL) + (1 - lambda_dl) * (1 - p_0)
           P(theta_i = 0) = 1 - P(theta_i = 1)
       giving:
           psi_i(0) = -log f_0(z_i) - log P(theta_i = 0)
           psi_i(1) = -log f_1(z_i) - log P(theta_i = 1)

       *Note*: Setting lambda_dl = 0.0 identically collapses to the original fcHMRF-LIS.
    =============================================================================

    Args:
        z_scores: 1D array of z-statistics (length N).
        mixture_params: Fitted two-group Gaussian mixture emission parameters.
        dl_prior_p1: Optional 1D array of DL prior activation probabilities pi_i in [0, 1].
        lambda_dl: Weighting hyperparameter in [0.0, 1.0] for DL prior vs empirical prior.
        method_mode: 'original_fchmrf' or 'adaptive_dl_fchmrf'.
        eps: Numerical guard to prevent log(0).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - unary_0: (N,) energy potential for state 0.
            - unary_1: (N,) energy potential for state 1.
    """
    mode = MethodMode(method_mode.lower() if isinstance(method_mode, str) else method_mode)

    log_f0 = mixture_params.log_likelihood_null(z_scores)
    log_f1 = mixture_params.log_likelihood_active(z_scores)

    if mode == MethodMode.ADAPTIVE_DL_FCHMRF and dl_prior_p1 is not None and lambda_dl > 0.0:
        dl_prior_clean = np.clip(np.asarray(dl_prior_p1, dtype=np.float64), 0.0, 1.0)
        empirical_p1 = 1.0 - mixture_params.p0
        prior_1 = lambda_dl * dl_prior_clean + (1.0 - lambda_dl) * empirical_p1
    else:
        # Original fcHMRF methodology: spatially homogeneous mixture prior
        prior_1 = np.full_like(z_scores, 1.0 - mixture_params.p0)

    prior_1 = np.clip(prior_1, eps, 1.0 - eps)
    prior_0 = 1.0 - prior_1

    unary_0 = -log_f0 - np.log(prior_0)
    unary_1 = -log_f1 - np.log(prior_1)

    return unary_0.astype(np.float64), unary_1.astype(np.float64)


def compute_bilateral_weights(
    distances: np.ndarray,
    intensity_diffs: np.ndarray,
    sigma_spatial: float = 2.5,
    sigma_intensity: float = 1.5,
) -> np.ndarray:
    """Compute bilateral kernel weights for fcHMRF edge interactions:

        w_ij = exp( - ||p_i - p_j||^2 / (2 * sigma_s^2) ) * exp( - (z_i - z_j)^2 / (2 * sigma_r^2) )

    Args:
        distances: Spatial Euclidean distances matrix or neighbor distances.
        intensity_diffs: Statistical contrast absolute differences |z_i - z_j|.
        sigma_spatial: Spatial kernel bandwidth (in voxels or mm).
        sigma_intensity: Radiometric / statistical feature bandwidth.

    Returns:
        np.ndarray: Computed edge affinity weights in [0, 1].
    """
    spatial_term = np.exp(- (distances ** 2) / (2.0 * max(sigma_spatial, 1e-4) ** 2))
    feature_term = np.exp(- (intensity_diffs ** 2) / (2.0 * max(sigma_intensity, 1e-4) ** 2))
    return spatial_term * feature_term
