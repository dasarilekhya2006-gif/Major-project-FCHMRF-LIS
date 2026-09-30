"""Grid evaluation and Information Criterion (BIC/AIC) model selection for fcHMRF."""

from typing import List, Optional, Tuple
import numpy as np

from src.core.logging import get_logger
from src.adaptive_parameter_selection.base_selector import BaseParameterSelector
from src.fchmrf_lis.graph_potentials import compute_unary_potentials
from src.fchmrf_lis.mean_field import run_mean_field_inference
from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters

logger = get_logger(__name__)


class GridBICSelector(BaseParameterSelector):
    """Selects beta and lambda_dl over a candidate discrete grid using Bayesian Information Criterion."""

    def __init__(
        self,
        beta_grid: Optional[List[float]] = None,
        lambda_grid: Optional[List[float]] = None,
    ) -> None:
        self.beta_grid = beta_grid if beta_grid is not None else [0.2, 0.5, 0.8, 1.2, 1.8]
        self.lambda_grid = lambda_grid if lambda_grid is not None else [0.0, 0.25, 0.5, 0.75, 1.0]

    def fit(
        self,
        z_scores: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        dl_prior_p1: np.ndarray,
        initial_mixture: TwoGroupMixtureParameters,
    ) -> Tuple[float, float, TwoGroupMixtureParameters]:
        """Evaluate pseudo-BIC across the Cartesian product of beta and lambda grid."""
        best_bic = np.inf
        best_beta = self.beta_grid[0]
        best_lambda = self.lambda_grid[0]
        N = len(z_scores)

        for beta_val in self.beta_grid:
            for lambda_val in self.lambda_grid:
                u0, u1 = compute_unary_potentials(
                    z_scores=z_scores,
                    mixture_params=initial_mixture,
                    dl_prior_p1=dl_prior_p1,
                    lambda_dl=lambda_val,
                )
                q1, _, _ = run_mean_field_inference(
                    unary_0=u0,
                    unary_1=u1,
                    neighbor_indices=neighbor_indices,
                    neighbor_weights=neighbor_weights,
                    beta=beta_val,
                    max_iterations=15,
                )

                # Variational free energy / pseudo log-likelihood approximation
                log_evidence = -np.sum(q1 * u1 + (1.0 - q1) * u0)
                # k = 4 parameters (beta, lambda, mu1, sigma1)
                bic = -2.0 * log_evidence + 4.0 * np.log(N)

                if bic < best_bic:
                    best_bic = bic
                    best_beta = beta_val
                    best_lambda = lambda_val

        logger.info(
            f"Grid BIC selected optimal: beta={best_beta:.2f}, lambda={best_lambda:.2f} (BIC={best_bic:.2f})"
        )
        return best_beta, best_lambda, initial_mixture
