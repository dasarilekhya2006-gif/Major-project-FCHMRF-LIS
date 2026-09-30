"""Mean-Field Expectation-Maximization (MF-EM) parameter selection algorithm."""

from typing import Tuple
import numpy as np

from src.core.logging import get_logger
from src.adaptive_parameter_selection.base_selector import BaseParameterSelector
from src.adaptive_parameter_selection.pseudo_likelihood import compute_mple_beta
from src.fchmrf_lis.graph_potentials import compute_unary_potentials
from src.fchmrf_lis.mean_field import run_mean_field_inference
from src.statistical_analysis.mixture_models import TwoGroupMixtureParameters

logger = get_logger(__name__)


class MeanFieldEMSelector(BaseParameterSelector):
    """Mean-Field Expectation-Maximization algorithm for fcHMRF hyperparameter estimation.

    Alternates between:
      E-step: Inferring marginal posterior states Q(theta_i) using Mean-Field approximation.
      M-step: Updating spatial interaction strength beta via pseudo-likelihood,
              updating deep-learning weighting lambda_dl, and refreshing mixture emissions.
    """

    def __init__(
        self,
        max_em_iter: int = 15,
        tolerance: float = 1e-3,
        beta_init: float = 0.8,
        lambda_init: float = 0.5,
        beta_bounds: Tuple[float, float] = (0.05, 3.0),
        lambda_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.max_em_iter = max_em_iter
        self.tolerance = tolerance
        self.beta = beta_init
        self.lambda_dl = lambda_init
        self.beta_bounds = beta_bounds
        self.lambda_bounds = lambda_bounds

    def fit(
        self,
        z_scores: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        dl_prior_p1: np.ndarray,
        initial_mixture: TwoGroupMixtureParameters,
    ) -> Tuple[float, float, TwoGroupMixtureParameters]:
        """Execute the MF-EM optimization cycle.

        Args:
            z_scores: 1D array of z-statistics (length N).
            neighbor_indices: (N, K) neighbor index matrix.
            neighbor_weights: (N, K) edge affinity weights.
            dl_prior_p1: (N,) deep learning prior probabilities.
            initial_mixture: Initial emission mixture parameters.

        Returns:
            Tuple[float, float, TwoGroupMixtureParameters]: (beta, lambda_dl, mixture_params)
        """
        mixture = initial_mixture
        prev_beta = self.beta
        prev_lambda = self.lambda_dl
        N = len(z_scores)

        logger.info(
            f"Starting MF-EM parameter selection (initial beta={self.beta:.2f}, lambda={self.lambda_dl:.2f})"
        )

        for it in range(self.max_em_iter):
            # 1. E-Step: Compute unary potentials and run Mean-Field inference
            u0, u1 = compute_unary_potentials(
                z_scores=z_scores,
                mixture_params=mixture,
                dl_prior_p1=dl_prior_p1,
                lambda_dl=self.lambda_dl,
            )
            q1, mf_iters, mf_delta = run_mean_field_inference(
                unary_0=u0,
                unary_1=u1,
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                beta=self.beta,
                max_iterations=20,
            )

            # 2. M-Step:
            # 2a. Update beta using Maximum Pseudo-Likelihood on inferred marginals
            new_beta = compute_mple_beta(
                binary_states_or_marginals=q1,
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                beta_bounds=self.beta_bounds,
            )

            # 2b. Update lambda_dl via correlation between DL prior and inferred posterior
            # If DL prior correlates strongly with posterior activations, increase lambda
            dl_centered = dl_prior_p1 - np.mean(dl_prior_p1)
            q_centered = q1 - np.mean(q1)
            corr_denom = (np.std(dl_prior_p1) * np.std(q1) + 1e-8)
            alignment = float(np.mean(dl_centered * q_centered) / corr_denom)
            # Map correlation [-1, 1] to lambda bounds [0, 1]
            new_lambda = float(np.clip(0.5 + 0.5 * alignment, self.lambda_bounds[0], self.lambda_bounds[1]))

            # 2c. Update non-null emission mean mu1 and sigma1
            sum_q1 = np.sum(q1)
            if sum_q1 > 5.0:
                new_mu1 = float(np.sum(q1 * z_scores) / sum_q1)
                new_sigma1 = float(np.sqrt(np.sum(q1 * (z_scores - new_mu1) ** 2) / sum_q1))
                mixture = TwoGroupMixtureParameters(
                    p0=float(np.clip(1.0 - sum_q1 / N, 0.5, 0.999)),
                    mu0=mixture.mu0,
                    sigma0=mixture.sigma0,
                    mu1=max(new_mu1, mixture.mu0 + 0.5),
                    sigma1=max(new_sigma1, 0.5),
                )

            diff_beta = abs(new_beta - self.beta)
            diff_lambda = abs(new_lambda - self.lambda_dl)

            self.beta = new_beta
            self.lambda_dl = new_lambda

            logger.debug(
                f"MF-EM iter {it + 1}: beta={self.beta:.4f} (diff={diff_beta:.4f}), "
                f"lambda={self.lambda_dl:.4f} (diff={diff_lambda:.4f})"
            )

            if max(diff_beta, diff_lambda) < self.tolerance:
                logger.info(f"MF-EM converged at iteration {it + 1}.")
                break

        return self.beta, self.lambda_dl, mixture
