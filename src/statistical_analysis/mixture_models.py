"""Two-group mixture model estimation for neuroimaging statistical emission densities."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import stats

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TwoGroupMixtureParameters:
    """Parameters for the two-group emission density: f(z) = p0 * f0(z) + (1 - p0) * f1(z).

    Attributes:
        p0: Prior probability of null state (theta = 0).
        mu0: Mean of the null distribution (default: 0.0 or empirical null).
        sigma0: Standard deviation of null distribution (default: 1.0).
        mu1: Mean of the non-null activation distribution (mu1 > mu0).
        sigma1: Standard deviation of non-null distribution.
    """

    p0: float = 0.90
    mu0: float = 0.0
    sigma0: float = 1.0
    mu1: float = 2.5
    sigma1: float = 1.2

    @property
    def p1(self) -> float:
        return 1.0 - self.p0

    def log_likelihood_null(self, z: np.ndarray) -> np.ndarray:
        """Log f_0(z) under Gaussian null."""
        return stats.norm.logpdf(z, loc=self.mu0, scale=self.sigma0)

    def log_likelihood_active(self, z: np.ndarray) -> np.ndarray:
        """Log f_1(z) under Gaussian active state."""
        return stats.norm.logpdf(z, loc=self.mu1, scale=self.sigma1)


def fit_two_group_mixture_em(
    z_scores: np.ndarray,
    fix_null: bool = True,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> TwoGroupMixtureParameters:
    """Fit a two-group Gaussian mixture model via expectation-maximization.

    Args:
        z_scores: 1D array of z-statistics across voxels.
        fix_null: If True, fixes mu0=0.0 and sigma0=1.0 (standard normal null).
        max_iter: Maximum number of EM iterations.
        tol: Convergence tolerance for log-likelihood.

    Returns:
        TwoGroupMixtureParameters: Estimated mixture parameters.
    """
    z = np.asarray(z_scores, dtype=np.float64).ravel()
    N = len(z)

    # Initial guesses
    p0 = 0.90
    raw_iqr = float(stats.iqr(z) / 1.349)
    mu0 = 0.0 if fix_null else float(np.median(z))
    sigma0 = 1.0 if fix_null else (raw_iqr if raw_iqr >= 0.5 and not np.isnan(raw_iqr) else 1.0)

    top_tail = z[z > np.percentile(z, 95)]
    if len(top_tail) == 0 or np.isnan(np.mean(top_tail)):
        mu1 = mu0 + 2.0
        sigma1 = 1.0
    else:
        mu1 = float(np.mean(top_tail))
        sigma1 = float(np.std(top_tail))
    if np.isnan(sigma1) or sigma1 < 0.5:
        sigma1 = 1.0

    prev_ll = -np.inf

    for it in range(max_iter):
        # E-step: compute posterior responsibilities gamma_i = P(theta_i = 1 | z_i)
        f0 = stats.norm.pdf(z, loc=mu0, scale=sigma0)
        f1 = stats.norm.pdf(z, loc=mu1, scale=sigma1)

        denom = p0 * f0 + (1.0 - p0) * f1 + 1e-12
        gamma1 = ((1.0 - p0) * f1) / denom
        gamma0 = 1.0 - gamma1

        # Check log-likelihood convergence
        current_ll = np.sum(np.log(denom))
        if abs(current_ll - prev_ll) < tol:
            logger.debug(f"Two-group EM converged at iteration {it}")
            break
        prev_ll = current_ll

        # M-step: update parameters
        sum_gamma1 = np.sum(gamma1)
        sum_gamma0 = np.sum(gamma0)

        p0 = float(np.clip(sum_gamma0 / N, 0.50, 0.999))

        if not fix_null:
            mu0 = float(np.sum(gamma0 * z) / max(sum_gamma0, 1e-6))
            sigma0_val = float(np.sqrt(np.sum(gamma0 * (z - mu0) ** 2) / max(sum_gamma0, 1e-6)))
            sigma0 = sigma0_val if sigma0_val >= 0.5 and not np.isnan(sigma0_val) else 0.5

        mu1_val = float(np.sum(gamma1 * z) / max(sum_gamma1, 1e-6))
        mu1 = mu1_val if not np.isnan(mu1_val) else mu0 + 2.0
        sigma1_val = float(np.sqrt(np.sum(gamma1 * (z - mu1) ** 2) / max(sum_gamma1, 1e-6)))
        sigma1 = sigma1_val if sigma1_val >= 0.5 and not np.isnan(sigma1_val) else 0.5

        # Constrain active mean to be greater than null mean
        if mu1 <= mu0 + 0.5:
            mu1 = mu0 + 1.0

    params = TwoGroupMixtureParameters(
        p0=p0,
        mu0=mu0,
        sigma0=max(sigma0, 0.5),
        mu1=mu1,
        sigma1=max(sigma1, 0.5),
    )
    logger.info(
        f"Fitted mixture parameters: p0={params.p0:.3f}, mu0={params.mu0:.2f}, "
        f"sigma0={params.sigma0:.2f}, mu1={params.mu1:.2f}, sigma1={params.sigma1:.2f}"
    )
    return params
