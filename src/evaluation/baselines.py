"""Implementation of standard baselines and comparative methods for Phase 8 benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats

from src.core.logging import get_logger
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr

logger = get_logger(__name__)


@dataclass
class MethodOutput:
    """Output container for any evaluated spatial multiple-testing method."""

    method_name: str
    significant_mask: np.ndarray  # Binary mask of declared significant voxels
    test_statistic_map: np.ndarray  # Decision statistic (e.g. 1 - p, 1 - lfdr, or 1 - LIS)
    threshold_value: float  # Critical cutoff threshold used
    num_discoveries: int
    runtime_seconds: float
    peak_memory_mb: float
    metadata: Dict[str, Any]


class BaseCorrectionMethod(ABC):
    """Abstract base class for all comparative spatial multiple-testing methods."""

    def __init__(self, name: str, alpha: float = 0.05) -> None:
        self.name = name
        self.alpha = alpha

    @abstractmethod
    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """Core algorithm logic returning (significant_mask, stat_map, threshold, metadata)."""
        pass

    def run(
        self,
        z_scores: np.ndarray,
        p_values: Optional[np.ndarray] = None,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> MethodOutput:
        """Execute method with automated wall-clock and peak memory benchmarking."""
        if p_values is None:
            # Two-sided p-values from standard normal if not provided
            p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))
            p_values = np.clip(p_values, 1e-15, 1.0)

        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()

        sig_mask, stat_map, thresh, meta = self._execute(
            z_scores=z_scores,
            p_values=p_values,
            dl_prior=dl_prior,
            spatial_mask=spatial_mask,
            **kwargs,
        )

        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024.0 * 1024.0)

        num_disc = int(np.sum(sig_mask))

        return MethodOutput(
            method_name=self.name,
            significant_mask=sig_mask.astype(bool),
            test_statistic_map=stat_map,
            threshold_value=thresh,
            num_discoveries=num_disc,
            runtime_seconds=elapsed,
            peak_memory_mb=peak_mb,
            metadata=meta,
        )


# ==============================================================================
# Baseline 1: Standard Voxel-Wise Benjamini-Hochberg FDR (BH-FDR)
# ==============================================================================

class BenjaminiHochbergCorrection(BaseCorrectionMethod):
    """Standard voxel-wise False Discovery Rate control via Benjamini-Hochberg (1995).

    Ranks unadjusted p-values: P_(1) <= P_(2) <= ... <= P_(m)
    Threshold: k = max { i : P_(i) <= (i / m) * alpha }
    Rejects H_0 for all voxels with P_i <= P_(k).
    """

    def __init__(self, alpha: float = 0.05, name: str = "Benjamini-Hochberg (BH-FDR)") -> None:
        super().__init__(name=name, alpha=alpha)

    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        shape = z_scores.shape
        if spatial_mask is not None:
            mask = spatial_mask.astype(bool)
            p_flat = p_values[mask].ravel()
        else:
            mask = np.ones(shape, dtype=bool)
            p_flat = p_values.ravel()

        m = len(p_flat)
        if m == 0:
            return np.zeros(shape, dtype=bool), np.zeros(shape), 0.0, {}

        # Sort p-values
        sort_idx = np.argsort(p_flat)
        p_sorted = p_flat[sort_idx]

        # BH critical line: (i / m) * alpha for i = 1, ..., m
        ranks = np.arange(1, m + 1)
        bh_thresholds = (ranks / m) * self.alpha
        valid = p_sorted <= bh_thresholds

        if np.any(valid):
            k = int(np.max(np.where(valid)[0]))
            p_cutoff = float(p_sorted[k])
            sig_flat = p_flat <= p_cutoff
        else:
            p_cutoff = 0.0
            sig_flat = np.zeros(m, dtype=bool)

        sig_mask = np.zeros(shape, dtype=bool)
        sig_mask[mask] = sig_flat

        stat_map = np.zeros(shape, dtype=np.float32)
        stat_map[mask] = 1.0 - p_flat

        meta = {
            "p_cutoff": p_cutoff,
            "total_tested_voxels": m,
            "bh_formula": "p_(k) <= (k/m)*alpha",
        }
        return sig_mask, stat_map, p_cutoff, meta


# ==============================================================================
# Baseline 2: Efron's Local False Discovery Rate (lfdr)
# ==============================================================================

class LocalFDRCorrection(BaseCorrectionMethod):
    """Efron's Local False Discovery Rate (lfdr) without spatial regularization.

    Mixture model: f(z) = p0 * f0(z) + (1 - p0) * f1(z)
    Local FDR: lfdr(z_i) = p0 * f0(z_i) / f(z_i)
    Controls Bayesian FDR at alpha:
        k = max { k : (1/k) * sum_{i=1}^k lfdr_(i) <= alpha }
    """

    def __init__(
        self,
        alpha: float = 0.05,
        p0_upper_bound: float = 0.95,
        name: str = "Local FDR (Efron lfdr)",
    ) -> None:
        super().__init__(name=name, alpha=alpha)
        self.p0_upper_bound = p0_upper_bound

    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        shape = z_scores.shape
        if spatial_mask is not None:
            mask = spatial_mask.astype(bool)
            z_flat = z_scores[mask].ravel()
        else:
            mask = np.ones(shape, dtype=bool)
            z_flat = z_scores.ravel()

        n = len(z_flat)
        if n == 0:
            return np.zeros(shape, dtype=bool), np.zeros(shape), 0.0, {}

        # 1. Estimate marginal mixture density f(z) via Gaussian KDE
        # Add slight jitter if variance is near-zero to prevent singular covariance matrix
        if np.std(z_flat) < 1e-4:
            rng = np.random.default_rng(42)
            z_eval = z_flat + 1e-4 * rng.normal(size=len(z_flat))
        else:
            z_eval = z_flat

        try:
            kde = stats.gaussian_kde(z_eval)
            f_z = np.maximum(kde.evaluate(z_flat), 1e-12)
            f_zero = float(kde.evaluate([0.0])[0])
        except Exception:
            # Fallback histogram/normal estimation
            mu, sigma = float(np.mean(z_flat)), float(np.std(z_flat)) + 1e-6
            f_z = stats.norm.pdf(z_flat, loc=mu, scale=sigma)
            f_zero = float(stats.norm.pdf(0.0, loc=mu, scale=sigma))

        # 2. Standard theoretical null f0(z) ~ N(0, 1)
        f0_z = stats.norm.pdf(z_flat, loc=0.0, scale=1.0)

        # 3. Estimate null proportion p0 using central zero-density ratio: p0 <= min(f(0)/f0(0), 1.0)
        p0_est = min(self.p0_upper_bound, float(f_zero / stats.norm.pdf(0.0)))
        p0_est = max(0.1, min(p0_est, 1.0))

        # 4. Compute Local FDR: lfdr_i = min(1.0, p0 * f0(z_i) / f(z_i))
        lfdr = np.clip(p0_est * f0_z / f_z, 0.0, 1.0)

        # 5. Control average local FDR (Bayes FDR) <= alpha
        sort_idx = np.argsort(lfdr)
        lfdr_sorted = lfdr[sort_idx]
        cum_avg_fdr = np.cumsum(lfdr_sorted) / np.arange(1, n + 1)

        valid = cum_avg_fdr <= self.alpha
        if np.any(valid):
            k = int(np.max(np.where(valid)[0]))
            lfdr_cutoff = float(lfdr_sorted[k])
            sig_flat = lfdr <= lfdr_cutoff
        else:
            lfdr_cutoff = 0.0
            sig_flat = np.zeros(n, dtype=bool)

        sig_mask = np.zeros(shape, dtype=bool)
        sig_mask[mask] = sig_flat

        stat_map = np.zeros(shape, dtype=np.float32)
        stat_map[mask] = 1.0 - lfdr

        meta = {
            "p0_estimated": p0_est,
            "lfdr_cutoff": lfdr_cutoff,
            "total_tested_voxels": n,
        }
        return sig_mask, stat_map, lfdr_cutoff, meta


# ==============================================================================
# Method 3: Original Published fcHMRF-LIS (Homogeneous Prior, lambda_dl = 0.0)
# ==============================================================================

class OriginalFCHMRFLISCorrection(BaseCorrectionMethod):
    """Original published fcHMRF-LIS implementation.

    - Dense bilateral kernel message passing across all voxel pairs.
    - Spatially uniform emission prior P(theta_i = 1) = 1 - p0 (lambda_DL = 0.0).
    - FDR control via Local Index of Significance (LIS) ranking.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.8,
        sigma_spatial: float = 2.5,
        sigma_intensity: float = 1.5,
        max_iterations: int = 35,
        name: str = "Original fcHMRF-LIS (lambda_dl=0)",
    ) -> None:
        super().__init__(name=name, alpha=alpha)
        self.beta = beta
        self.sigma_spatial = sigma_spatial
        self.sigma_intensity = sigma_intensity
        self.max_iterations = max_iterations

    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        params = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            beta=self.beta,
            sigma_spatial=self.sigma_spatial,
            sigma_intensity=self.sigma_intensity,
            lambda_dl=0.0,  # Strictly zero in original published method
            alpha=self.alpha,
            max_iterations=self.max_iterations,
        )
        model = FCHMRFModel(params=params)
        res = model.fit_predict(
            z_scores=z_scores,
            dl_prior=None,  # Spatially homogeneous prior
            spatial_mask=spatial_mask,
        )
        meta = {
            "method_mode": res.method_mode,
            "iterations_to_converge": res.iterations_to_converge,
            "lis_threshold": res.fdr_results["lis_threshold"],
            "estimated_fdr": res.fdr_results["estimated_fdr"],
        }
        return res.significant_mask, res.lis_map, float(res.fdr_results["lis_threshold"]), meta


# ==============================================================================
# Method 4: Proposed ROI-Assisted fcHMRF-LIS (Fixed lambda_dl = 0.5)
# ==============================================================================

class ROIAssistedFCHMRFLISCorrection(BaseCorrectionMethod):
    """Proposed ROI-Assisted fcHMRF-LIS with fixed default hyperparameters.

    - Integrates continuous 3D U-Net localization prior pi_i^DL into unary potentials.
    - Dense bilateral kernel spatial regularizer.
    - Default hyperparameters: beta=0.8, sigma_s=2.5, sigma_r=1.5, lambda_dl=0.5.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.8,
        sigma_spatial: float = 2.5,
        sigma_intensity: float = 1.5,
        lambda_dl: float = 0.5,
        max_iterations: int = 35,
        name: str = "ROI-Assisted fcHMRF-LIS (Fixed)",
    ) -> None:
        super().__init__(name=name, alpha=alpha)
        self.beta = beta
        self.sigma_spatial = sigma_spatial
        self.sigma_intensity = sigma_intensity
        self.lambda_dl = lambda_dl
        self.max_iterations = max_iterations

    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        params = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            beta=self.beta,
            sigma_spatial=self.sigma_spatial,
            sigma_intensity=self.sigma_intensity,
            lambda_dl=self.lambda_dl,
            alpha=self.alpha,
            max_iterations=self.max_iterations,
        )
        model = FCHMRFModel(params=params)
        res = model.fit_predict(
            z_scores=z_scores,
            dl_prior=dl_prior,
            spatial_mask=spatial_mask,
        )
        meta = {
            "method_mode": res.method_mode,
            "lambda_dl": self.lambda_dl,
            "iterations_to_converge": res.iterations_to_converge,
            "lis_threshold": res.fdr_results["lis_threshold"],
            "estimated_fdr": res.fdr_results["estimated_fdr"],
        }
        return res.significant_mask, res.lis_map, float(res.fdr_results["lis_threshold"]), meta


# ==============================================================================
# Method 5: Adaptive fcHMRF-LIS (Optimized Spatial Parameters, lambda_dl = 0.0)
# ==============================================================================

class AdaptiveFCHMRFLISCorrection(BaseCorrectionMethod):
    """Adaptive fcHMRF-LIS without ROI deep learning prior.

    - Uses spatial hyperparameters (beta, sigma_spatial, sigma_intensity)
      optimized via Bayesian Optimization.
    - lambda_dl is strictly 0.0 (no ROI assistance / spatially homogeneous prior).
    - Scientifically isolates the exact gain of adaptive spatial parameters.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        optimized_parameters: Optional[Dict[str, float]] = None,
        max_iterations: int = 35,
        name: str = "Adaptive fcHMRF-LIS",
    ) -> None:
        super().__init__(name=name, alpha=alpha)
        self.params_dict = optimized_parameters or {
            "beta": 1.1,
            "sigma_spatial": 2.2,
            "sigma_intensity": 1.4,
            "w_pure_spatial": 0.2,
        }
        self.max_iterations = max_iterations

    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        beta = float(self.params_dict.get("beta", 1.1))
        sigma_s = float(self.params_dict.get("sigma_spatial", 2.2))
        sigma_r = float(self.params_dict.get("sigma_intensity", 1.4))
        w_pure = float(self.params_dict.get("w_pure_spatial", 0.2))

        params = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            beta=beta,
            sigma_spatial=sigma_s,
            sigma_intensity=sigma_r,
            w_pure_spatial=w_pure,
            lambda_dl=0.0,
            alpha=self.alpha,
            max_iterations=self.max_iterations,
        )
        model = FCHMRFModel(params=params)
        res = model.fit_predict(
            z_scores=z_scores,
            dl_prior=None,
            spatial_mask=spatial_mask,
        )
        meta = {
            "method_mode": res.method_mode,
            "optimized_parameters": self.params_dict,
            "iterations_to_converge": res.iterations_to_converge,
            "lis_threshold": res.fdr_results["lis_threshold"],
            "estimated_fdr": res.fdr_results["estimated_fdr"],
        }
        return res.significant_mask, res.lis_map, float(res.fdr_results["lis_threshold"]), meta


# ==============================================================================
# Method 6: Proposed Adaptive ROI-Assisted fcHMRF-LIS (ROI + Adaptive fcHMRF-LIS)
# ==============================================================================

class AdaptiveROIAssistedFCHMRFLISCorrection(BaseCorrectionMethod):
    """Proposed Adaptive ROI-Assisted fcHMRF-LIS (ROI + Adaptive fcHMRF-LIS).

    - Uses hyperparameters (beta, sigma_s, sigma_r, lambda_dl) optimized via
      Phase 6 Bayesian Optimization on a dedicated validation partition.
    - Prevents data leakage by evaluating on test volumes using pre-selected parameters.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        optimized_parameters: Optional[Dict[str, float]] = None,
        max_iterations: int = 35,
        name: str = "Adaptive ROI-Assisted fcHMRF-LIS",
    ) -> None:
        super().__init__(name=name, alpha=alpha)
        # Default fallback if parameters not supplied
        self.params_dict = optimized_parameters or {
            "beta": 1.1,
            "sigma_spatial": 2.2,
            "sigma_intensity": 1.4,
            "lambda_dl": 0.65,
        }
        self.max_iterations = max_iterations

    def _execute(
        self,
        z_scores: np.ndarray,
        p_values: np.ndarray,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        beta = float(self.params_dict.get("beta", 1.1))
        sigma_s = float(self.params_dict.get("sigma_spatial", 2.2))
        sigma_r = float(self.params_dict.get("sigma_intensity", 1.4))
        lambda_dl = float(self.params_dict.get("lambda_dl", 0.65))

        params = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            beta=beta,
            sigma_spatial=sigma_s,
            sigma_intensity=sigma_r,
            lambda_dl=lambda_dl,
            alpha=self.alpha,
            max_iterations=self.max_iterations,
        )
        model = FCHMRFModel(params=params)
        res = model.fit_predict(
            z_scores=z_scores,
            dl_prior=dl_prior,
            spatial_mask=spatial_mask,
        )
        meta = {
            "method_mode": res.method_mode,
            "optimized_parameters": self.params_dict,
            "iterations_to_converge": res.iterations_to_converge,
            "lis_threshold": res.fdr_results["lis_threshold"],
            "estimated_fdr": res.fdr_results["estimated_fdr"],
        }
        return res.significant_mask, res.lis_map, float(res.fdr_results["lis_threshold"]), meta
