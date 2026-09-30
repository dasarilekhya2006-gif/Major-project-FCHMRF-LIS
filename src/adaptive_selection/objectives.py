"""Objective functions for adaptive hyperparameter optimization in fcHMRF-LIS.

Supports:
1. Supervised Benchmark Objectives: F1-score, MCC, or FDR-penalized Statistical Power on validation data.
2. Unsupervised Variational Objectives: Evidence Lower Bound (ELBO) on held-out validation voxels.
3. Strict Data Leakage Prevention: Evaluates solely on validation partitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.core.logging import get_logger
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult

logger = get_logger(__name__)


class ObjectiveType(str, Enum):
    """Supported objective evaluation metrics."""

    F1_SCORE = "f1_score"
    MCC = "mcc"
    PENALIZED_POWER = "penalized_power"
    VARIATIONAL_ELBO = "variational_elbo"


@dataclass
class ObjectiveResult:
    """Detailed score and diagnostic metrics for an optimization trial.

    Attributes:
        score: Single scalar optimization objective (higher is always better).
        metrics: Dictionary of constituent metrics (power, fdp, f1, mcc, discoveries).
        elapsed_seconds: Wall-clock execution time for this trial.
    """

    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class BaseObjective(ABC):
    """Abstract base class for hyperparameter optimization objectives."""

    @abstractmethod
    def evaluate(self, param_dict: Dict[str, Any]) -> ObjectiveResult:
        """Evaluate a set of hyperparameters and return the objective score."""
        pass


class ValidationFDRBenchmarkObjective(BaseObjective):
    """Supervised objective evaluating fcHMRF-LIS on a dedicated validation split with known ground truth.

    Prevents data leakage by ensuring that hyperparameter search evaluates only on
    the validation set, completely isolated from test sets.
    """

    def __init__(
        self,
        z_scores_val: np.ndarray,
        ground_truth_val: np.ndarray,
        dl_prior_val: Optional[np.ndarray] = None,
        spatial_mask_val: Optional[np.ndarray] = None,
        method_mode: MethodMode = MethodMode.ADAPTIVE_DL_FCHMRF,
        target_alpha: float = 0.05,
        objective_type: ObjectiveType = ObjectiveType.PENALIZED_POWER,
        fdr_penalty_weight: float = 4.0,
        max_iterations: int = 30,
    ) -> None:
        """Initialize the validation benchmark objective.

        Args:
            z_scores_val: 3D or 1D array of validation z-scores.
            ground_truth_val: Binary boolean array of true active voxels on validation partition.
            dl_prior_val: Optional continuous DL prior on validation partition.
            spatial_mask_val: Optional spatial domain mask for validation data.
            method_mode: ORIGINAL_FCHMRF or ADAPTIVE_DL_FCHMRF.
            target_alpha: Nominal FDR target (e.g. 0.05).
            objective_type: PENALIZED_POWER, F1_SCORE, or MCC.
            fdr_penalty_weight: Weight gamma penalizing empirical FDR above target alpha.
            max_iterations: Max mean-field iterations during tuning (capped for speed).
        """
        self.z_val = z_scores_val
        self.gt_val = np.asarray(ground_truth_val, dtype=bool)
        self.dl_prior_val = dl_prior_val
        self.mask_val = spatial_mask_val
        self.method_mode = method_mode
        self.target_alpha = target_alpha
        self.objective_type = objective_type
        self.fdr_penalty_weight = fdr_penalty_weight
        self.max_iterations = max_iterations

        if np.sum(self.gt_val) == 0:
            raise DataValidationError("Validation ground truth active set cannot be empty.")

    def evaluate(self, param_dict: Dict[str, Any]) -> ObjectiveResult:
        """Execute fcHMRF-LIS on validation data and compute objective metric."""
        t_start = time.perf_counter()

        # Build parameters
        params = FCHMRFParameters(
            method_mode=self.method_mode,
            beta=float(param_dict.get("beta", 0.8)),
            sigma_spatial=float(param_dict.get("sigma_spatial", 2.5)),
            sigma_intensity=float(param_dict.get("sigma_intensity", 1.5)),
            w_spatial=float(param_dict.get("w_spatial", 1.0)),
            w_pure_spatial=float(param_dict.get("w_pure_spatial", 0.2)),
            lambda_dl=float(param_dict.get("lambda_dl", 0.5)),
            alpha=self.target_alpha,
            max_iterations=self.max_iterations,
        )

        model = FCHMRFModel(params=params)
        try:
            result = model.fit_predict(
                z_scores=self.z_val,
                dl_prior=self.dl_prior_val,
                spatial_mask=self.mask_val,
            )
        except Exception as e:
            logger.warning(f"Trial failed during inference: {e}")
            return ObjectiveResult(score=-1.0, metrics={"error": 1.0}, elapsed_seconds=time.perf_counter() - t_start)

        elapsed = time.perf_counter() - t_start

        # Compute discovery metrics against validation ground truth
        predicted = result.significant_mask
        actual = self.gt_val

        tp = float(np.sum(predicted & actual))
        fp = float(np.sum(predicted & (~actual)))
        fn = float(np.sum((~predicted) & actual))
        tn = float(np.sum((~predicted) & (~actual)))

        # Power (Sensitivity / True Positive Rate)
        power = tp / max(1.0, tp + fn)
        # False Discovery Proportion (FDP)
        fdp = fp / max(1.0, tp + fp)
        # Precision & Recall
        precision = tp / max(1.0, tp + fp)
        recall = power
        # F1 Score
        f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)

        # Matthews Correlation Coefficient (MCC)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = float((tp * tn - fp * fn) / max(1e-8, denom))

        # Objective calculation
        if self.objective_type == ObjectiveType.F1_SCORE:
            score = float(f1)
        elif self.objective_type == ObjectiveType.MCC:
            score = float(mcc)
        else:
            # Penalized Power: Maximize power while heavily penalizing FDR violation
            excess_fdr = max(0.0, fdp - self.target_alpha)
            score = float(power - self.fdr_penalty_weight * excess_fdr)

        metrics = {
            "power": power,
            "fdp": fdp,
            "f1": f1,
            "mcc": mcc,
            "precision": precision,
            "num_discoveries": float(result.num_discoveries),
            "lis_threshold": float(result.fdr_results["lis_threshold"]),
            "mf_iterations": float(result.model_parameters["mean_field_iterations"]),
        }

        return ObjectiveResult(score=score, metrics=metrics, elapsed_seconds=elapsed)


class VariationalEvidenceObjective(BaseObjective):
    """Unsupervised objective evaluating Evidence Lower Bound (ELBO) on held-out validation voxels.

    Suitable for real experimental datasets where true ground-truth binary masks are unavailable.
    """

    def __init__(
        self,
        z_scores_val: np.ndarray,
        spatial_mask_val: Optional[np.ndarray] = None,
        dl_prior_val: Optional[np.ndarray] = None,
        method_mode: MethodMode = MethodMode.ORIGINAL_FCHMRF,
        target_alpha: float = 0.05,
        max_iterations: int = 25,
    ) -> None:
        self.z_val = z_scores_val
        self.mask_val = spatial_mask_val
        self.dl_prior_val = dl_prior_val
        self.method_mode = method_mode
        self.target_alpha = target_alpha
        self.max_iterations = max_iterations

    def evaluate(self, param_dict: Dict[str, Any]) -> ObjectiveResult:
        """Compute variational free energy / evidence lower bound on validation voxels."""
        t_start = time.perf_counter()

        params = FCHMRFParameters(
            method_mode=self.method_mode,
            beta=float(param_dict.get("beta", 0.8)),
            sigma_spatial=float(param_dict.get("sigma_spatial", 2.5)),
            sigma_intensity=float(param_dict.get("sigma_intensity", 1.5)),
            w_spatial=float(param_dict.get("w_spatial", 1.0)),
            w_pure_spatial=float(param_dict.get("w_pure_spatial", 0.2)),
            lambda_dl=float(param_dict.get("lambda_dl", 0.5)),
            alpha=self.target_alpha,
            max_iterations=self.max_iterations,
        )

        model = FCHMRFModel(params=params)
        result = model.fit_predict(
            z_scores=self.z_val,
            dl_prior=self.dl_prior_val,
            spatial_mask=self.mask_val,
        )
        elapsed = time.perf_counter() - t_start

        # Approximate negative variational free energy on validation domain
        q1 = result.posterior_prob[result.spatial_mask]
        q1 = np.clip(q1, 1e-6, 1.0 - 1e-6)
        q0 = 1.0 - q1

        # Entropy term H(Q) = - sum (q0 log q0 + q1 log q1)
        entropy = -np.sum(q0 * np.log(q0) + q1 * np.log(q1))
        # Expected log-likelihood proxy: mean LIS alignment
        expected_loglik = np.sum(q1 * np.log(np.maximum(1e-6, 1.0 - result.lis_vector)))
        elbo_proxy = float((expected_loglik + entropy) / max(1, len(q1)))

        return ObjectiveResult(
            score=elbo_proxy,
            metrics={"elbo": elbo_proxy, "entropy": float(entropy / len(q1))},
            elapsed_seconds=elapsed,
        )
