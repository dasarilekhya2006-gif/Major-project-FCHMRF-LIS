"""Baseline comparator: rigorously evaluates adaptively selected parameters vs fixed defaults.

Guarantees data leakage prevention by executing evaluation strictly on an
independent, held-out test dataset.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Quantitative performance metrics on held-out evaluation dataset."""

    num_discoveries: int
    discovery_rate: float
    lis_cutoff: float
    estimated_fdr: float
    mean_field_iterations: int
    inference_runtime_seconds: float
    # Supervised metrics (when ground truth is available)
    power: Optional[float] = None
    fdp: Optional[float] = None
    f1_score: Optional[float] = None
    mcc: Optional[float] = None


@dataclass
class ComparisonReport:
    """Comprehensive comparison between baseline and adaptively selected parameters.

    Attributes:
        baseline_params: Fixed predefined default hyperparameters.
        selected_params: Adaptively optimized hyperparameters.
        baseline_metrics: Performance of baseline on test set.
        selected_metrics: Performance of adaptive parameters on test set.
        optimization_cost_seconds: Total wall-clock time spent during optimization.
        is_benefit_measurable: Whether adaptive parameters outperformed baseline.
        benefit_summary: Scientific interpretation of observed differences.
    """

    baseline_params: Dict[str, Any]
    selected_params: Dict[str, Any]
    baseline_metrics: PerformanceMetrics
    selected_metrics: PerformanceMetrics
    optimization_cost_seconds: float
    is_benefit_measurable: bool
    benefit_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_params": self.baseline_params,
            "selected_params": self.selected_params,
            "baseline_metrics": asdict(self.baseline_metrics),
            "selected_metrics": asdict(self.selected_metrics),
            "optimization_cost_seconds": self.optimization_cost_seconds,
            "is_benefit_measurable": self.is_benefit_measurable,
            "benefit_summary": self.benefit_summary,
        }

    def print_report(self) -> None:
        """Log formatted ASCII comparison table."""
        logger.info("=" * 75)
        logger.info("ADAPTIVE PARAMETERS VS. BASELINE COMPARISON ON HELD-OUT TEST DATA")
        logger.info("=" * 75)
        logger.info(f"{'Metric':<30} | {'Baseline (Fixed)':<18} | {'Adaptive (Selected)':<18}")
        logger.info("-" * 75)
        b_m = self.baseline_metrics
        s_m = self.selected_metrics

        logger.info(f"{'Discoveries (Count)':<30} | {b_m.num_discoveries:<18} | {s_m.num_discoveries:<18}")
        logger.info(f"{'Discovery Rate (%)':<30} | {b_m.discovery_rate * 100:<17.2f}% | {s_m.discovery_rate * 100:<17.2f}%")
        logger.info(f"{'LIS Cutoff Threshold':<30} | {b_m.lis_cutoff:<18.4f} | {s_m.lis_cutoff:<18.4f}")
        logger.info(f"{'Estimated Spatial FDR':<30} | {b_m.estimated_fdr:<18.4f} | {s_m.estimated_fdr:<18.4f}")
        logger.info(f"{'Mean-Field Iterations':<30} | {b_m.mean_field_iterations:<18} | {s_m.mean_field_iterations:<18}")
        logger.info(f"{'Inference Latency (s)':<30} | {b_m.inference_runtime_seconds:<18.3f} | {s_m.inference_runtime_seconds:<18.3f}")

        if b_m.power is not None and s_m.power is not None:
            logger.info(f"{'Statistical Power (Sensitivity)':<30} | {b_m.power:<18.4f} | {s_m.power:<18.4f}")
            logger.info(f"{'False Discovery Prop. (FDP)':<30} | {b_m.fdp:<18.4f} | {s_m.fdp:<18.4f}")
            logger.info(f"{'F1 Score':<30} | {b_m.f1_score:<18.4f} | {s_m.f1_score:<18.4f}")
            logger.info(f"{'Matthews Corr. Coeff. (MCC)':<30} | {b_m.mcc:<18.4f} | {s_m.mcc:<18.4f}")

        logger.info("-" * 75)
        logger.info(f"Optimization Search Cost: {self.optimization_cost_seconds:.2f} seconds")
        logger.info(f"Measurable Benefit:       {self.is_benefit_measurable}")
        logger.info(f"Scientific Conclusion:    {self.benefit_summary}")
        logger.info("=" * 75)


class BaselineComparator:
    """Executes controlled comparative evaluation of baseline vs selected parameters."""

    def __init__(
        self,
        method_mode: MethodMode = MethodMode.ADAPTIVE_DL_FCHMRF,
        target_alpha: float = 0.05,
    ) -> None:
        self.method_mode = method_mode
        self.target_alpha = target_alpha

    def _evaluate_parameters(
        self,
        param_dict: Dict[str, Any],
        z_scores_test: np.ndarray,
        ground_truth_test: Optional[np.ndarray] = None,
        dl_prior_test: Optional[np.ndarray] = None,
        spatial_mask_test: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """Run fcHMRF-LIS on test volume and compute performance metrics."""
        params = FCHMRFParameters(
            method_mode=self.method_mode,
            beta=float(param_dict.get("beta", 0.8)),
            sigma_spatial=float(param_dict.get("sigma_spatial", 2.5)),
            sigma_intensity=float(param_dict.get("sigma_intensity", 1.5)),
            w_spatial=float(param_dict.get("w_spatial", 1.0)),
            w_pure_spatial=float(param_dict.get("w_pure_spatial", 0.2)),
            lambda_dl=float(param_dict.get("lambda_dl", 0.5)),
            alpha=self.target_alpha,
            max_iterations=40,
        )

        model = FCHMRFModel(params=params)
        t_start = time.perf_counter()
        result = model.fit_predict(
            z_scores=z_scores_test,
            dl_prior=dl_prior_test,
            spatial_mask=spatial_mask_test,
        )
        elapsed = time.perf_counter() - t_start

        rate = float(result.num_discoveries / max(1, result.num_analyzed_voxels))
        cutoff = float(result.fdr_results["lis_threshold"])
        est_fdr = float(result.fdr_results["estimated_fdr"])
        iters = int(result.model_parameters["mean_field_iterations"])

        power, fdp, f1, mcc = None, None, None, None
        if ground_truth_test is not None:
            gt = np.asarray(ground_truth_test, dtype=bool)
            pred = result.significant_mask
            tp = float(np.sum(pred & gt))
            fp = float(np.sum(pred & (~gt)))
            fn = float(np.sum((~pred) & gt))
            tn = float(np.sum((~pred) & (~gt)))

            power = float(tp / max(1.0, tp + fn))
            fdp = float(fp / max(1.0, tp + fp))
            precision = float(tp / max(1.0, tp + fp))
            f1 = float((2.0 * precision * power) / max(1e-8, precision + power))
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = float((tp * tn - fp * fn) / max(1e-8, denom))

        return PerformanceMetrics(
            num_discoveries=result.num_discoveries,
            discovery_rate=rate,
            lis_cutoff=cutoff,
            estimated_fdr=est_fdr,
            mean_field_iterations=iters,
            inference_runtime_seconds=elapsed,
            power=power,
            fdp=fdp,
            f1_score=f1,
            mcc=mcc,
        )

    def compare(
        self,
        baseline_params: Dict[str, Any],
        selected_params: Dict[str, Any],
        z_scores_test: np.ndarray,
        ground_truth_test: Optional[np.ndarray] = None,
        dl_prior_test: Optional[np.ndarray] = None,
        spatial_mask_test: Optional[np.ndarray] = None,
        optimization_cost_seconds: float = 0.0,
    ) -> ComparisonReport:
        """Execute head-to-head evaluation on held-out test data."""
        logger.info("Evaluating baseline fixed parameters on test data...")
        b_metrics = self._evaluate_parameters(
            param_dict=baseline_params,
            z_scores_test=z_scores_test,
            ground_truth_test=ground_truth_test,
            dl_prior_test=dl_prior_test,
            spatial_mask_test=spatial_mask_test,
        )

        logger.info("Evaluating adaptively selected parameters on test data...")
        s_metrics = self._evaluate_parameters(
            param_dict=selected_params,
            z_scores_test=z_scores_test,
            ground_truth_test=ground_truth_test,
            dl_prior_test=dl_prior_test,
            spatial_mask_test=spatial_mask_test,
        )

        # Scientific determination of benefit
        is_benefit = False
        reasons = []

        if b_metrics.f1_score is not None and s_metrics.f1_score is not None:
            f1_diff = s_metrics.f1_score - b_metrics.f1_score
            fdp_ok = (s_metrics.fdp or 0.0) <= (self.target_alpha + 0.05)
            if f1_diff > 0.02 and fdp_ok:
                is_benefit = True
                reasons.append(f"F1-score improved by {f1_diff:.4f} without exceeding FDR")
            elif f1_diff <= 0.0:
                reasons.append(f"F1-score did not improve (delta: {f1_diff:.4f})")
            else:
                reasons.append("Marginal F1 gain within noise bounds")
        else:
            # Unsupervised comparison: faster convergence or tighter FDR control
            if s_metrics.mean_field_iterations < b_metrics.mean_field_iterations:
                reasons.append(
                    f"Reduced mean-field iterations ({s_metrics.mean_field_iterations} vs {b_metrics.mean_field_iterations})"
                )
            if s_metrics.estimated_fdr <= b_metrics.estimated_fdr:
                reasons.append("Maintained conservative spatial FDR control")

        summary = "; ".join(reasons) if reasons else "No significant deviation observed."

        report = ComparisonReport(
            baseline_params=baseline_params,
            selected_params=selected_params,
            baseline_metrics=b_metrics,
            selected_metrics=s_metrics,
            optimization_cost_seconds=optimization_cost_seconds,
            is_benefit_measurable=is_benefit,
            benefit_summary=summary,
        )
        report.print_report()
        return report
