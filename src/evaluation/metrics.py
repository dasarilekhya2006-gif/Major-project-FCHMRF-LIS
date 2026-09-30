"""Rigorous evaluation metrics for neuroimaging spatial inference and FDR control."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage


@dataclass
class EvaluationMetrics:
    """Comprehensive performance and computational metrics for spatial detection.

    Attributes:
        method_name: Identifier of the evaluated method.
        nominal_alpha: Prespecified significance level (e.g. 0.05).
        total_voxels: Total number of evaluated voxels in domain.
        true_positives: Number of correctly detected active voxels.
        false_positives: Number of falsely declared active voxels.
        true_negatives: Number of correctly identified null voxels.
        false_negatives: Number of missed active voxels.
        tpr: True Positive Rate / Sensitivity / Recall (TP / (TP + FN)).
        fpr: False Positive Rate (FP / (FP + TN)).
        specificity: True Negative Rate (TN / (TN + FP)).
        precision: Positive Predictive Value (TP / (TP + FP)).
        fdp: Empirical False Discovery Proportion (FP / max(1, TP + FP)).
        f1_score: Dice Similarity Coefficient (2TP / (2TP + FP + FN)).
        iou: Jaccard Index / Intersection over Union (TP / (TP + FP + FN)).
        mcc: Matthews Correlation Coefficient.
        fdr_controlled: True if empirical FDP <= nominal_alpha.
        centroid_error_voxels: Euclidean distance between true and detected centroids.
        runtime_seconds: Total wall-clock execution time.
        peak_memory_mb: Peak memory consumption during inference in MB.
    """

    method_name: str
    nominal_alpha: float = 0.05
    total_voxels: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    tpr: float = 0.0
    fpr: float = 0.0
    specificity: float = 0.0
    precision: float = 0.0
    fdp: float = 0.0
    f1_score: float = 0.0
    iou: float = 0.0
    mcc: float = 0.0
    fdr_controlled: bool = False
    centroid_error_voxels: float = 0.0
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def sensitivity(self) -> float:
        """Alias for True Positive Rate / Recall."""
        return self.tpr

    @property
    def recall(self) -> float:
        """Alias for True Positive Rate / Sensitivity."""
        return self.tpr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "nominal_alpha": self.nominal_alpha,
            "total_voxels": self.total_voxels,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "specificity": self.specificity,
            "precision": self.precision,
            "fdp": self.fdp,
            "f1_score": self.f1_score,
            "iou": self.iou,
            "mcc": self.mcc,
            "fdr_controlled": self.fdr_controlled,
            "centroid_error_voxels": self.centroid_error_voxels,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }


def compute_centroid_distance(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> float:
    """Compute Euclidean distance in voxel units between the centers-of-mass.

    Returns:
        Euclidean distance in voxels. Returns 0.0 if both masks are empty, or
        NaN if only one mask is empty (i.e. complete miss or total false alarm).
    """
    n_pred = int(np.sum(pred_mask > 0))
    n_true = int(np.sum(true_mask > 0))

    if n_pred == 0 and n_true == 0:
        return 0.0
    if n_pred == 0 or n_true == 0:
        return float("nan")

    c_pred = np.array(ndimage.center_of_mass(pred_mask.astype(float)))
    c_true = np.array(ndimage.center_of_mass(true_mask.astype(float)))
    return float(np.linalg.norm(c_pred - c_true))


def calculate_metrics(
    pred_mask: np.ndarray,
    ground_truth: np.ndarray,
    method_name: str,
    nominal_alpha: float = 0.05,
    runtime_seconds: float = 0.0,
    peak_memory_mb: float = 0.0,
    spatial_mask: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """Calculate comprehensive classification, FDR, localization, and resource metrics.

    Args:
        pred_mask: Binary boolean or {0, 1} array of declared significant voxels.
        ground_truth: Binary boolean or {0, 1} array of true active voxels.
        method_name: Name of evaluated algorithm.
        nominal_alpha: Prespecified FDR threshold (default: 0.05).
        runtime_seconds: Execution wall-clock time.
        peak_memory_mb: Peak RAM consumption in MB.
        spatial_mask: Optional boolean mask restricting evaluation domain.

    Returns:
        EvaluationMetrics instance.
    """
    if spatial_mask is not None:
        y_pred = (pred_mask[spatial_mask] > 0).astype(bool)
        y_true = (ground_truth[spatial_mask] > 0).astype(bool)
        pred_3d = np.zeros_like(pred_mask, dtype=bool)
        pred_3d[spatial_mask] = y_pred
        true_3d = np.zeros_like(ground_truth, dtype=bool)
        true_3d[spatial_mask] = y_true
    else:
        y_pred = (pred_mask > 0).astype(bool).ravel()
        y_true = (ground_truth > 0).astype(bool).ravel()
        pred_3d = pred_mask > 0
        true_3d = ground_truth > 0

    total_voxels = len(y_pred)
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))

    # Rate calculations with zero-division protection
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fdp = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Localization overlap metrics
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # Matthews Correlation Coefficient
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq > 0:
        mcc = float((tp * tn - fp * fn) / np.sqrt(float(denom_sq)))
    else:
        mcc = 0.0

    # Centroid error
    centroid_err = compute_centroid_distance(pred_3d, true_3d)

    # Check whether nominal FDR is empirically controlled
    fdr_controlled = fdp <= nominal_alpha

    return EvaluationMetrics(
        method_name=method_name,
        nominal_alpha=nominal_alpha,
        total_voxels=total_voxels,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        tpr=tpr,
        fpr=fpr,
        specificity=specificity,
        precision=precision,
        fdp=fdp,
        f1_score=f1,
        iou=iou,
        mcc=mcc,
        fdr_controlled=fdr_controlled,
        centroid_error_voxels=centroid_err if not np.isnan(centroid_err) else 0.0,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=peak_memory_mb,
    )


@dataclass
class AggregatedMetrics:
    """Aggregated statistics across multiple Monte Carlo simulation trials."""

    method_name: str
    num_trials: int
    nominal_alpha: float

    # Mean ± Standard Error
    mean_tpr: float
    sem_tpr: float

    mean_fpr: float
    sem_fpr: float

    mean_specificity: float
    sem_specificity: float

    mean_precision: float
    sem_precision: float

    mean_fdp: float
    sem_fdp: float

    mean_f1: float
    sem_f1: float

    mean_iou: float
    sem_iou: float

    mean_mcc: float
    sem_mcc: float

    mean_centroid_error: float
    sem_centroid_error: float

    mean_runtime_seconds: float
    sem_runtime_seconds: float

    mean_peak_memory_mb: float
    sem_peak_memory_mb: float

    fdr_control_rate: float  # Proportion of trials where FDP <= nominal_alpha

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "num_trials": self.num_trials,
            "nominal_alpha": self.nominal_alpha,
            "tpr": f"{self.mean_tpr:.4f} ± {self.sem_tpr:.4f}",
            "fpr": f"{self.mean_fpr:.4f} ± {self.sem_fpr:.4f}",
            "specificity": f"{self.mean_specificity:.4f} ± {self.sem_specificity:.4f}",
            "precision": f"{self.mean_precision:.4f} ± {self.sem_precision:.4f}",
            "fdp": f"{self.mean_fdp:.4f} ± {self.sem_fdp:.4f}",
            "f1_score": f"{self.mean_f1:.4f} ± {self.sem_f1:.4f}",
            "iou": f"{self.mean_iou:.4f} ± {self.sem_iou:.4f}",
            "mcc": f"{self.mean_mcc:.4f} ± {self.sem_mcc:.4f}",
            "centroid_error": f"{self.mean_centroid_error:.2f} ± {self.sem_centroid_error:.2f}",
            "runtime_s": f"{self.mean_runtime_seconds:.3f} ± {self.sem_runtime_seconds:.3f}",
            "peak_memory_mb": f"{self.mean_peak_memory_mb:.2f} ± {self.sem_peak_memory_mb:.2f}",
            "fdr_control_rate": f"{self.fdr_control_rate * 100.0:.1f}%",
        }


def aggregate_metrics_across_trials(
    trial_metrics: List[EvaluationMetrics],
) -> AggregatedMetrics:
    """Compute mean and standard error of metrics across Monte Carlo repetitions."""
    if not trial_metrics:
        raise ValueError("Cannot aggregate empty list of metrics.")

    n = len(trial_metrics)
    name = trial_metrics[0].method_name
    alpha = trial_metrics[0].nominal_alpha

    def _mean_sem(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=float)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return m, s

    tpr_m, tpr_s = _mean_sem([m.tpr for m in trial_metrics])
    fpr_m, fpr_s = _mean_sem([m.fpr for m in trial_metrics])
    spec_m, spec_s = _mean_sem([m.specificity for m in trial_metrics])
    prec_m, prec_s = _mean_sem([m.precision for m in trial_metrics])
    fdp_m, fdp_s = _mean_sem([m.fdp for m in trial_metrics])
    f1_m, f1_s = _mean_sem([m.f1_score for m in trial_metrics])
    iou_m, iou_s = _mean_sem([m.iou for m in trial_metrics])
    mcc_m, mcc_s = _mean_sem([m.mcc for m in trial_metrics])
    cent_m, cent_s = _mean_sem([m.centroid_error_voxels for m in trial_metrics])
    time_m, time_s = _mean_sem([m.runtime_seconds for m in trial_metrics])
    mem_m, mem_s = _mean_sem([m.peak_memory_mb for m in trial_metrics])

    ctrl_rate = float(np.mean([1.0 if m.fdr_controlled else 0.0 for m in trial_metrics]))

    return AggregatedMetrics(
        method_name=name,
        num_trials=n,
        nominal_alpha=alpha,
        mean_tpr=tpr_m,
        sem_tpr=tpr_s,
        mean_fpr=fpr_m,
        sem_fpr=fpr_s,
        mean_specificity=spec_m,
        sem_specificity=spec_s,
        mean_precision=prec_m,
        sem_precision=prec_s,
        mean_fdp=fdp_m,
        sem_fdp=fdp_s,
        mean_f1=f1_m,
        sem_f1=f1_s,
        mean_iou=iou_m,
        sem_iou=iou_s,
        mean_mcc=mcc_m,
        sem_mcc=mcc_s,
        mean_centroid_error=cent_m,
        sem_centroid_error=cent_s,
        mean_runtime_seconds=time_m,
        sem_runtime_seconds=time_s,
        mean_peak_memory_mb=mem_m,
        sem_peak_memory_mb=mem_s,
        fdr_control_rate=ctrl_rate,
    )
