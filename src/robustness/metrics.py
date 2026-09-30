"""Stability and uncertainty metrics for evaluating neuroimaging spatial inference."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class StabilityMetrics:
    """Quantitative stability and uncertainty metrics across repeated runs or perturbations.

    Attributes:
        condition_name: Name of evaluated stress condition or sweep level.
        condition_value: Numerical value (e.g. SNR, noise sigma, perturbation scale).
        num_repetitions: Number of repeated Monte Carlo trials.
        mean_pairwise_jaccard: Mean pairwise Jaccard index between declared discovery masks.
        std_pairwise_jaccard: Standard deviation of pairwise Jaccard index.
        mean_pairwise_dice: Mean pairwise Dice similarity coefficient.
        mean_discovery_count: Mean number of declared significant voxels.
        std_discovery_count: Standard deviation of discovery count.
        discovery_count_cv: Coefficient of variation (std / mean) of discovery count.
        mean_boundary_entropy: Mean voxel-wise Shannon entropy of detection probabilities.
        mean_tpr: Average True Positive Rate (if ground truth available).
        mean_fdp: Average False Discovery Proportion (if ground truth available).
        fdr_control_rate: Proportion of runs where FDP <= nominal alpha.
    """

    condition_name: str
    condition_value: float
    num_repetitions: int
    mean_pairwise_jaccard: float
    std_pairwise_jaccard: float
    mean_pairwise_dice: float
    mean_discovery_count: float
    std_discovery_count: float
    discovery_count_cv: float
    mean_boundary_entropy: float
    mean_tpr: float = 0.0
    mean_fdp: float = 0.0
    fdr_control_rate: float = 0.0

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "condition_name": self.condition_name,
            "condition_value": self.condition_value,
            "num_repetitions": self.num_repetitions,
            "mean_pairwise_jaccard": self.mean_pairwise_jaccard,
            "std_pairwise_jaccard": self.std_pairwise_jaccard,
            "mean_pairwise_dice": self.mean_pairwise_dice,
            "mean_discovery_count": self.mean_discovery_count,
            "std_discovery_count": self.std_discovery_count,
            "discovery_count_cv": self.discovery_count_cv,
            "mean_boundary_entropy": self.mean_boundary_entropy,
            "mean_tpr": self.mean_tpr,
            "mean_fdp": self.mean_fdp,
            "fdr_control_rate": self.fdr_control_rate,
        }


def compute_pairwise_jaccard_and_dice(
    masks: List[np.ndarray],
) -> Tuple[float, float, float]:
    """Calculate mean pairwise Jaccard index, std, and mean pairwise Dice coefficient.

    For M repeated runs with binary discovery sets D_1, ..., D_M:
        J(D_j, D_k) = |D_j ∩ D_k| / |D_j ∪ D_k|
        Dice(D_j, D_k) = 2 |D_j ∩ D_k| / (|D_j| + |D_k|)

    Returns:
        (mean_jaccard, std_jaccard, mean_dice)
    """
    m = len(masks)
    if m < 2:
        return 1.0, 0.0, 1.0

    jaccards = []
    dices = []

    for i in range(m):
        m_i = masks[i] > 0
        for j in range(i + 1, m):
            m_j = masks[j] > 0
            intersection = np.sum(m_i & m_j)
            union = np.sum(m_i | m_j)

            if union == 0:
                # Both empty: perfectly stable absence of activations
                jaccards.append(1.0)
                dices.append(1.0)
            else:
                j_idx = float(intersection / union)
                d_idx = float(2 * intersection / (np.sum(m_i) + np.sum(m_j)))
                jaccards.append(j_idx)
                dices.append(d_idx)

    return (
        float(np.mean(jaccards)),
        float(np.std(jaccards)),
        float(np.mean(dices)),
    )


def compute_detection_probability_map(
    masks: List[np.ndarray],
) -> np.ndarray:
    """Compute empirical voxel-wise detection probability across repetitions:

    P_detect(v) = (1 / M) * sum_{m=1}^M I(v in D_m)
    """
    if not masks:
        raise ValueError("Masks list cannot be empty.")
    stacked = np.stack([m.astype(float) for m in masks], axis=0)
    return np.mean(stacked, axis=0)


def compute_voxel_uncertainty_entropy(
    detection_prob_map: np.ndarray,
    spatial_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Compute Shannon entropy of the detection probability map in bits:

    H(v) = - p * log2(p) - (1 - p) * log2(1 - p)

    H(v) = 0 bits when detection is completely certain (p=0 or p=1).
    H(v) = 1 bit (maximum uncertainty) when p = 0.5 (activation boundary).

    Returns:
        (entropy_map, mean_boundary_entropy)
    """
    p = np.clip(detection_prob_map, 1e-12, 1.0 - 1e-12)
    entropy_map = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    if spatial_mask is not None:
        valid = spatial_mask > 0
        # Only evaluate entropy in regions with non-trivial variation (0 < p < 1)
        active_zone = valid & (detection_prob_map > 0.01) & (detection_prob_map < 0.99)
        if np.any(active_zone):
            mean_entropy = float(np.mean(entropy_map[active_zone]))
        else:
            mean_entropy = 0.0
    else:
        active_zone = (detection_prob_map > 0.01) & (detection_prob_map < 0.99)
        mean_entropy = float(np.mean(entropy_map[active_zone])) if np.any(active_zone) else 0.0

    return entropy_map.astype(np.float32), mean_entropy


def evaluate_repetition_stability(
    masks: List[np.ndarray],
    condition_name: str,
    condition_value: float,
    ground_truth: Optional[np.ndarray] = None,
    nominal_alpha: float = 0.05,
    spatial_mask: Optional[np.ndarray] = None,
) -> StabilityMetrics:
    """Evaluate full stability profile across repeated runs for a given condition."""
    m_reps = len(masks)
    mean_j, std_j, mean_d = compute_pairwise_jaccard_and_dice(masks)

    counts = [int(np.sum(m > 0)) for m in masks]
    mean_c = float(np.mean(counts))
    std_c = float(np.std(counts))
    cv_c = float(std_c / mean_c) if mean_c > 0 else 0.0

    prob_map = compute_detection_probability_map(masks)
    _, mean_ent = compute_voxel_uncertainty_entropy(prob_map, spatial_mask=spatial_mask)

    mean_tpr = 0.0
    mean_fdp = 0.0
    ctrl_rate = 0.0

    if ground_truth is not None:
        gt_bool = ground_truth > 0
        tprs = []
        fdps = []
        ctrls = []
        num_gt = int(np.sum(gt_bool))

        for m in masks:
            m_bool = m > 0
            tp = int(np.sum(m_bool & gt_bool))
            fp = int(np.sum(m_bool & ~gt_bool))

            tpr = tp / num_gt if num_gt > 0 else 0.0
            fdp = fp / max(1, tp + fp)

            tprs.append(tpr)
            fdps.append(fdp)
            ctrls.append(1.0 if fdp <= nominal_alpha else 0.0)

        mean_tpr = float(np.mean(tprs))
        mean_fdp = float(np.mean(fdps))
        ctrl_rate = float(np.mean(ctrls))

    return StabilityMetrics(
        condition_name=condition_name,
        condition_value=condition_value,
        num_repetitions=m_reps,
        mean_pairwise_jaccard=mean_j,
        std_pairwise_jaccard=std_j,
        mean_pairwise_dice=mean_d,
        mean_discovery_count=mean_c,
        std_discovery_count=std_c,
        discovery_count_cv=cv_c,
        mean_boundary_entropy=mean_ent,
        mean_tpr=mean_tpr,
        mean_fdp=mean_fdp,
        fdr_control_rate=ctrl_rate,
    )
