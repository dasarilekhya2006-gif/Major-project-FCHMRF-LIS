"""Segmentation evaluation metrics: Dice score, IoU, Precision, and Recall."""

from typing import Dict, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_segmentation_metrics(
    predictions: Union[np.ndarray, "torch.Tensor"],
    targets: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute volumetric segmentation metrics: Dice, IoU (Jaccard), Precision, and Recall.

    Args:
        predictions: Predicted probabilities or binary masks (Tensor or ndarray).
        targets: Ground-truth binary labels (Tensor or ndarray).
        threshold: Decision threshold for binarizing continuous probabilities.
        eps: Epsilon to prevent division by zero.

    Returns:
        Dict[str, float]: Dictionary with 'dice', 'iou', 'precision', 'recall'.
    """
    if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
        preds_arr = predictions.detach().cpu().numpy()
    else:
        preds_arr = np.asarray(predictions)

    if TORCH_AVAILABLE and isinstance(targets, torch.Tensor):
        targets_arr = targets.detach().cpu().numpy()
    else:
        targets_arr = np.asarray(targets)

    # Binarize
    binary_preds = (preds_arr >= threshold).astype(bool).ravel()
    binary_targets = (targets_arr > 0.5).astype(bool).ravel()

    # True positives, false positives, false negatives
    tp = float(np.sum(binary_preds & binary_targets))
    fp = float(np.sum(binary_preds & ~binary_targets))
    fn = float(np.sum(~binary_preds & binary_targets))
    tn = float(np.sum(~binary_preds & ~binary_targets))

    # Corner case: Both masks are completely empty (perfect null agreement)
    if tp == 0 and fp == 0 and fn == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        }

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(np.clip(dice, 0.0, 1.0)),
        "iou": float(np.clip(iou, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
    }


class MetricTracker:
    """Accumulates and computes running averages of loss and segmentation metrics across batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.total_precision = 0.0
        self.total_recall = 0.0
        self.num_batches = 0

    def update(
        self,
        loss: float,
        predictions: Union[np.ndarray, "torch.Tensor"],
        targets: Union[np.ndarray, "torch.Tensor"],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Update running metrics with a new batch."""
        metrics = compute_segmentation_metrics(predictions, targets, threshold=threshold)
        self.total_loss += float(loss)
        self.total_dice += metrics["dice"]
        self.total_iou += metrics["iou"]
        self.total_precision += metrics["precision"]
        self.total_recall += metrics["recall"]
        self.num_batches += 1
        return metrics

    def compute_averages(self) -> Dict[str, float]:
        """Return average metrics across all accumulated batches."""
        n = max(self.num_batches, 1)
        return {
            "loss": float(self.total_loss / n),
            "dice": float(self.total_dice / n),
            "iou": float(self.total_iou / n),
            "precision": float(self.total_precision / n),
            "recall": float(self.total_recall / n),
        }
