"""Segmentation loss functions tailored for sparse 3D neuroimaging ROI detection."""

from typing import Optional, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore

from src.core.exceptions import DeepLearningModelError


if TORCH_AVAILABLE:

    class DiceLoss(nn.Module):
        """Soft Dice Loss for volumetric overlap optimization with smooth Laplace regularization."""

        def __init__(self, smooth: float = 1e-5, from_logits: bool = True) -> None:
            super().__init__()
            self.smooth = smooth
            self.from_logits = from_logits

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if self.from_logits:
                probs = torch.sigmoid(inputs)
            else:
                probs = inputs

            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)

            intersection = (probs_flat * targets_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                probs_flat.sum() + targets_flat.sum() + self.smooth
            )
            return 1.0 - dice


    class FocalLoss(nn.Module):
        """Focal Loss to counter extreme null/active class imbalance in neuroimaging.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """

        def __init__(
            self,
            alpha: float = 0.75,
            gamma: float = 2.0,
            from_logits: bool = True,
        ) -> None:
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.from_logits = from_logits

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if self.from_logits:
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
                probs = torch.sigmoid(inputs)
            else:
                bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
                probs = inputs

            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal_weight = (1.0 - p_t) ** self.gamma

            if self.alpha is not None:
                alpha_weight = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
                focal_weight = alpha_weight * focal_weight

            return torch.mean(focal_weight * bce_loss)


    class CombinedBceDiceLoss(nn.Module):
        """Linear combination of binary cross-entropy and soft Dice loss."""

        def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            smooth: float = 1e-5,
            from_logits: bool = True,
        ) -> None:
            super().__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.from_logits = from_logits
            self.dice_fn = DiceLoss(smooth=smooth, from_logits=from_logits)

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if self.from_logits:
                bce = F.binary_cross_entropy_with_logits(inputs, targets)
            else:
                bce = F.binary_cross_entropy(inputs, targets)

            dice = self.dice_fn(inputs, targets)
            return (self.bce_weight * bce) + (self.dice_weight * dice)


    class FocalDiceLoss(nn.Module):
        """Combined Focal and Dice Loss for balanced voxel-wise and region-level learning."""

        def __init__(
            self,
            alpha: float = 0.75,
            gamma: float = 2.0,
            dice_weight: float = 0.5,
            from_logits: bool = True,
        ) -> None:
            super().__init__()
            self.focal = FocalLoss(alpha=alpha, gamma=gamma, from_logits=from_logits)
            self.dice = DiceLoss(from_logits=from_logits)
            self.dice_weight = dice_weight

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return (1.0 - self.dice_weight) * self.focal(inputs, targets) + self.dice_weight * self.dice(inputs, targets)


    def get_segmentation_loss(
        loss_name: str = "focal_dice",
        from_logits: bool = True,
        **kwargs,
    ) -> nn.Module:
        """Factory function to instantiate segmentation loss modules."""
        name = loss_name.lower()
        if name in ("dice", "dice_loss"):
            return DiceLoss(from_logits=from_logits, **kwargs)
        elif name in ("focal", "focal_loss"):
            return FocalLoss(from_logits=from_logits, **kwargs)
        elif name in ("bce_dice", "combined"):
            return CombinedBceDiceLoss(from_logits=from_logits, **kwargs)
        elif name in ("focal_dice", "focaldice"):
            return FocalDiceLoss(from_logits=from_logits, **kwargs)
        elif name in ("bce", "binary_cross_entropy"):
            if from_logits:
                return nn.BCEWithLogitsLoss()
            return nn.BCELoss()
        else:
            raise DeepLearningModelError(
                f"Unknown segmentation loss '{loss_name}'. Choose 'dice', 'focal', 'bce_dice', or 'focal_dice'."
            )

else:

    class DiceLoss:  # type: ignore
        pass

    class FocalLoss:  # type: ignore
        pass

    class CombinedBceDiceLoss:  # type: ignore
        pass

    class FocalDiceLoss:  # type: ignore
        pass

    def get_segmentation_loss(*args, **kwargs):  # type: ignore
        raise DeepLearningModelError("PyTorch is required for segmentation losses.")
