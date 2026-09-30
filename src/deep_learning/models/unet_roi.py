"""Lightweight 3D U-Net architecture for voxel-wise ROI prior detection.

Avoids overly complex transformers or GNNs; focuses on fast, reproducible local
context aggregation to produce spatial prior probabilities pi_i in [0, 1].
"""

from typing import List, Optional
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
from src.core.logging import get_logger
from src.deep_learning.models.base_model import BaseROIPriorModel

logger = get_logger(__name__)


if TORCH_AVAILABLE:

    class ConvBlock3D(nn.Module):
        """Two consecutive 3D Conv-BatchNorm-ReLU layers."""

        def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)


    class LightweightUNet3D(BaseROIPriorModel):
        """A lightweight, parameter-efficient 3D U-Net for spatial neuroimaging prior estimation.

        Input: Tensor of shape (B, 1, D, H, W)
        Output: Tensor of prior probabilities P(theta_i = 1) in [0, 1] of shape (B, 1, D, H, W).
        """

        def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 16,
            dropout_rate: float = 0.1,
        ) -> None:
            super().__init__()
            self.enc1 = ConvBlock3D(in_channels, base_channels, dropout=dropout_rate)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

            self.enc2 = ConvBlock3D(base_channels, base_channels * 2, dropout=dropout_rate)
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

            self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4, dropout=dropout_rate)

            self.up2 = nn.ConvTranspose3d(
                base_channels * 4, base_channels * 2, kernel_size=2, stride=2
            )
            self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2, dropout=dropout_rate)

            self.up1 = nn.ConvTranspose3d(
                base_channels * 2, base_channels, kernel_size=2, stride=2
            )
            self.dec1 = ConvBlock3D(base_channels * 2, base_channels, dropout=dropout_rate)

            self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

            logger.debug(
                f"Initialized LightweightUNet3D with in_ch={in_channels}, base_ch={base_channels}"
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pad dimensions if not divisible by 4
            d, h, w = x.shape[2], x.shape[3], x.shape[4]
            pad_d = (4 - d % 4) % 4
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")

            e1 = self.enc1(x)
            p1 = self.pool1(e1)

            e2 = self.enc2(p1)
            p2 = self.pool2(e2)

            b = self.bottleneck(p2)

            d2 = self.up2(b)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            logits = self.out_conv(d1)
            probs = self.sigmoid(logits)

            # Crop back to original dimensions
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                probs = probs[:, :, :d, :h, :w]

            return probs

        def predict_prior_map(self, volume: np.ndarray) -> np.ndarray:
            """Inference helper taking a 3D volume and returning a 3D prior map."""
            self.eval()
            device = next(self.parameters()).device
            with torch.no_grad():
                tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)
                prior = self.forward(tensor)
                return prior.squeeze().cpu().numpy()

else:

    class LightweightUNet3D(BaseROIPriorModel):  # type: ignore
        """Dummy placeholder when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def forward(self, x: object) -> object:
            raise DeepLearningModelError("PyTorch is required to run LightweightUNet3D.")

        def predict_prior_map(self, volume: np.ndarray) -> np.ndarray:
            raise DeepLearningModelError("PyTorch is required to run LightweightUNet3D.")
