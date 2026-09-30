"""Configurable 3D U-Net architecture for neuroimaging Region of Interest (ROI) detection.

Supports configurable depth, channel progression, activation functions, normalization
layers, and dropout for robust 3D volumetric feature representation.
"""

from pathlib import Path
from typing import List, Optional, Union
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

    def _get_activation(act_name: str) -> nn.Module:
        """Instantiate activation layer by name."""
        act_lower = act_name.lower()
        if act_lower == "relu":
            return nn.ReLU(inplace=True)
        elif act_lower in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif act_lower == "elu":
            return nn.ELU(inplace=True)
        elif act_lower == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: '{act_name}'. Choose 'relu', 'leaky_relu', 'elu', or 'gelu'.")

    def _get_norm(norm_type: str, num_channels: int) -> nn.Module:
        """Instantiate normalization layer by type."""
        norm_lower = norm_type.lower()
        if norm_lower in ("batch_norm", "batchnorm", "bn"):
            return nn.BatchNorm3d(num_channels)
        elif norm_lower in ("instance_norm", "instancenorm", "in"):
            return nn.InstanceNorm3d(num_channels, affine=True)
        elif norm_lower in ("group_norm", "groupnorm", "gn"):
            # Ensure number of groups divides num_channels
            num_groups = min(8, num_channels)
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            return nn.GroupNorm(num_groups, num_channels)
        elif norm_lower in ("none", "identity"):
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: '{norm_type}'. Choose 'batch_norm', 'instance_norm', 'group_norm', or 'none'.")

    class ConvBlock3D(nn.Module):
        """Dual 3D convolutional block with normalization, activation, and dropout."""

        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            activation: str = "relu",
            norm_type: str = "batch_norm",
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.norm1 = _get_norm(norm_type, out_ch)
            self.act1 = _get_activation(activation)
            self.drop = nn.Dropout3d(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

            self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.norm2 = _get_norm(norm_type, out_ch)
            self.act2 = _get_activation(activation)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.act1(self.norm1(self.conv1(x)))
            x = self.drop(x)
            x = self.act2(self.norm2(self.conv2(x)))
            return x

    class ConfigurableUNet3D(BaseROIPriorModel):
        """Fully configurable 3D U-Net for neuroimaging volumetric ROI segmentation.

        Attributes:
            in_channels: Number of input channels (e.g. 1 for single contrast or statistical map).
            out_classes: Number of output prediction classes (e.g. 1 for binary ROI mask).
            base_channels: Number of feature maps in the first convolutional level.
            depth: Number of hierarchical downsampling and upsampling levels (e.g. 2, 3, or 4).
            activation: Non-linear activation function ('relu', 'leaky_relu', 'elu', 'gelu').
            norm_type: Normalization type ('batch_norm', 'instance_norm', 'group_norm', 'none').
            dropout_rate: Dropout probability in feature blocks.
            channel_multipliers: Optional explicit filter counts per level (length == depth + 1).
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 1,
            base_channels: int = 16,
            depth: int = 3,
            activation: str = "relu",
            norm_type: str = "batch_norm",
            dropout_rate: float = 0.1,
            channel_multipliers: Optional[List[int]] = None,
            out_channels: Optional[int] = None,
            init_features: Optional[int] = None,
        ) -> None:
            super().__init__()

            if out_channels is not None:
                out_classes = out_channels
            if init_features is not None:
                base_channels = init_features

            if depth < 1 or depth > 5:
                raise ValueError(f"U-Net depth must be between 1 and 5, got {depth}")

            self.in_channels = in_channels
            self.out_classes = out_classes
            self.depth = depth
            self.activation = activation
            self.norm_type = norm_type
            self.dropout_rate = dropout_rate

            # Determine filter counts per depth level
            if channel_multipliers is not None:
                if len(channel_multipliers) != depth + 1:
                    raise ValueError(f"channel_multipliers length must be depth + 1 = {depth + 1}, got {len(channel_multipliers)}")
                self.filters = [int(base_channels * m) for m in channel_multipliers]
            else:
                self.filters = [base_channels * (2 ** i) for i in range(depth + 1)]

            logger.info(
                f"Initializing ConfigurableUNet3D: depth={depth}, filters={self.filters}, "
                f"activation='{activation}', norm='{norm_type}', dropout={dropout_rate}"
            )

            # --- Encoder stages ---
            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()

            curr_in = in_channels
            for i in range(depth):
                curr_out = self.filters[i]
                self.encoders.append(
                    ConvBlock3D(curr_in, curr_out, activation=activation, norm_type=norm_type, dropout_rate=dropout_rate)
                )
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
                curr_in = curr_out

            # --- Bottleneck stage ---
            bottleneck_filters = self.filters[depth]
            self.bottleneck = ConvBlock3D(
                curr_in, bottleneck_filters, activation=activation, norm_type=norm_type, dropout_rate=dropout_rate
            )

            # --- Decoder stages ---
            self.upsamples = nn.ModuleList()
            self.decoders = nn.ModuleList()

            dec_in = bottleneck_filters
            for i in reversed(range(depth)):
                skip_filters = self.filters[i]
                out_filters = skip_filters
                self.upsamples.append(
                    nn.ConvTranspose3d(dec_in, skip_filters, kernel_size=2, stride=2)
                )
                self.decoders.append(
                    ConvBlock3D(
                        skip_filters + skip_filters,
                        out_filters,
                        activation=activation,
                        norm_type=norm_type,
                        dropout_rate=dropout_rate,
                    )
                )
                dec_in = out_filters

            # --- Final output projection ---
            self.final_conv = nn.Conv3d(self.filters[0], out_classes, kernel_size=1)

        @property
        def required_divisibility(self) -> int:
            """Spatial dimension divisibility required by the architecture (2^depth)."""
            return 2 ** self.depth

        def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
            """Forward pass through 3D U-Net.

            Args:
                x: Input tensor of shape (B, C, D, H, W).
                return_logits: If True, returns raw un-activated logits (for numerical loss stability).
                               If False, returns sigmoid probabilities for binary or softmax for multi-class.

            Returns:
                torch.Tensor of shape (B, out_classes, D, H, W).
            """
            # Handle padding if spatial dimensions are not divisible by 2^depth
            div = self.required_divisibility
            _, _, d, h, w = x.shape
            pad_d = (div - d % div) % div
            pad_h = (div - h % div) % div
            pad_w = (div - w % div) % div

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                # pad format for 5D tensor: (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back)
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

            # Encoder pass
            skips = []
            curr = x
            for enc, pool in zip(self.encoders, self.pools):
                curr = enc(curr)
                skips.append(curr)
                curr = pool(curr)

            # Bottleneck
            curr = self.bottleneck(curr)

            # Decoder pass with skip connections
            for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
                curr = up(curr)
                # Handle potential spatial mismatch from odd dimensions
                if curr.shape[2:] != skip.shape[2:]:
                    curr = F.interpolate(curr, size=skip.shape[2:], mode="trilinear", align_corners=False)
                curr = torch.cat([curr, skip], dim=1)
                curr = dec(curr)

            logits = self.final_conv(curr)

            # Crop back to original input dimensions
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                logits = logits[:, :, :d, :h, :w]

            if return_logits:
                return logits

            if self.out_classes == 1:
                return torch.sigmoid(logits)
            else:
                return torch.softmax(logits, dim=1)

        def predict_prior_map(self, volume: np.ndarray) -> np.ndarray:
            """Inference helper taking a 3D numpy volume and returning a 3D probability map."""
            self.eval()
            device = next(self.parameters()).device
            with torch.no_grad():
                tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)
                probs = self.forward(tensor, return_logits=False)
                return probs.squeeze().cpu().numpy()

        def load_weights(self, path: Union[str, Path]) -> None:
            """Load checkpoint weights into model."""
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.load_state_dict(state)

else:

    class ConfigurableUNet3D(BaseROIPriorModel):  # type: ignore
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def forward(self, x: object, return_logits: bool = False) -> object:
            raise DeepLearningModelError("PyTorch is required to run ConfigurableUNet3D.")

        def predict_prior_map(self, volume: np.ndarray) -> np.ndarray:
            raise DeepLearningModelError("PyTorch is required to run ConfigurableUNet3D.")
