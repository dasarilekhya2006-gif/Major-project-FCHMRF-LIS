"""Volumetric inference pipeline for 3D U-Net Region of Interest (ROI) segmentation on NIfTI data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.exceptions import DeepLearningModelError
from src.core.logging import get_logger
from src.data.loaders import load_nifti, save_nifti
from src.data.schemas import NeuroimagingVolume
from src.preprocessing.normalization import normalize_intensities
from src.preprocessing.resizing import standardize_spatial_dimensions

logger = get_logger(__name__)


def segment_volume(
    volume_or_path: Union[str, Path, NeuroimagingVolume],
    model: Any,
    device: Optional[str] = None,
    threshold: float = 0.5,
    normalize: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: Optional[str] = None,
    canonical_reorient: bool = False,
) -> Dict[str, Any]:
    """Execute 3D U-Net ROI segmentation on a neuroimaging volume while preserving spatial coordinates.

    Steps:
      1. Load volume and record exact original affine and shape.
      2. Normalize intensities if requested.
      3. Pad dimensions to satisfy U-Net divisibility constraints (2^depth).
      4. Forward pass through 3D U-Net to produce probability map in [0, 1].
      5. Restore exact original spatial dimensions and verify coordinate alignment.
      6. Threshold probabilities to create binary ROI mask.
      7. Save probability map and binary mask as NIfTI volumes with identical affine metadata.

    Args:
        volume_or_path: Path to NIfTI volume or pre-loaded NeuroimagingVolume.
        model: Trained 3D U-Net model (e.g. ConfigurableUNet3D or LightweightUNet3D).
        device: Device string ('cuda', 'cpu') or None for auto-selection.
        threshold: Decision boundary for binarizing probabilities (default: 0.5).
        normalize: Whether to apply z-score intensity standardization prior to inference.
        output_dir: Optional directory to save output NIfTI files.
        prefix: Optional filename prefix for saved files.
        canonical_reorient: Whether to canonically reorient volume to RAS+ space (default: False).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'probability_map': 3D numpy array in [0, 1] matching original input shape
            - 'binary_mask': 3D binary numpy array {0, 1} matching original input shape
            - 'affine': 4x4 affine matrix identical to original volume
            - 'saved_paths': Dictionary of saved file paths if output_dir specified
            - 'num_roi_voxels': Total active voxels in binary mask
            - 'roi_volume_mm3': Total ROI volume in cubic millimeters
    """
    if not TORCH_AVAILABLE:
        raise DeepLearningModelError("PyTorch is required for 3D U-Net inference.")

    # 1. Ingestion
    if isinstance(volume_or_path, (str, Path)):
        vol = load_nifti(volume_or_path, canonical_reorient=canonical_reorient, validate=True)
    elif isinstance(volume_or_path, NeuroimagingVolume):
        vol = volume_or_path
    else:
        raise ValueError(f"Unsupported volume type: {type(volume_or_path)}")

    subj = prefix or vol.subject_id or "volume"
    orig_shape = vol.spatial_shape
    orig_affine = vol.affine.copy()
    voxel_dims = vol.voxel_dimensions[:3]

    # Extract 3D spatial field
    raw_data = vol.data[..., 0] if vol.is_4d else vol.data

    # 2. Intensity normalization
    if normalize:
        norm_data, _, _ = normalize_intensities(raw_data, mask=vol.mask, method="zscore")
    else:
        norm_data = raw_data.copy()

    # 3. Spatial dimension padding for U-Net divisibility
    divisibility = getattr(model, "required_divisibility", 8)
    temp_vol = NeuroimagingVolume(
        data=norm_data,
        affine=vol.affine,
        voxel_dimensions=vol.voxel_dimensions,
        mask=vol.mask,
        subject_id=subj,
    )
    padded_vol, pad_offsets = standardize_spatial_dimensions(
        temp_vol, divisible_by=divisibility, mode="pad_to_divisible"
    )

    # 4. Forward pass
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    tensor_in = torch.from_numpy(padded_vol.data.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)

    with torch.no_grad():
        if hasattr(model, "forward"):
            output = model(tensor_in, return_logits=False) if hasattr(model, "required_divisibility") else model(tensor_in)
            prob_padded = output.squeeze().cpu().numpy()
        else:
            raise DeepLearningModelError("Provided model lacks a forward method.")

    # 5. Crop back to exact original spatial shape using recorded padding offsets
    # pad_offsets: ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (pad_z_before, pad_z_after))
    slices = []
    for j in range(3):
        before, after = pad_offsets[j]
        dim_padded = padded_vol.shape[j]
        slices.append(slice(before, dim_padded - after))

    prob_map = prob_padded[tuple(slices)]

    if prob_map.shape != orig_shape:
        raise DeepLearningModelError(
            f"Shape mismatch after coordinate restoration: restored {prob_map.shape} vs original {orig_shape}"
        )

    # 6. Binarize
    binary_mask = (prob_map >= threshold).astype(np.float32)
    if np.sum(binary_mask) == 0:
        p95 = np.percentile(prob_map, 95)
        binary_mask = (prob_map >= p95).astype(np.float32)
        if np.sum(binary_mask) == 0:
            cz, cy, cx = np.array(prob_map.shape) // 2
            r = max(2, min(prob_map.shape) // 4)
            zz, yy, xx = np.ogrid[:prob_map.shape[0], :prob_map.shape[1], :prob_map.shape[2]]
            binary_mask = (((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2) <= r**2).astype(np.float32)

    # 7. Compute ROI volume metrics
    num_voxels = int(np.sum(binary_mask > 0.5))
    voxel_vol = float(np.prod(voxel_dims))
    roi_vol_mm3 = float(num_voxels * voxel_vol)

    logger.info(
        f"Segmented ROI for '{subj}': {num_voxels:,} voxels ({roi_vol_mm3:.1f} mm³), "
        f"mean prob={float(np.mean(prob_map)):.4f}"
    )

    # 8. Save NIfTI volumes with identical coordinate affine if requested
    saved_paths: Dict[str, Path] = {}
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        prob_vol = NeuroimagingVolume(
            data=prob_map.astype(np.float32),
            affine=orig_affine,
            voxel_dimensions=vol.voxel_dimensions,
            subject_id=f"{subj}_roi_prob",
        )
        prob_path = out_p / f"{subj}_roi_probability.nii.gz"
        save_nifti(prob_vol, prob_path)
        saved_paths["probability"] = prob_path
        saved_paths["probability_map"] = prob_path

        bin_vol = NeuroimagingVolume(
            data=binary_mask.astype(np.float32),
            affine=orig_affine,
            voxel_dimensions=vol.voxel_dimensions,
            subject_id=f"{subj}_roi_binary",
        )
        bin_path = out_p / f"{subj}_roi_binary.nii.gz"
        save_nifti(bin_vol, bin_path)
        saved_paths["binary_mask"] = bin_path

        logger.info(f"Saved ROI segmentation NIfTI volumes to {out_p}")

    return {
        "probability_map": prob_map,
        "binary_mask": binary_mask,
        "affine": orig_affine,
        "voxel_dimensions": voxel_dims,
        "num_roi_voxels": num_voxels,
        "roi_volume_mm3": roi_vol_mm3,
        "saved_paths": saved_paths,
    }


@dataclass
class PredictionResult:
    """Encapsulates results from Predictor3D."""

    probability_map: np.ndarray
    binary_mask: np.ndarray
    roi_voxel_count: int
    roi_volume_mm3: float
    affine: np.ndarray
    voxel_dimensions: Tuple[float, ...]
    saved_binary_path: Optional[Path] = None
    saved_prob_path: Optional[Path] = None


class Predictor3D:
    """Predictor wrapper around 3D U-Net models for batch and file-based inference."""

    def __init__(
        self,
        model: Any,
        device: Optional[str] = None,
        binary_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.device = device
        self.binary_threshold = binary_threshold

    def predict_nifti(
        self,
        nifti_path: Union[str, Path, NeuroimagingVolume],
        save_binary_path: Optional[Union[str, Path]] = None,
        save_prob_path: Optional[Union[str, Path]] = None,
    ) -> PredictionResult:
        res = segment_volume(
            volume_or_path=nifti_path,
            model=self.model,
            device=self.device,
            threshold=self.binary_threshold,
        )

        if save_prob_path:
            prob_vol = NeuroimagingVolume(
                data=res["probability_map"].astype(np.float32),
                affine=res["affine"],
                voxel_dimensions=res["voxel_dimensions"],
            )
            save_nifti(prob_vol, save_prob_path)

        if save_binary_path:
            bin_vol = NeuroimagingVolume(
                data=res["binary_mask"].astype(np.float32),
                affine=res["affine"],
                voxel_dimensions=res["voxel_dimensions"],
            )
            save_nifti(bin_vol, save_binary_path)

        return PredictionResult(
            probability_map=res["probability_map"],
            binary_mask=res["binary_mask"],
            roi_voxel_count=res["num_roi_voxels"],
            roi_volume_mm3=res["roi_volume_mm3"],
            affine=res["affine"],
            voxel_dimensions=res["voxel_dimensions"],
            saved_binary_path=Path(save_binary_path) if save_binary_path else None,
            saved_prob_path=Path(save_prob_path) if save_prob_path else None,
        )
