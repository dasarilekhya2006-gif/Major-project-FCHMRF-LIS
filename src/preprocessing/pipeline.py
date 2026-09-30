"""Integrated, modular neuroimaging preprocessing pipeline.

Orchestrates missing voxel imputation, spatial brain masking, optional denoising,
intensity normalization, spatial resampling, and dimension standardization for 3D deep-learning models.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger
from src.data.loaders import load_nifti, save_nifti
from src.data.schemas import NeuroimagingVolume
from src.preprocessing.denoising import denoise_volume
from src.preprocessing.imputation import handle_missing_voxels
from src.preprocessing.masking import apply_mask, load_or_compute_mask, unmask
from src.preprocessing.normalization import normalize_intensities
from src.preprocessing.resampling import resample_to_spacing
from src.preprocessing.resizing import standardize_spatial_dimensions
from src.preprocessing.schemas import (
    PreprocessingConfig,
    PreprocessingMetadata,
    PreprocessingStepRecord,
)

logger = get_logger(__name__)


@dataclass
class PreprocessedVolume(NeuroimagingVolume):
    """Normalized and standardized volume output from the preprocessing pipeline.

    Attributes:
        preprocessing_metadata: Audit metadata tracking all executed preprocessing steps.
    """

    preprocessing_metadata: PreprocessingMetadata = field(default_factory=PreprocessingMetadata)

    def to_torch_tensor(
        self,
        device: Optional[str] = None,
        add_batch_dim: bool = True,
        add_channel_dim: bool = True,
    ) -> Any:
        """Convert preprocessed volume data into a PyTorch 3D tensor suitable for 3D U-Net inference.

        Returns:
            torch.Tensor with shape (Batch, Channel, Depth, Height, Width) -> (1, 1, D, H, W)
        """
        try:
            import torch
        except ImportError as e:
            raise PreprocessingError("PyTorch is required to convert volume to torch.Tensor.") from e

        # Extract 3D spatial field (first volume if 4D)
        arr = self.data[..., 0] if self.is_4d else self.data
        tensor = torch.from_numpy(arr.astype(np.float32))

        if add_channel_dim:
            tensor = tensor.unsqueeze(0)  # (1, D, H, W)
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)  # (1, 1, D, H, W)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def save_preprocessed(
        self,
        output_dir: Union[str, Path],
        prefix: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Save preprocessed volume, mask, and audit metadata to disk.

        Args:
            output_dir: Destination directory.
            prefix: Optional filename prefix (defaults to subject_id or 'preproc').

        Returns:
            Dict[str, Path]: Paths to saved volume, mask, and metadata JSON files.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pfx = prefix or self.subject_id or "preproc"

        vol_path = out_dir / f"{pfx}_preprocessed.nii.gz"
        save_nifti(self, vol_path)

        mask_path = None
        if self.mask is not None:
            mask_vol = NeuroimagingVolume(
                data=self.mask.astype(np.float32),
                affine=self.affine,
                voxel_dimensions=self.voxel_dimensions,
                subject_id=f"{pfx}_mask",
            )
            mask_path = out_dir / f"{pfx}_brain_mask.nii.gz"
            save_nifti(mask_vol, mask_path)

        meta_path = out_dir / f"{pfx}_preprocessing_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.preprocessing_metadata.to_dict(), f, indent=2)

        saved = {
            "volume": vol_path,
            "metadata": meta_path,
        }
        if mask_path is not None:
            saved["mask"] = mask_path

        logger.info(f"Saved preprocessed bundle ({pfx}) in {out_dir}")
        return saved


class PreprocessingPipeline:
    """Configurable, reproducible neuroimaging preprocessing pipeline."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """Initialize the pipeline with user-specified or default configuration.

        Args:
            config: Optional PreprocessingConfig instance.
        """
        self.config = config or PreprocessingConfig()
        self.config.validate()

    def run(
        self,
        volume_or_path: Union[str, Path, NeuroimagingVolume],
        mask_path: Optional[Union[str, Path]] = None,
        subject_id: Optional[str] = None,
    ) -> PreprocessedVolume:
        """Execute all configured preprocessing steps in disciplined sequential order.

        Order of operations:
          1. Ingestion / canonical reorientation
          2. Missing/non-finite voxel imputation
          3. Brain mask resolution
          4. Spatial resampling (if target_voxel_spacing configured)
          5. Optional denoising (Gaussian / median)
          6. Intensity normalization (z-score / robust / min-max)
          7. Spatial dimension standardization (padding/cropping for 3D U-Net)
          8. Audit metadata compilation

        Args:
            volume_or_path: Path to NIfTI file or loaded NeuroimagingVolume.
            mask_path: Optional path to binary brain mask.
            subject_id: Optional subject identifier.

        Returns:
            PreprocessedVolume: Fully preprocessed volume with attached audit trail.
        """
        # 1. Ingestion
        if isinstance(volume_or_path, (str, Path)):
            volume = load_nifti(
                volume_or_path,
                mask_path=mask_path,
                canonical_reorient=self.config.canonical_reorient,
                validate=True,
            )
        elif isinstance(volume_or_path, NeuroimagingVolume):
            volume = volume_or_path
        else:
            raise PreprocessingError(f"Unsupported input type: {type(volume_or_path)}")

        subj = subject_id or volume.subject_id or "volume"
        history: List[PreprocessingStepRecord] = []

        # Record initial baseline state
        orig_shape = volume.shape
        orig_affine = volume.affine.copy()
        orig_zooms = volume.voxel_dimensions

        # Helper to compute summary stats
        def _get_stats(arr: np.ndarray) -> Dict[str, float]:
            fin = arr[np.isfinite(arr)]
            if len(fin) == 0:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            return {
                "min": float(np.min(fin)),
                "max": float(np.max(fin)),
                "mean": float(np.mean(fin)),
                "std": float(np.std(fin)),
            }

        # 2. Missing/invalid voxel imputation
        imputed_data, num_imputed = handle_missing_voxels(
            volume.data,
            mask=volume.mask,
            method=self.config.imputation_method,
        )
        history.append(
            PreprocessingStepRecord(
                step_name="imputation",
                parameters={"method": self.config.imputation_method, "num_imputed": num_imputed},
                input_shape=volume.shape,
                output_shape=imputed_data.shape,
                summary_stats=_get_stats(imputed_data),
            )
        )
        volume = NeuroimagingVolume(
            data=imputed_data,
            affine=volume.affine,
            voxel_dimensions=volume.voxel_dimensions,
            mask=volume.mask,
            subject_id=subj,
            metadata=volume.metadata,
        )

        # 3. Brain masking resolution
        resolved_mask = volume.mask
        if resolved_mask is None and self.config.compute_mask_if_missing:
            resolved_mask = load_or_compute_mask(
                volume_data=volume.data,
                mask_path=mask_path,
                percentile=self.config.mask_threshold_percentile,
                fill_holes=self.config.fill_mask_holes,
            )
            history.append(
                PreprocessingStepRecord(
                    step_name="brain_masking",
                    parameters={
                        "computed": True,
                        "percentile": self.config.mask_threshold_percentile,
                        "mask_voxels": int(np.sum(resolved_mask)),
                    },
                    input_shape=volume.shape,
                    output_shape=volume.shape,
                    summary_stats={"in_mask_voxels": float(np.sum(resolved_mask))},
                )
            )
            volume = NeuroimagingVolume(
                data=volume.data,
                affine=volume.affine,
                voxel_dimensions=volume.voxel_dimensions,
                mask=resolved_mask,
                subject_id=subj,
                metadata=volume.metadata,
            )

        # 4. Spatial resampling (when required)
        if self.config.target_voxel_spacing is not None:
            shape_before_resample = volume.shape
            volume = resample_to_spacing(
                volume,
                target_spacing=self.config.target_voxel_spacing,
                order=1,
            )
            history.append(
                PreprocessingStepRecord(
                    step_name="spatial_resampling",
                    parameters={"target_voxel_spacing": list(self.config.target_voxel_spacing)},
                    input_shape=shape_before_resample,
                    output_shape=volume.shape,
                    summary_stats=_get_stats(volume.data),
                )
            )

        # 5. Optional spatial denoising
        if self.config.denoising_method != "none":
            denoised_data = denoise_volume(
                volume.data,
                method=self.config.denoising_method,
                fwhm_mm=self.config.gaussian_fwhm_mm,
                voxel_dims=volume.voxel_dimensions,
                median_kernel_size=self.config.median_kernel_size,
                mask=volume.mask,
            )
            history.append(
                PreprocessingStepRecord(
                    step_name="denoising",
                    parameters={
                        "method": self.config.denoising_method,
                        "fwhm_mm": self.config.gaussian_fwhm_mm,
                        "median_kernel_size": self.config.median_kernel_size,
                    },
                    input_shape=volume.shape,
                    output_shape=denoised_data.shape,
                    summary_stats=_get_stats(denoised_data),
                )
            )
            volume = NeuroimagingVolume(
                data=denoised_data,
                affine=volume.affine,
                voxel_dimensions=volume.voxel_dimensions,
                mask=volume.mask,
                subject_id=subj,
                metadata=volume.metadata,
            )

        # 6. Intensity normalization
        norm_data, center, scale = normalize_intensities(
            volume.data,
            mask=volume.mask,
            method=self.config.normalization_method,
            percentile_bounds=self.config.percentile_bounds,
        )
        history.append(
            PreprocessingStepRecord(
                step_name="intensity_normalization",
                parameters={
                    "method": self.config.normalization_method,
                    "center": center,
                    "scale": scale,
                    "percentile_bounds": list(self.config.percentile_bounds),
                },
                input_shape=volume.shape,
                output_shape=norm_data.shape,
                summary_stats=_get_stats(norm_data),
            )
        )
        volume = NeuroimagingVolume(
            data=norm_data,
            affine=volume.affine,
            voxel_dimensions=volume.voxel_dimensions,
            mask=volume.mask,
            subject_id=subj,
            metadata=volume.metadata,
        )

        # 7. Spatial dimension standardization (padding/cropping for 3D U-Net)
        shape_before_standardize = volume.shape
        volume, pad_crop_offsets = standardize_spatial_dimensions(
            volume,
            target_shape=self.config.target_spatial_shape,
            divisible_by=self.config.divisible_by,
            mode=self.config.pad_crop_mode,
        )
        if volume.shape != shape_before_standardize:
            history.append(
                PreprocessingStepRecord(
                    step_name="dimension_standardization",
                    parameters={
                        "mode": self.config.pad_crop_mode,
                        "divisible_by": self.config.divisible_by,
                        "target_shape": self.config.target_spatial_shape,
                        "offsets": pad_crop_offsets,
                    },
                    input_shape=shape_before_standardize,
                    output_shape=volume.shape,
                    summary_stats=_get_stats(volume.data),
                )
            )

        # 8. Assemble PreprocessingMetadata
        meta = PreprocessingMetadata(
            original_shape=orig_shape,
            original_affine=orig_affine,
            original_voxel_dimensions=orig_zooms,
            preprocessed_shape=volume.shape,
            preprocessed_affine=volume.affine.copy(),
            preprocessed_voxel_dimensions=volume.voxel_dimensions,
            pad_crop_offsets=pad_crop_offsets,
            normalization_center=center,
            normalization_scale=scale,
            history=history,
        )

        logger.info(
            f"Completed preprocessing for '{subj}': shape {orig_shape} -> {volume.shape}, "
            f"norm method='{self.config.normalization_method}' (center={center:.3f}, scale={scale:.3f})"
        )

        return PreprocessedVolume(
            data=volume.data,
            affine=volume.affine,
            voxel_dimensions=volume.voxel_dimensions,
            mask=volume.mask,
            subject_id=subj,
            metadata=volume.metadata,
            preprocessing_metadata=meta,
        )


def preprocess_neuroimaging_volume(
    volume_or_path: Union[str, Path, NeuroimagingVolume],
    config: Optional[PreprocessingConfig] = None,
    **kwargs: Any,
) -> PreprocessedVolume:
    """Convenience functional API to execute the neuroimaging preprocessing pipeline.

    Args:
        volume_or_path: Path to NIfTI file or loaded NeuroimagingVolume.
        config: Optional PreprocessingConfig.
        **kwargs: Overrides for PreprocessingConfig fields (e.g. normalization_method="min_max").

    Returns:
        PreprocessedVolume: Processed volume ready for 3D U-Net and statistical analysis.
    """
    cfg = config or PreprocessingConfig()
    if kwargs:
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    pipeline = PreprocessingPipeline(config=cfg)
    return pipeline.run(volume_or_path)
