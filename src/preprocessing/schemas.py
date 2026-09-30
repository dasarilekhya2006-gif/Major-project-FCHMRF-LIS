"""Configuration schemas, audit metadata, and container structures for preprocessing."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PreprocessingStepRecord:
    """Record of an individual preprocessing transformation applied to a volume.

    Attributes:
        step_name: Identifier for the preprocessing step (e.g. 'imputation', 'normalization').
        timestamp: ISO timestamp when the step executed.
        parameters: Dictionary of parameters used for the transformation.
        input_shape: Shape of the array prior to the step.
        output_shape: Shape of the array after the step.
        summary_stats: Numeric intensity statistics before and after (min, max, mean, std).
    """

    step_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    output_shape: Tuple[int, ...] = field(default_factory=tuple)
    summary_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class PreprocessingMetadata:
    """Complete audit record of all preprocessing applied to a volume.

    Attributes:
        pipeline_version: Version identifier of the preprocessing pipeline.
        created_at: ISO timestamp of preprocessing completion.
        original_shape: Shape of the input volume prior to any transformation.
        original_affine: 4x4 affine matrix prior to transformation.
        original_voxel_dimensions: Voxel spacing prior to transformation.
        preprocessed_shape: Final spatial shape.
        preprocessed_affine: Final updated 4x4 affine matrix.
        preprocessed_voxel_dimensions: Final updated voxel spacing in mm.
        pad_crop_offsets: Spatial crop/pad offsets applied along each axis ((x0, x1), (y0, y1), (z0, z1)).
        normalization_center: Centering parameter (e.g. mean or median) used.
        normalization_scale: Scaling parameter (e.g. std or IQR) used.
        history: Ordered list of PreprocessingStepRecord objects.
    """

    pipeline_version: str = "0.2.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    original_shape: Tuple[int, ...] = field(default_factory=tuple)
    original_affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    original_voxel_dimensions: Tuple[float, ...] = (2.0, 2.0, 2.0)
    preprocessed_shape: Tuple[int, ...] = field(default_factory=tuple)
    preprocessed_affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    preprocessed_voxel_dimensions: Tuple[float, ...] = (2.0, 2.0, 2.0)
    pad_crop_offsets: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0), (0, 0))
    normalization_center: float = 0.0
    normalization_scale: float = 1.0
    history: List[PreprocessingStepRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        d = asdict(self)
        d["original_affine"] = self.original_affine.tolist()
        d["preprocessed_affine"] = self.preprocessed_affine.tolist()
        return d


@dataclass
class PreprocessingConfig:
    """Configurable parameters for the neuroimaging preprocessing pipeline.

    Attributes:
        imputation_method: Strategy for handling missing/invalid voxels ('zero', 'mean', 'nearest_finite').
        normalization_method: Strategy for intensity scaling ('zscore', 'robust_zscore', 'min_max', 'percentile_clip', 'none').
        percentile_bounds: Lower and upper percentiles for winsorization if percentile_clip is enabled.
        denoising_method: Strategy for noise reduction ('none', 'gaussian', 'median').
        gaussian_fwhm_mm: Full Width at Half Maximum in millimeters for Gaussian denoising.
        median_kernel_size: 3D kernel radius or size for median filtering.
        target_voxel_spacing: Optional target isotropic voxel spacing in mm (e.g. (2.0, 2.0, 2.0)). If None, preserved.
        target_spatial_shape: Optional fixed target 3D spatial shape (e.g. (64, 64, 32)). If None, calculated dynamically.
        pad_crop_mode: Dimension standardization mode ('center_pad_crop', 'pad_to_divisible', 'none').
        divisible_by: Divisibility factor required by 3D U-Net pooling stages (typically 8 or 16).
        compute_mask_if_missing: If True, compute brain mask automatically if none provided.
        mask_threshold_percentile: Lower intensity percentile cutoff for brain mask estimation.
        fill_mask_holes: If True, apply morphological hole-filling to estimated brain masks.
        canonical_reorient: If True, standardize orientation to RAS+ coordinate axes.
    """

    imputation_method: str = "zero"
    normalization_method: str = "zscore"
    percentile_bounds: Tuple[float, float] = (1.0, 99.0)
    denoising_method: str = "none"
    gaussian_fwhm_mm: float = 4.0
    median_kernel_size: int = 3
    target_voxel_spacing: Optional[Tuple[float, float, float]] = None
    target_spatial_shape: Optional[Tuple[int, int, int]] = None
    pad_crop_mode: str = "pad_to_divisible"
    divisible_by: int = 8
    compute_mask_if_missing: bool = True
    mask_threshold_percentile: float = 10.0
    fill_mask_holes: bool = True
    canonical_reorient: bool = True

    def validate(self) -> None:
        """Validate parameter configuration."""
        valid_imputations = ("zero", "mean", "nearest_finite")
        if self.imputation_method not in valid_imputations:
            raise ValueError(f"imputation_method must be one of {valid_imputations}, got '{self.imputation_method}'")

        valid_normalizations = ("zscore", "robust_zscore", "min_max", "percentile_clip", "none")
        if self.normalization_method not in valid_normalizations:
            raise ValueError(f"normalization_method must be one of {valid_normalizations}, got '{self.normalization_method}'")

        valid_denoisings = ("none", "gaussian", "median")
        if self.denoising_method not in valid_denoisings:
            raise ValueError(f"denoising_method must be one of {valid_denoisings}, got '{self.denoising_method}'")

        valid_pad_crop = ("center_pad_crop", "pad_to_divisible", "none")
        if self.pad_crop_mode not in valid_pad_crop:
            raise ValueError(f"pad_crop_mode must be one of {valid_pad_crop}, got '{self.pad_crop_mode}'")

        if self.divisible_by <= 0 or (self.divisible_by & (self.divisible_by - 1)) != 0:
            raise ValueError(f"divisible_by must be a positive power of 2, got {self.divisible_by}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PreprocessingConfig":
        """Load PreprocessingConfig from a YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Handle nested or flat YAML
        data: Dict[str, Any] = {}
        if "imputation" in raw and isinstance(raw["imputation"], dict):
            data["imputation_method"] = raw["imputation"].get("method", "zero")
        if "normalization" in raw and isinstance(raw["normalization"], dict):
            data["normalization_method"] = raw["normalization"].get("method", "zscore")
            if "percentile_bounds" in raw["normalization"]:
                data["percentile_bounds"] = tuple(raw["normalization"]["percentile_bounds"])
        if "denoising" in raw and isinstance(raw["denoising"], dict):
            data["denoising_method"] = raw["denoising"].get("method", "none")
            data["gaussian_fwhm_mm"] = raw["denoising"].get("gaussian_fwhm_mm", 4.0)
            data["median_kernel_size"] = raw["denoising"].get("median_kernel_size", 3)
        if "resampling" in raw and isinstance(raw["resampling"], dict):
            if raw["resampling"].get("enabled", False) and "target_voxel_spacing_mm" in raw["resampling"]:
                data["target_voxel_spacing"] = tuple(raw["resampling"]["target_voxel_spacing_mm"])
        if "masking" in raw and isinstance(raw["masking"], dict):
            data["compute_mask_if_missing"] = raw["masking"].get("compute_if_missing", True)
            data["mask_threshold_percentile"] = raw["masking"].get("threshold_percentile", 10.0)
            data["fill_mask_holes"] = raw["masking"].get("fill_holes", True)
        if "standardization_for_dl" in raw and isinstance(raw["standardization_for_dl"], dict):
            data["pad_crop_mode"] = raw["standardization_for_dl"].get("pad_crop_mode", "pad_to_divisible")
            data["divisible_by"] = raw["standardization_for_dl"].get("divisible_by", 8)
            tgt = raw["standardization_for_dl"].get("target_spatial_shape", None)
            data["target_spatial_shape"] = tuple(tgt) if tgt else None

        # Allow direct top-level keys to override
        for k, v in raw.items():
            if hasattr(cls, k) and not isinstance(v, dict):
                data[k] = tuple(v) if isinstance(v, list) and k in ("percentile_bounds", "target_voxel_spacing", "target_spatial_shape") else v

        instance = cls(**data)
        instance.validate()
        return instance

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Export PreprocessingConfig to a YAML file."""
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False)

