"""Schemas and configuration dataclasses for automated research experiments."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml


@dataclass
class DatasetConfig:
    """Specification of the experimental neuroimaging dataset."""

    dataset_type: str = "synthetic"  # "synthetic" or "nifti"
    grid_size: Tuple[int, int, int] = (40, 40, 20)
    snr: float = 2.0
    num_clusters: int = 3
    cluster_radius: int = 4
    spatial_smoothing_sigma: float = 1.5
    noise_distribution: str = "gaussian"  # "gaussian", "student_t", "correlated"
    baseline_offset: float = 0.0

    # For NIfTI input
    input_path: Optional[str] = None
    ground_truth_path: Optional[str] = None
    mask_path: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.grid_size, (list, tuple)):
            self.grid_size = tuple(int(x) for x in self.grid_size)


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""

    norm_method: str = "zscore"  # "zscore", "robust", "minmax", "none"
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    smoothing_fwhm: float = 0.0  # 0.0 = no extra smoothing
    apply_brain_mask: bool = True

    def __post_init__(self):
        if isinstance(self.clip_percentiles, (list, tuple)):
            self.clip_percentiles = tuple(float(x) for x in self.clip_percentiles)


@dataclass
class UNetConfig:
    """3D U-Net Deep Learning ROI detection configuration."""

    enabled: bool = True
    checkpoint_path: Optional[str] = None
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 16
    depth: int = 3
    activation: str = "relu"
    roi_threshold: float = 0.5
    prior_mode: str = "simulated_expert"  # "model_checkpoint", "simulated_expert", "uniform"


@dataclass
class StatisticalTestConfig:
    """Voxel-level statistical testing configuration."""

    test_type: str = "one_sample_t"  # "one_sample_t", "two_sample_t", "glm"
    mu_null: float = 0.0
    alternative: str = "greater"  # "greater", "two_sided"
    restrict_to_roi: bool = True


@dataclass
class FCHMRFConfig:
    """Fully Connected HMRF and LIS spatial modeling configuration."""

    method_mode: str = "proposed_adaptive_fchmrf"
    # "proposed_adaptive_fchmrf", "proposed_fchmrf", "original_fchmrf", "baseline_bh", "baseline_locfdr"
    beta: float = 0.8  # Spatial coupling strength
    sigma_spatial: float = 2.5  # Spatial kernel bandwidth
    sigma_intensity: float = 1.5  # Feature / intensity bandwidth
    lambda_dl: float = 0.5  # Weight of deep learning prior (0 = pure HMRF, 1 = pure DL)
    alpha: float = 0.05  # Target nominal FDR level
    max_iterations: int = 30  # Max Mean-Field iterations
    tolerance: float = 1e-4  # Convergence criterion
    damping_tau: float = 0.8  # Mean-field damping factor (0 < tau <= 1)


@dataclass
class AdaptiveConfig:
    """Adaptive hyperparameter selection configuration."""

    enabled: bool = False
    algorithm: str = "bayesian_optimization"  # "bayesian_optimization", "random_search", "none"
    n_iterations: int = 8
    n_initial: int = 3
    acquisition_function: str = "expected_improvement"
    target_metric: str = "validation_fdr"


@dataclass
class ExperimentConfig:
    """Complete specification of a single reproducible research experiment."""

    experiment_id: str = "exp_default"
    name: str = "Default Experiment"
    description: str = "Standard reproducible experiment run."
    seed: int = 42
    device: str = "cpu"  # "cpu", "cuda", "auto"
    output_dir: str = "outputs/automated_experiments"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    statistical_test: StatisticalTestConfig = field(default_factory=StatisticalTestConfig)
    fchmrf: FCHMRFConfig = field(default_factory=FCHMRFConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "device": self.device,
            "output_dir": str(self.output_dir),
            "dataset": asdict(self.dataset),
            "preprocessing": asdict(self.preprocessing),
            "unet": asdict(self.unet),
            "statistical_test": asdict(self.statistical_test),
            "fchmrf": asdict(self.fchmrf),
            "adaptive": asdict(self.adaptive),
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Serialize configuration to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Serialize configuration to YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Construct ExperimentConfig from a dictionary."""
        ds = d.get("dataset", {})
        ds_cfg = ds if isinstance(ds, DatasetConfig) else DatasetConfig(**ds)

        pre = d.get("preprocessing", {})
        pre_cfg = pre if isinstance(pre, PreprocessingConfig) else PreprocessingConfig(**pre)

        un = d.get("unet", {})
        un_cfg = un if isinstance(un, UNetConfig) else UNetConfig(**un)

        st = d.get("statistical_test", {})
        st_cfg = st if isinstance(st, StatisticalTestConfig) else StatisticalTestConfig(**st)

        fc = d.get("fchmrf", {})
        fc_cfg = fc if isinstance(fc, FCHMRFConfig) else FCHMRFConfig(**fc)

        ad = d.get("adaptive", {})
        ad_cfg = ad if isinstance(ad, AdaptiveConfig) else AdaptiveConfig(**ad)

        return cls(
            experiment_id=d.get("experiment_id", "exp_default"),
            name=d.get("name", "Default Experiment"),
            description=d.get("description", ""),
            seed=int(d.get("seed", 42)),
            device=d.get("device", "cpu"),
            output_dir=d.get("output_dir", "outputs/automated_experiments"),
            dataset=ds_cfg,
            preprocessing=pre_cfg,
            unet=un_cfg,
            statistical_test=st_cfg,
            fchmrf=fc_cfg,
            adaptive=ad_cfg,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass
class ExperimentResult:
    """Complete recorded results and provenance from an automated experiment run."""

    experiment_id: str
    config: ExperimentConfig
    status: str = "completed"  # "completed", "failed"
    error: Optional[str] = None

    # Latency tracking across pipeline stages
    runtime_breakdown: Dict[str, float] = field(default_factory=dict)
    total_runtime: float = 0.0

    # Quantitative evaluation metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Saved output artifact locations
    output_locations: Dict[str, str] = field(default_factory=dict)

    # Environment and reproducibility audit
    system_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "error": self.error,
            "config": self.config.to_dict(),
            "total_runtime": self.total_runtime,
            "runtime_breakdown": self.runtime_breakdown,
            "metrics": self.metrics,
            "output_locations": self.output_locations,
            "system_metadata": self.system_metadata,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save results summary to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_current_system_metadata() -> Dict[str, Any]:
    """Capture current execution environment metadata for research provenance."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor(),
    }
    try:
        import numpy as np
        meta["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import scipy
        meta["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        meta["cuda_available"] = False
    try:
        import nibabel as nib
        meta["nibabel_version"] = nib.__version__
    except ImportError:
        pass
    return meta
