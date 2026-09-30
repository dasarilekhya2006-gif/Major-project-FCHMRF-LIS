"""Strongly-typed configuration loader using Pydantic and PyYAML."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

from src.core.exceptions import ConfigurationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ProjectSettings(BaseModel):
    name: str = "Adaptive Deep Learning-Assisted fcHMRF-LIS"
    version: str = "0.1.0"
    description: str = "Spatial False Discovery Rate Control in Neuroimaging"
    seed: int = 42
    device: str = "cpu"
    log_level: str = "INFO"


class DataSettings(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    default_affine: str = "MNI152"
    mask_threshold: float = 0.1
    spatial_shape: List[int] = Field(default_factory=lambda: [64, 64, 32])


class PreprocessingSettings(BaseModel):
    normalize: bool = True
    normalization_method: str = "zscore"
    gaussian_fwhm_mm: float = 4.0
    spatial_connectivity: int = 26


class DeepLearningSettings(BaseModel):
    model_name: str = "lightweight_unet3d"
    in_channels: int = 1
    base_channels: int = 16
    num_classes: int = 1
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 4
    epochs: int = 50
    checkpoint_dir: str = "checkpoints"
    use_pretrained_prior: bool = false_value if False else False


class StatisticalAnalysisSettings(BaseModel):
    nominal_fdr_alpha: float = 0.05
    null_distribution: str = "empirical_gaussian"
    empirical_null_method: str = "central_matching"
    run_baselines: List[str] = Field(
        default_factory=lambda: [
            "benjamini_hochberg",
            "benjamini_yekutieli",
            "storey_qvalue",
            "efron_locfdr",
            "standard_hmrf_lis",
        ]
    )


class FCHMRFSettings(BaseModel):
    max_iterations: int = 30
    convergence_tolerance: float = 1e-4
    spatial_sigma: float = 2.0
    feature_sigma: float = 1.5
    spatial_cutoff_radius: int = 5
    mean_field_damping: float = 0.5


class AdaptiveParameterSettings(BaseModel):
    method: str = "mean_field_em"
    beta_initial: float = 0.8
    beta_bounds: List[float] = Field(default_factory=lambda: [0.0, 3.0])
    lambda_dl_initial: float = 0.5
    lambda_dl_bounds: List[float] = Field(default_factory=lambda: [0.0, 1.0])
    em_max_iter: int = 20
    em_tolerance: float = 1e-3


class EvaluationSettings(BaseModel):
    nominal_alpha_levels: List[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10])
    metrics: List[str] = Field(
        default_factory=lambda: [
            "empirical_fdr",
            "true_positive_rate",
            "false_nondiscovery_rate",
            "f1_score",
            "matthews_corrcoef",
            "dice_coefficient",
        ]
    )
    output_dir: str = "outputs/evaluation"


class RobustnessSettings(BaseModel):
    snr_levels: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0])
    noise_types: List[str] = Field(
        default_factory=lambda: ["gaussian", "student_t_3df", "spatially_correlated_gaussian"]
    )
    null_shifts: List[float] = Field(default_factory=lambda: [-0.5, 0.0, 0.5])
    null_scale_distortions: List[float] = Field(default_factory=lambda: [0.8, 1.0, 1.25])
    hyperparameter_perturbation_pct: float = 20.0
    output_dir: str = "outputs/robustness"


class VisualizationSettings(BaseModel):
    dpi: int = 300
    colormap_zscore: str = "cold_hot"
    colormap_lis: str = "viridis_r"
    export_format: str = "png"
    output_dir: str = "outputs/figures"


class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True


class ProjectConfig(BaseModel):
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    deep_learning: DeepLearningSettings = Field(default_factory=DeepLearningSettings)
    statistical_analysis: StatisticalAnalysisSettings = Field(
        default_factory=StatisticalAnalysisSettings
    )
    fchmrf_lis: FCHMRFSettings = Field(default_factory=FCHMRFSettings)
    adaptive_parameter_selection: AdaptiveParameterSettings = Field(
        default_factory=AdaptiveParameterSettings
    )
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    robustness: RobustnessSettings = Field(default_factory=RobustnessSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    api: APISettings = Field(default_factory=APISettings)


def load_config(config_path: Optional[Path] = None) -> ProjectConfig:
    """Load configuration from a YAML file or return defaults.

    Args:
        config_path: Path to the YAML file. If None, default configs/default.yaml is used.

    Returns:
        ProjectConfig: Validated strongly-typed configuration object.

    Raises:
        ConfigurationError: If the file cannot be parsed or validation fails.
    """
    if config_path is None:
        config_path = Path("configs/default.yaml")

    if not config_path.exists():
        logger.warning(f"Configuration file not found at {config_path}. Using default configuration.")
        return ProjectConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
        config = ProjectConfig.model_validate(raw_dict)
        logger.info(f"Loaded configuration successfully from {config_path}")
        return config
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}") from e
