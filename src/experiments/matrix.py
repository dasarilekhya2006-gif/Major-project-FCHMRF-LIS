"""Experiment matrix and grid configuration builder for systematic research sweeps."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.experiments.schemas import (
    AdaptiveConfig,
    DatasetConfig,
    ExperimentConfig,
    FCHMRFConfig,
    PreprocessingConfig,
    StatisticalTestConfig,
    UNetConfig,
)


def build_experiment_matrix(
    snr_list: List[float] = [1.5, 2.5],
    methods: List[str] = [
        "baseline_bh",
        "original_fchmrf",
        "proposed_fchmrf",
        "proposed_adaptive_fchmrf",
    ],
    seeds: List[int] = [42],
    alpha: float = 0.05,
    grid_size: tuple = (40, 40, 20),
    base_output_dir: str = "outputs/automated_experiments",
) -> List[ExperimentConfig]:
    """Build a systematic grid matrix of experiment configurations across SNR, methods, and seeds."""
    configs: List[ExperimentConfig] = []

    for snr in snr_list:
        for seed in seeds:
            for method in methods:
                exp_id = f"exp_snr{snr:.1f}_{method}_seed{seed}"
                name = f"Benchmark SNR={snr:.1f} ({method.replace('_', ' ').title()})"

                # Base dataset configuration
                ds_cfg = DatasetConfig(
                    dataset_type="synthetic",
                    grid_size=grid_size,
                    snr=snr,
                    num_clusters=3,
                    cluster_radius=4,
                )

                # Preprocessing
                preproc_cfg = PreprocessingConfig(norm_method="zscore")

                # U-Net Prior (enabled for proposed methods)
                is_proposed = "proposed" in method
                unet_cfg = UNetConfig(
                    enabled=is_proposed,
                    roi_threshold=0.5,
                )

                # Statistical Test
                stat_cfg = StatisticalTestConfig(
                    test_type="one_sample_t",
                    mu_null=0.0,
                    alternative="greater",
                    restrict_to_roi=is_proposed,
                )

                # fcHMRF
                is_adaptive = "adaptive" in method
                lambda_val = 0.5 if is_proposed else 0.0
                fchmrf_cfg = FCHMRFConfig(
                    method_mode=method,
                    beta=0.8,
                    sigma_spatial=2.5,
                    sigma_intensity=1.5,
                    lambda_dl=lambda_val,
                    alpha=alpha,
                    max_iterations=30,
                    tolerance=1e-4,
                    damping_tau=0.7,
                )

                # Adaptive Optimization
                adapt_cfg = AdaptiveConfig(
                    enabled=is_adaptive,
                    algorithm="bayesian_optimization" if is_adaptive else "none",
                    n_iterations=6,
                    n_initial=2,
                    acquisition_function="expected_improvement",
                )

                cfg = ExperimentConfig(
                    experiment_id=exp_id,
                    name=name,
                    description=f"Systematic benchmark evaluating {method} under SNR={snr} with seed={seed}",
                    seed=seed,
                    device="cpu",
                    output_dir=base_output_dir,
                    dataset=ds_cfg,
                    preprocessing=preproc_cfg,
                    unet=unet_cfg,
                    statistical_test=stat_cfg,
                    fchmrf=fchmrf_cfg,
                    adaptive=adapt_cfg,
                )
                configs.append(cfg)

    return configs


def load_suite_from_yaml(yaml_path: Union[str, Path]) -> List[ExperimentConfig]:
    """Parse an experiment suite configuration file and expand matrices into ExperimentConfig objects."""
    p = Path(yaml_path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "experiments" in data and isinstance(data["experiments"], list):
        # Explicit list of experiments
        return [ExperimentConfig.from_dict(d) for d in data["experiments"]]

    if "matrix" in data:
        # Parameter sweep matrix
        mat = data["matrix"]
        snr_list = mat.get("snr_list", [2.0])
        methods = mat.get("methods", ["baseline_bh", "proposed_fchmrf"])
        seeds = mat.get("seeds", [42])
        alpha = float(mat.get("alpha", 0.05))
        grid = tuple(mat.get("grid_size", [40, 40, 20]))
        out_dir = mat.get("output_dir", "outputs/automated_experiments")
        return build_experiment_matrix(
            snr_list=snr_list,
            methods=methods,
            seeds=seeds,
            alpha=alpha,
            grid_size=grid,
            base_output_dir=out_dir,
        )

    # Single experiment fallback
    return [ExperimentConfig.from_dict(data)]
