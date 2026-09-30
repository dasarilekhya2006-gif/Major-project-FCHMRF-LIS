"""Robustness and uncertainty evaluation orchestrator for adaptive fcHMRF-LIS."""

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.robustness.metrics import (
    StabilityMetrics,
    compute_detection_probability_map,
    compute_pairwise_jaccard_and_dice,
    compute_voxel_uncertainty_entropy,
    evaluate_repetition_stability,
)
from src.robustness.perturbations import (
    apply_additive_voxel_perturbation,
    generate_snr_sweep_dataset,
    generate_spatial_correlation_sweep_dataset,
    generate_roi_size_sweep_dataset,
)
from src.evaluation.synthetic_benchmarks import (
    SyntheticBenchmarkVolume,
    generate_synthetic_benchmark_volume,
)

logger = get_logger(__name__)


@dataclass
class RobustnessEvaluationReport:
    """Consolidated report across all robustness sweeps and uncertainty analyses."""

    snr_sweep_metrics: List[StabilityMetrics]
    perturbation_sweep_metrics: List[StabilityMetrics]
    spatial_corr_metrics: List[StabilityMetrics]
    hyperparam_sweep_metrics: List[StabilityMetrics]
    multi_seed_metrics: StabilityMetrics
    detection_probability_map: np.ndarray
    uncertainty_entropy_map: np.ndarray
    nominal_alpha: float = 0.05

    def to_markdown_table(self) -> str:
        """Render publication-grade Markdown table summarizing all robustness sweeps."""
        lines = [
            "### 1. Signal-to-Noise Ratio (SNR) Sensitivity Sweep",
            "| SNR Level | Pairwise Jaccard | Discovery Count CV | Mean TPR (Power) | Empirical FDP | FDR Controlled? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for m in self.snr_sweep_metrics:
            ctrl = "Yes" if m.fdr_control_rate > 0.8 else "Partial/Degraded"
            lines.append(
                f"| **SNR = {m.condition_value:.1f}** | {m.mean_pairwise_jaccard:.3f} ± {m.std_pairwise_jaccard:.3f} "
                f"| {m.discovery_count_cv * 100.0:.1f}% | {m.mean_tpr:.3f} | {m.mean_fdp:.3f} | {ctrl} |"
            )

        lines.extend([
            "",
            "### 2. Input Data Perturbation Sweep (Additive Jitter)",
            "| Noise Sigma | Pairwise Jaccard | Discovery Count CV | Boundary Entropy (bits) | Empirical FDP |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for m in self.perturbation_sweep_metrics:
            lines.append(
                f"| **sigma = {m.condition_value:.2f}** | {m.mean_pairwise_jaccard:.3f} ± {m.std_pairwise_jaccard:.3f} "
                f"| {m.discovery_count_cv * 100.0:.1f}% | {m.mean_boundary_entropy:.3f} | {m.mean_fdp:.3f} |"
            )

        lines.extend([
            "",
            "### 3. Spatial Autocorrelation Sweep",
            "| Smooth Sigma | Pairwise Jaccard | Mean Power (TPR) | Empirical FDP | FDR Control Rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for m in self.spatial_corr_metrics:
            lines.append(
                f"| **sigma = {m.condition_value:.1f}** | {m.mean_pairwise_jaccard:.3f} "
                f"| {m.mean_tpr:.3f} | {m.mean_fdp:.3f} | {m.fdr_control_rate * 100.0:.1f}% |"
            )

        lines.extend([
            "",
            "### 4. Hyperparameter Sensitivity (lambda_DL Weighting)",
            "| lambda_DL | Pairwise Jaccard | Mean Discoveries | Mean TPR | Empirical FDP |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for m in self.hyperparam_sweep_metrics:
            lines.append(
                f"| **lambda = {m.condition_value:.2f}** | {m.mean_pairwise_jaccard:.3f} "
                f"| {m.mean_discovery_count:.1f} | {m.mean_tpr:.3f} | {m.mean_fdp:.3f} |"
            )

        lines.extend([
            "",
            "### 5. Multi-Seed Uncertainty & Boundary Entropy",
            f"- **Number of Independent Monte Carlo Seeds**: {self.multi_seed_metrics.num_repetitions}",
            f"- **Pairwise Jaccard Stability**: {self.multi_seed_metrics.mean_pairwise_jaccard:.3f} ± {self.multi_seed_metrics.std_pairwise_jaccard:.3f}",
            f"- **Discovery Count Coefficient of Variation**: {self.multi_seed_metrics.discovery_count_cv * 100.0:.1f}%",
            f"- **Mean Boundary Uncertainty Entropy**: {self.multi_seed_metrics.mean_boundary_entropy:.3f} bits",
            f"- **Empirical FDR Control Rate**: {self.multi_seed_metrics.fdr_control_rate * 100.0:.1f}%",
        ])
        return "\n".join(lines)

    def to_ascii_summary(self) -> str:
        """Render concise ASCII console summary."""
        lines = [
            f"\n{'=' * 95}",
            "PHASE 9 ROBUSTNESS AND UNCERTAINTY EVALUATION SUMMARY",
            f"{'=' * 95}",
            f"{'Condition':<28} | {'Param':<8} | {'Jaccard':<15} | {'Power (TPR)':<13} | {'Empirical FDP':<15} | {'FDR Ctrl'}",
            f"{'-' * 95}",
        ]
        for m in self.snr_sweep_metrics:
            ctrl = "OK" if m.mean_fdp <= self.nominal_alpha else "FAIL"
            lines.append(
                f"{'SNR Sensitivity':<28} | {m.condition_value:<8.1f} | "
                f"{m.mean_pairwise_jaccard:.3f} ± {m.std_pairwise_jaccard:.3f} | "
                f"{m.mean_tpr:.3f}{' ' * 7} | {m.mean_fdp:.3f}{' ' * 9} | {ctrl}"
            )
        for m in self.perturbation_sweep_metrics:
            ctrl = "OK" if m.mean_fdp <= self.nominal_alpha else "FAIL"
            lines.append(
                f"{'Input Perturbation':<28} | {m.condition_value:<8.2f} | "
                f"{m.mean_pairwise_jaccard:.3f} ± {m.std_pairwise_jaccard:.3f} | "
                f"{m.mean_tpr:.3f}{' ' * 7} | {m.mean_fdp:.3f}{' ' * 9} | {ctrl}"
            )
        for m in self.spatial_corr_metrics:
            ctrl = "OK" if m.mean_fdp <= self.nominal_alpha else "FAIL"
            lines.append(
                f"{'Spatial Autocorrelation':<28} | {m.condition_value:<8.1f} | "
                f"{m.mean_pairwise_jaccard:.3f} ± {m.std_pairwise_jaccard:.3f} | "
                f"{m.mean_tpr:.3f}{' ' * 7} | {m.mean_fdp:.3f}{' ' * 9} | {ctrl}"
            )
        lines.append(f"{'-' * 95}")
        lines.append(
            f"Multi-Seed Monte Carlo: Stability={self.multi_seed_metrics.mean_pairwise_jaccard:.3f}, "
            f"Boundary Entropy={self.multi_seed_metrics.mean_boundary_entropy:.3f} bits, "
            f"FDR Control Rate={self.multi_seed_metrics.fdr_control_rate * 100.0:.1f}%"
        )
        lines.append(f"{'=' * 95}\n")
        return "\n".join(lines)

    def export_csv(self, output_path: Union[str, Path]) -> Path:
        """Export all metrics to CSV."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Sweep_Type",
                "Condition_Value",
                "Num_Repetitions",
                "Mean_Pairwise_Jaccard",
                "Std_Pairwise_Jaccard",
                "Mean_Pairwise_Dice",
                "Mean_Discovery_Count",
                "Std_Discovery_Count",
                "Discovery_Count_CV",
                "Mean_Boundary_Entropy_Bits",
                "Mean_TPR_Power",
                "Mean_Empirical_FDP",
                "FDR_Control_Rate",
            ])
            all_metrics = (
                self.snr_sweep_metrics
                + self.perturbation_sweep_metrics
                + self.spatial_corr_metrics
                + self.hyperparam_sweep_metrics
                + [self.multi_seed_metrics]
            )
            for m in all_metrics:
                writer.writerow([
                    m.condition_name,
                    m.condition_value,
                    m.num_repetitions,
                    m.mean_pairwise_jaccard,
                    m.std_pairwise_jaccard,
                    m.mean_pairwise_dice,
                    m.mean_discovery_count,
                    m.std_discovery_count,
                    m.discovery_count_cv,
                    m.mean_boundary_entropy,
                    m.mean_tpr,
                    m.mean_fdp,
                    m.fdr_control_rate,
                ])
        logger.info(f"Saved robustness CSV to {out_p}")
        return out_p


class RobustnessEvaluator:
    """Executes controlled multi-dimensional robustness sweeps on adaptive fcHMRF-LIS."""

    def __init__(
        self,
        nominal_alpha: float = 0.05,
        default_beta: float = 0.8,
        default_sigma_spatial: float = 2.5,
        default_sigma_intensity: float = 1.5,
        default_lambda_dl: float = 0.5,
        max_iterations: int = 30,
    ) -> None:
        self.nominal_alpha = nominal_alpha
        self.default_beta = default_beta
        self.default_sigma_spatial = default_sigma_spatial
        self.default_sigma_intensity = default_sigma_intensity
        self.default_lambda_dl = default_lambda_dl
        self.max_iterations = max_iterations

    def _run_fchmrf(
        self,
        volume: SyntheticBenchmarkVolume,
        beta: Optional[float] = None,
        lambda_dl: Optional[float] = None,
    ) -> np.ndarray:
        """Helper running fcHMRF-LIS on a single volume and returning boolean significant mask."""
        params = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            beta=beta if beta is not None else self.default_beta,
            sigma_spatial=self.default_sigma_spatial,
            sigma_intensity=self.default_sigma_intensity,
            lambda_dl=lambda_dl if lambda_dl is not None else self.default_lambda_dl,
            alpha=self.nominal_alpha,
            max_iterations=self.max_iterations,
        )
        model = FCHMRFModel(params=params)
        res = model.fit_predict(
            z_scores=volume.z_scores,
            dl_prior=volume.dl_prior,
            spatial_mask=volume.spatial_mask,
        )
        return res.significant_mask

    def run_snr_sensitivity_sweep(
        self,
        snr_levels: List[float] = [1.5, 2.5, 3.5],
        num_repetitions: int = 4,
        shape: Tuple[int, int, int] = (20, 20, 10),
    ) -> List[StabilityMetrics]:
        """Sweep across varying Signal-to-Noise Ratio (SNR) levels."""
        logger.info(f"Executing SNR Sensitivity Sweep across levels {snr_levels}...")
        dataset = generate_snr_sweep_dataset(
            snr_levels=snr_levels,
            num_repetitions_per_level=num_repetitions,
            shape=shape,
        )
        results = []
        for snr, vols in dataset.items():
            masks = [self._run_fchmrf(v) for v in vols]
            metrics = evaluate_repetition_stability(
                masks=masks,
                condition_name="SNR_Sensitivity",
                condition_value=float(snr),
                ground_truth=vols[0].ground_truth,
                nominal_alpha=self.nominal_alpha,
                spatial_mask=vols[0].spatial_mask,
            )
            results.append(metrics)
        return results

    def run_input_perturbation_sweep(
        self,
        perturbation_sigmas: List[float] = [0.0, 0.15, 0.35],
        num_repetitions: int = 4,
        shape: Tuple[int, int, int] = (20, 20, 10),
    ) -> List[StabilityMetrics]:
        """Sweep across input data perturbation (additive Gaussian jitter)."""
        logger.info(f"Executing Input Perturbation Sweep across sigmas {perturbation_sigmas}...")
        base_vol = generate_synthetic_benchmark_volume(shape=shape, snr=3.0, seed=42)
        results = []

        for sigma in perturbation_sigmas:
            masks = []
            for rep in range(num_repetitions):
                pert_vol = apply_additive_voxel_perturbation(
                    volume=base_vol,
                    noise_sigma=sigma,
                    seed=100 + rep * 13,
                )
                masks.append(self._run_fchmrf(pert_vol))

            metrics = evaluate_repetition_stability(
                masks=masks,
                condition_name="Input_Perturbation",
                condition_value=float(sigma),
                ground_truth=base_vol.ground_truth,
                nominal_alpha=self.nominal_alpha,
                spatial_mask=base_vol.spatial_mask,
            )
            results.append(metrics)
        return results

    def run_spatial_correlation_sweep(
        self,
        smooth_sigmas: List[float] = [0.0, 1.0, 2.0],
        num_repetitions: int = 4,
        shape: Tuple[int, int, int] = (20, 20, 10),
    ) -> List[StabilityMetrics]:
        """Sweep across background spatial correlation bandwidths."""
        logger.info(f"Executing Spatial Autocorrelation Sweep across sigmas {smooth_sigmas}...")
        dataset = generate_spatial_correlation_sweep_dataset(
            smooth_sigmas=smooth_sigmas,
            num_repetitions=num_repetitions,
            snr=3.0,
            shape=shape,
        )
        results = []
        for sigma, vols in dataset.items():
            masks = [self._run_fchmrf(v) for v in vols]
            metrics = evaluate_repetition_stability(
                masks=masks,
                condition_name="Spatial_Correlation",
                condition_value=float(sigma),
                ground_truth=vols[0].ground_truth,
                nominal_alpha=self.nominal_alpha,
                spatial_mask=vols[0].spatial_mask,
            )
            results.append(metrics)
        return results

    def run_hyperparameter_sensitivity_sweep(
        self,
        lambda_dl_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        num_repetitions: int = 3,
        shape: Tuple[int, int, int] = (20, 20, 10),
    ) -> List[StabilityMetrics]:
        """Sweep across DL prior weight lambda_DL."""
        logger.info(f"Executing Hyperparameter Sensitivity Sweep across lambda_DL {lambda_dl_values}...")
        base_vols = [
            generate_synthetic_benchmark_volume(shape=shape, snr=3.0, seed=42 + i * 10)
            for i in range(num_repetitions)
        ]
        results = []

        for lam in lambda_dl_values:
            masks = [self._run_fchmrf(v, lambda_dl=lam) for v in base_vols]
            metrics = evaluate_repetition_stability(
                masks=masks,
                condition_name="Hyperparam_LambdaDL",
                condition_value=float(lam),
                ground_truth=base_vols[0].ground_truth,
                nominal_alpha=self.nominal_alpha,
                spatial_mask=base_vols[0].spatial_mask,
            )
            results.append(metrics)
        return results

    def run_multi_seed_uncertainty_analysis(
        self,
        num_seeds: int = 8,
        snr: float = 3.0,
        shape: Tuple[int, int, int] = (20, 20, 10),
    ) -> Tuple[StabilityMetrics, np.ndarray, np.ndarray]:
        """Analyze detection probability map and boundary entropy across multiple random seeds."""
        logger.info(f"Executing Multi-Seed Uncertainty Analysis across {num_seeds} random seeds...")
        vols = [
            generate_synthetic_benchmark_volume(shape=shape, snr=snr, seed=1000 + i * 73)
            for i in range(num_seeds)
        ]
        masks = [self._run_fchmrf(v) for v in vols]

        prob_map = compute_detection_probability_map(masks)
        entropy_map, _ = compute_voxel_uncertainty_entropy(prob_map, spatial_mask=vols[0].spatial_mask)

        metrics = evaluate_repetition_stability(
            masks=masks,
            condition_name="Multi_Seed_Analysis",
            condition_value=float(num_seeds),
            ground_truth=vols[0].ground_truth,
            nominal_alpha=self.nominal_alpha,
            spatial_mask=vols[0].spatial_mask,
        )
        return metrics, prob_map, entropy_map

    def run_full_robustness_evaluation(
        self,
        snr_levels: List[float] = [1.5, 2.5, 3.5],
        perturbation_sigmas: List[float] = [0.0, 0.15, 0.35],
        spatial_sigmas: List[float] = [0.0, 1.0, 2.0],
        lambda_dl_values: List[float] = [0.2, 0.5, 0.8],
        num_seeds: int = 6,
        shape: Tuple[int, int, int] = (20, 20, 10),
    ) -> RobustnessEvaluationReport:
        """Run comprehensive multi-faceted robustness evaluation suite."""
        logger.info("Initiating Full Robustness and Uncertainty Evaluation Suite...")

        snr_metrics = self.run_snr_sensitivity_sweep(snr_levels=snr_levels, shape=shape)
        pert_metrics = self.run_input_perturbation_sweep(perturbation_sigmas=perturbation_sigmas, shape=shape)
        spatial_metrics = self.run_spatial_correlation_sweep(smooth_sigmas=spatial_sigmas, shape=shape)
        hyper_metrics = self.run_hyperparameter_sensitivity_sweep(lambda_dl_values=lambda_dl_values, shape=shape)
        seed_metrics, prob_map, ent_map = self.run_multi_seed_uncertainty_analysis(num_seeds=num_seeds, shape=shape)

        report = RobustnessEvaluationReport(
            snr_sweep_metrics=snr_metrics,
            perturbation_sweep_metrics=pert_metrics,
            spatial_corr_metrics=spatial_metrics,
            hyperparam_sweep_metrics=hyper_metrics,
            multi_seed_metrics=seed_metrics,
            detection_probability_map=prob_map,
            uncertainty_entropy_map=ent_map,
            nominal_alpha=self.nominal_alpha,
        )

        logger.info(report.to_ascii_summary())
        return report
