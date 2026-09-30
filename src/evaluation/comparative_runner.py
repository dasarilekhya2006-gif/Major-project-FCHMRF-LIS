"""Comparative experiment orchestrator running all methods under identical experimental conditions."""

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger
from src.evaluation.metrics import (
    AggregatedMetrics,
    EvaluationMetrics,
    aggregate_metrics_across_trials,
    calculate_metrics,
)
from src.evaluation.baselines import (
    AdaptiveFCHMRFLISCorrection,
    AdaptiveROIAssistedFCHMRFLISCorrection,
    BaseCorrectionMethod,
    BenjaminiHochbergCorrection,
    LocalFDRCorrection,
    MethodOutput,
    OriginalFCHMRFLISCorrection,
    ROIAssistedFCHMRFLISCorrection,
)
from src.evaluation.synthetic_benchmarks import SyntheticBenchmarkVolume

logger = get_logger(__name__)


@dataclass
class TrialResult:
    """Trial-level execution result across all evaluated methods."""

    trial_id: int
    volume: SyntheticBenchmarkVolume
    method_outputs: Dict[str, MethodOutput]
    method_metrics: Dict[str, EvaluationMetrics]


@dataclass
class BenchmarkComparisonReport:
    """Master benchmark report compiling trial records and aggregated summary statistics."""

    num_trials: int
    nominal_alpha: float
    methods_evaluated: List[str]
    trial_results: List[TrialResult]
    aggregated_metrics: Dict[str, AggregatedMetrics]

    def to_markdown_table(self) -> str:
        """Render a publication-ready Markdown comparison table with all required metrics."""
        lines = [
            "| Method | FDR (Empirical FDP) | TPR (Recall) | FPR | Precision | Specificity | Localization Error (vox) | Runtime (s) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for name, agg in self.aggregated_metrics.items():
            lines.append(
                f"| **{name}** "
                f"| {agg.mean_fdp:.4f} ± {agg.sem_fdp:.4f} "
                f"| {agg.mean_tpr:.4f} ± {agg.sem_tpr:.4f} "
                f"| {agg.mean_fpr:.4f} ± {agg.sem_fpr:.4f} "
                f"| {agg.mean_precision:.4f} ± {agg.sem_precision:.4f} "
                f"| {agg.mean_specificity:.4f} ± {agg.sem_specificity:.4f} "
                f"| {agg.mean_centroid_error:.2f} ± {agg.sem_centroid_error:.2f} "
                f"| {agg.mean_runtime_seconds:.3f}s |"
            )
        return "\n".join(lines)

    def to_ascii_summary(self) -> str:
        """Generate structured ASCII console summary."""
        hdr = (
            f"\n{'=' * 128}\n"
            f"PHASE 8 RIGOROUS EXPERIMENTAL COMPARISON RESULTS (N={self.num_trials} Trials, alpha={self.nominal_alpha})\n"
            f"{'=' * 128}\n"
            f"{'Method':<32} | {'FDR (FDP)':<13} | {'TPR (Recall)':<13} | {'FPR':<11} | {'Precision':<11} | {'Specificity':<11} | {'Loc Err (vox)':<13} | {'Time (s)':<9}\n"
            f"{'-' * 128}"
        )
        rows = []
        for name, agg in self.aggregated_metrics.items():
            row = (
                f"{name:<32} | "
                f"{agg.mean_fdp:.4f} ± {agg.sem_fdp:.4f} | "
                f"{agg.mean_tpr:.4f} ± {agg.sem_tpr:.4f} | "
                f"{agg.mean_fpr:.4f} ± {agg.sem_fpr:.4f} | "
                f"{agg.mean_precision:.4f} ± {agg.sem_precision:.4f} | "
                f"{agg.mean_specificity:.4f} ± {agg.sem_specificity:.4f} | "
                f"{agg.mean_centroid_error:5.2f} ± {agg.sem_centroid_error:4.2f}  | "
                f"{agg.mean_runtime_seconds:6.3f}s"
            )
            rows.append(row)
        ftr = (
            f"{'-' * 128}\n"
            f"FDR Target: alpha = {self.nominal_alpha}. Methods with empirical FDP <= alpha successfully control the spatial FDR.\n"
            f"{'=' * 128}\n"
        )
        return "\n".join([hdr] + rows + [ftr])

    def export_csv(self, output_path: Union[str, Path]) -> Path:
        """Export tabular summary to CSV."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method",
                "Num_Trials",
                "Nominal_Alpha",
                "FDR_Control_Rate",
                "Mean_FDP",
                "SEM_FDP",
                "Mean_TPR_Power",
                "SEM_TPR_Power",
                "Mean_FPR",
                "SEM_FPR",
                "Mean_Specificity",
                "SEM_Specificity",
                "Mean_Precision",
                "SEM_Precision",
                "Mean_Dice_F1",
                "SEM_Dice_F1",
                "Mean_IoU",
                "SEM_IoU",
                "Mean_MCC",
                "SEM_MCC",
                "Mean_Centroid_Error",
                "SEM_Centroid_Error",
                "Mean_Runtime_Sec",
                "SEM_Runtime_Sec",
                "Mean_Peak_RAM_MB",
                "SEM_Peak_RAM_MB",
            ])
            for name, agg in self.aggregated_metrics.items():
                writer.writerow([
                    name,
                    agg.num_trials,
                    agg.nominal_alpha,
                    agg.fdr_control_rate,
                    agg.mean_fdp,
                    agg.sem_fdp,
                    agg.mean_tpr,
                    agg.sem_tpr,
                    agg.mean_fpr,
                    agg.sem_fpr,
                    agg.mean_specificity,
                    agg.sem_specificity,
                    agg.mean_precision,
                    agg.sem_precision,
                    agg.mean_f1,
                    agg.sem_f1,
                    agg.mean_iou,
                    agg.sem_iou,
                    agg.mean_mcc,
                    agg.sem_mcc,
                    agg.mean_centroid_error,
                    agg.sem_centroid_error,
                    agg.mean_runtime_seconds,
                    agg.sem_runtime_seconds,
                    agg.mean_peak_memory_mb,
                    agg.sem_peak_memory_mb,
                ])
        logger.info(f"Saved benchmark CSV summary to {out_p}")
        return out_p


class ComparativeExperimentRunner:
    """Executes multi-method benchmarks under identical experimental datasets."""

    def __init__(
        self,
        nominal_alpha: float = 0.05,
        methods: Optional[List[BaseCorrectionMethod]] = None,
        optimized_parameters: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ) -> None:
        self.nominal_alpha = nominal_alpha
        self.device = device

        if methods is not None:
            self.methods = methods
        else:
            # Default: instantiate all 5 requested baseline and proposed methods
            self.methods = [
                BenjaminiHochbergCorrection(alpha=nominal_alpha),
                OriginalFCHMRFLISCorrection(alpha=nominal_alpha),
                ROIAssistedFCHMRFLISCorrection(alpha=nominal_alpha, lambda_dl=0.5),
                AdaptiveFCHMRFLISCorrection(
                    alpha=nominal_alpha,
                    optimized_parameters=optimized_parameters,
                ),
                AdaptiveROIAssistedFCHMRFLISCorrection(
                    alpha=nominal_alpha,
                    optimized_parameters=optimized_parameters,
                ),
            ]

    def run_benchmark(
        self,
        dataset: List[SyntheticBenchmarkVolume],
    ) -> BenchmarkComparisonReport:
        """Run all methods sequentially across all volumes in the dataset."""
        logger.info(
            f"Initiating rigorous comparative benchmark: {len(self.methods)} methods "
            f"across {len(dataset)} trials at alpha={self.nominal_alpha}..."
        )

        all_trial_results: List[TrialResult] = []
        method_metrics_collector: Dict[str, List[EvaluationMetrics]] = {
            m.name: [] for m in self.methods
        }

        for idx, volume in enumerate(dataset):
            logger.info(f"Processing Monte Carlo Trial {idx + 1}/{len(dataset)} (SNR={volume.snr:.1f})...")
            trial_outputs: Dict[str, MethodOutput] = {}
            trial_metrics: Dict[str, EvaluationMetrics] = {}

            for method in self.methods:
                # Execute under identical volume inputs
                output = method.run(
                    z_scores=volume.z_scores,
                    p_values=volume.p_values,
                    dl_prior=volume.dl_prior,
                    spatial_mask=volume.spatial_mask,
                )
                trial_outputs[method.name] = output

                # Calculate performance metrics against ground truth
                metrics = calculate_metrics(
                    pred_mask=output.significant_mask,
                    ground_truth=volume.ground_truth,
                    method_name=method.name,
                    nominal_alpha=self.nominal_alpha,
                    runtime_seconds=output.runtime_seconds,
                    peak_memory_mb=output.peak_memory_mb,
                    spatial_mask=volume.spatial_mask,
                )
                trial_metrics[method.name] = metrics
                method_metrics_collector[method.name].append(metrics)

            all_trial_results.append(
                TrialResult(
                    trial_id=volume.trial_id,
                    volume=volume,
                    method_outputs=trial_outputs,
                    method_metrics=trial_metrics,
                )
            )

        # Compute aggregated statistics
        aggregated: Dict[str, AggregatedMetrics] = {}
        for m in self.methods:
            metrics_list = method_metrics_collector[m.name]
            agg = aggregate_metrics_across_trials(metrics_list)
            aggregated[m.name] = agg

        report = BenchmarkComparisonReport(
            num_trials=len(dataset),
            nominal_alpha=self.nominal_alpha,
            methods_evaluated=[m.name for m in self.methods],
            trial_results=all_trial_results,
            aggregated_metrics=aggregated,
        )

        logger.info(report.to_ascii_summary())
        return report
