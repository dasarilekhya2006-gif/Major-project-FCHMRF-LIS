"""Systematic ablation study engine decomposing component contributions in fcHMRF-LIS."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger
from src.core.seed import set_seed
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.data.schemas import SyntheticBenchmarkSample
from src.deep_learning.predictor import ROIPriorPredictor
from src.evaluation.metrics import EvaluationMetrics, calculate_metrics
from src.evaluation.statistical_testing import (
    PairedTestResult,
    apply_holm_bonferroni,
    paired_statistical_comparison,
)
from src.fchmrf_lis.fchmrf_model import FCHMRFModel
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.statistical_analysis.classical_fdr import benjamini_hochberg
from src.statistical_analysis.local_fdr import efron_locfdr
from src.adaptive_selection.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
)
from src.adaptive_selection.search_space import create_fchmrf_search_space
from src.adaptive_selection.objectives import ValidationFDRBenchmarkObjective

logger = get_logger(__name__)


class AblationArm(str, Enum):
    """The four core ablation study conditions specified by the research protocol."""

    ARM_A_FCHMRF_ALONE = "Arm_A_fcHMRF_alone"
    ARM_B_ROI_FCHMRF = "Arm_B_ROI_fcHMRF"
    ARM_C_ADAPTIVE_FCHMRF = "Arm_C_Adaptive_fcHMRF"
    ARM_D_ROI_ADAPTIVE_FCHMRF = "Arm_D_ROI_Adaptive_fcHMRF"  # Full Proposed Method


@dataclass
class ComponentContribution:
    """Quantitative marginal contribution attributable to each pipeline component.

    Decomposition:
      Let M(x) be the mean evaluation metric (e.g. Power, Dice, Precision, Runtime).
      - Marginal Spatial Effect: Delta_spatial = M(Arm A) - M(BH-FDR)
      - Marginal Deep ROI Prior Effect: Delta_ROI = M(Arm B) - M(Arm A)
      - Marginal Adaptive Tuning Effect: Delta_Adaptive = M(Arm C) - M(Arm A)
      - Synergy / Interaction Effect: Delta_Synergy = M(Arm D) - [M(Arm A) + Delta_ROI + Delta_Adaptive]
      - Total Proposed Improvement over fcHMRF alone: Delta_Net = M(Arm D) - M(Arm A)
    """

    metric_name: str
    baseline_bh: float
    arm_a_fchmrf_alone: float
    arm_b_roi_fchmrf: float
    arm_c_adaptive_fchmrf: float
    arm_d_roi_adaptive_fchmrf: float

    # Marginal contributions
    delta_spatial_model: float = 0.0
    delta_roi_prior: float = 0.0
    delta_adaptive_tuning: float = 0.0
    delta_synergy_interaction: float = 0.0
    delta_total_improvement: float = 0.0
    percent_improvement_over_original: float = 0.0

    def compute_decomposition(self) -> None:
        """Compute the additive component attribution breakdown."""
        self.delta_spatial_model = self.arm_a_fchmrf_alone - self.baseline_bh
        self.delta_roi_prior = self.arm_b_roi_fchmrf - self.arm_a_fchmrf_alone
        self.delta_adaptive_tuning = self.arm_c_adaptive_fchmrf - self.arm_a_fchmrf_alone
        linear_expected = self.arm_a_fchmrf_alone + self.delta_roi_prior + self.delta_adaptive_tuning
        self.delta_synergy_interaction = self.arm_d_roi_adaptive_fchmrf - linear_expected
        self.delta_total_improvement = self.arm_d_roi_adaptive_fchmrf - self.arm_a_fchmrf_alone

        if abs(self.arm_a_fchmrf_alone) > 1e-9:
            self.percent_improvement_over_original = (
                self.delta_total_improvement / self.arm_a_fchmrf_alone
            ) * 100.0
        else:
            self.percent_improvement_over_original = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_bh": self.baseline_bh,
            "arm_a_fchmrf_alone": self.arm_a_fchmrf_alone,
            "arm_b_roi_fchmrf": self.arm_b_roi_fchmrf,
            "arm_c_adaptive_fchmrf": self.arm_c_adaptive_fchmrf,
            "arm_d_roi_adaptive_fchmrf": self.arm_d_roi_adaptive_fchmrf,
            "delta_spatial_model": self.delta_spatial_model,
            "delta_roi_prior": self.delta_roi_prior,
            "delta_adaptive_tuning": self.delta_adaptive_tuning,
            "delta_synergy_interaction": self.delta_synergy_interaction,
            "delta_total_improvement": self.delta_total_improvement,
            "percent_improvement_over_original": self.percent_improvement_over_original,
        }


@dataclass
class AblationStudyResult:
    """Master container for complete ablation study results across runs."""

    snr: float
    n_seeds: int
    raw_metrics_by_arm: Dict[str, List[EvaluationMetrics]]
    mean_metrics_by_arm: Dict[str, Dict[str, float]]
    std_metrics_by_arm: Dict[str, Dict[str, float]]
    component_attributions: Dict[str, ComponentContribution]
    statistical_tests: List[PairedTestResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snr": self.snr,
            "n_seeds": self.n_seeds,
            "mean_metrics": self.mean_metrics_by_arm,
            "std_metrics": self.std_metrics_by_arm,
            "component_attributions": {
                k: v.to_dict() for k, v in self.component_attributions.items()
            },
            "statistical_tests": [t.to_dict() for t in self.statistical_tests],
        }


class AblationStudyRunner:
    """Executes controlled ablation experiments isolating pipeline components."""

    def __init__(
        self,
        nominal_alpha: float = 0.05,
        default_beta: float = 0.8,
        default_lambda_dl: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.nominal_alpha = nominal_alpha
        self.default_beta = default_beta
        self.default_lambda_dl = default_lambda_dl
        self.device = device
        self.prior_predictor = ROIPriorPredictor(device=device)

    def run_single_ablation_sample(
        self,
        sample: SyntheticBenchmarkSample,
        seed: int = 42,
    ) -> Dict[str, EvaluationMetrics]:
        """Execute all 4 ablation arms plus reference baseline on a single sample."""
        set_seed(seed)
        z_scores = sample.z_map.astype(np.float32)
        ground_truth = sample.ground_truth.astype(bool)
        affine = np.eye(4, dtype=np.float64)

        from scipy.stats import norm
        p_values = 1.0 - norm.cdf(z_scores)
        p_values = np.clip(p_values, 1e-15, 1.0 - 1e-15)

        results: Dict[str, EvaluationMetrics] = {}

        # ---------------------------------------------------------------------
        # Reference Baseline: Standard Benjamini-Hochberg (BH-FDR)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        bh_sig, _, _ = benjamini_hochberg(p_values.ravel(), alpha=self.nominal_alpha)
        bh_mask = bh_sig.reshape(z_scores.shape)
        results["Baseline_BH"] = calculate_metrics(
            pred_mask=bh_mask,
            ground_truth=ground_truth,
            method_name="Baseline_BH",
            nominal_alpha=self.nominal_alpha,
            runtime_seconds=time.perf_counter() - t0,
        )

        # ---------------------------------------------------------------------
        # Deep Learning 3D U-Net Prior Map (for arms B and D)
        # ---------------------------------------------------------------------
        t_unet = time.perf_counter()
        prior_map = self.prior_predictor.predict(z_scores)
        roi_mask = (prior_map >= 0.5).astype(bool)
        if not np.any(roi_mask):
            thresh = np.percentile(prior_map, 85)
            roi_mask = (prior_map >= thresh).astype(bool)
        unet_duration = time.perf_counter() - t_unet

        # ---------------------------------------------------------------------
        # Arm A: fcHMRF-LIS alone (Published method: No ROI, lambda_dl = 0.0, fixed beta)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        params_a = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            beta=self.default_beta,
            lambda_dl=0.0,
            alpha=self.nominal_alpha,
            max_iterations=25,
            tolerance=1e-4,
        )
        model_a = FCHMRFModel(params=params_a)
        res_a = model_a.fit_predict(
            z_scores=z_scores,
            dl_prior=None,
            spatial_mask=np.ones_like(z_scores, dtype=bool),
            affine=affine,
        )
        results[AblationArm.ARM_A_FCHMRF_ALONE.value] = calculate_metrics(
            pred_mask=res_a.significant_mask,
            ground_truth=ground_truth,
            method_name=AblationArm.ARM_A_FCHMRF_ALONE.value,
            nominal_alpha=self.nominal_alpha,
            runtime_seconds=time.perf_counter() - t0,
        )

        # ---------------------------------------------------------------------
        # Arm B: ROI + fcHMRF-LIS (Deep Prior active, fixed beta & lambda_dl)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        params_b = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            beta=self.default_beta,
            lambda_dl=self.default_lambda_dl,
            alpha=self.nominal_alpha,
            max_iterations=25,
            tolerance=1e-4,
        )
        model_b = FCHMRFModel(params=params_b)
        res_b = model_b.fit_predict(
            z_scores=z_scores,
            dl_prior=prior_map,
            spatial_mask=roi_mask,
            affine=affine,
        )
        results[AblationArm.ARM_B_ROI_FCHMRF.value] = calculate_metrics(
            pred_mask=res_b.significant_mask,
            ground_truth=ground_truth,
            method_name=AblationArm.ARM_B_ROI_FCHMRF.value,
            nominal_alpha=self.nominal_alpha,
            runtime_seconds=(time.perf_counter() - t0) + unet_duration,
        )

        # ---------------------------------------------------------------------
        # Arm C: Adaptive fcHMRF-LIS (BO tuning beta, but NO ROI / lambda_dl = 0)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        search_space_c = create_fchmrf_search_space(
            include_spatial_bandwidths=False,
            include_lambda_dl=False,
        )
        obj_c = ValidationFDRBenchmarkObjective(
            z_scores_val=z_scores,
            ground_truth_val=ground_truth,
            spatial_mask_val=np.ones_like(z_scores, dtype=bool),
            dl_prior_val=None,
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            target_alpha=self.nominal_alpha,
            max_iterations=15,
        )
        opt_c = BayesianOptimizer(
            search_space=search_space_c,
            objective=obj_c,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            seed=seed,
        )
        opt_res_c = opt_c.optimize(n_iterations=5, n_initial=2)
        tuned_beta_c = float(opt_res_c.best_parameters.get("beta", self.default_beta))

        params_c = FCHMRFParameters(
            method_mode=MethodMode.ORIGINAL_FCHMRF,
            beta=tuned_beta_c,
            lambda_dl=0.0,
            alpha=self.nominal_alpha,
            max_iterations=25,
            tolerance=1e-4,
        )
        model_c = FCHMRFModel(params=params_c)
        res_c = model_c.fit_predict(
            z_scores=z_scores,
            dl_prior=None,
            spatial_mask=np.ones_like(z_scores, dtype=bool),
            affine=affine,
        )
        results[AblationArm.ARM_C_ADAPTIVE_FCHMRF.value] = calculate_metrics(
            pred_mask=res_c.significant_mask,
            ground_truth=ground_truth,
            method_name=AblationArm.ARM_C_ADAPTIVE_FCHMRF.value,
            nominal_alpha=self.nominal_alpha,
            runtime_seconds=time.perf_counter() - t0,
        )

        # ---------------------------------------------------------------------
        # Arm D: ROI + Adaptive fcHMRF-LIS (Full Proposed Method: Joint Tuning & Deep Prior)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        search_space_d = create_fchmrf_search_space(
            include_spatial_bandwidths=False,
            include_lambda_dl=True,
        )
        obj_d = ValidationFDRBenchmarkObjective(
            z_scores_val=z_scores,
            ground_truth_val=ground_truth,
            spatial_mask_val=roi_mask,
            dl_prior_val=prior_map,
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            target_alpha=self.nominal_alpha,
            max_iterations=15,
        )
        opt_d = BayesianOptimizer(
            search_space=search_space_d,
            objective=obj_d,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            seed=seed,
        )
        opt_res_d = opt_d.optimize(n_iterations=5, n_initial=2)
        tuned_beta_d = float(opt_res_d.best_parameters.get("beta", self.default_beta))
        tuned_lambda_d = float(opt_res_d.best_parameters.get("lambda_dl", self.default_lambda_dl))

        params_d = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            beta=tuned_beta_d,
            lambda_dl=tuned_lambda_d,
            alpha=self.nominal_alpha,
            max_iterations=25,
            tolerance=1e-4,
        )
        model_d = FCHMRFModel(params=params_d)
        res_d = model_d.fit_predict(
            z_scores=z_scores,
            dl_prior=prior_map,
            spatial_mask=roi_mask,
            affine=affine,
        )
        results[AblationArm.ARM_D_ROI_ADAPTIVE_FCHMRF.value] = calculate_metrics(
            pred_mask=res_d.significant_mask,
            ground_truth=ground_truth,
            method_name=AblationArm.ARM_D_ROI_ADAPTIVE_FCHMRF.value,
            nominal_alpha=self.nominal_alpha,
            runtime_seconds=(time.perf_counter() - t0) + unet_duration,
        )

        return results

    def run_multi_seed_ablation_study(
        self,
        snr: float = 2.0,
        seeds: List[int] = [42, 101, 202, 303, 404],
        grid_size: Tuple[int, int, int] = (32, 32, 16),
    ) -> AblationStudyResult:
        """Run replicated Monte Carlo ablation study across multiple random seeds."""
        logger.info(f"Starting Multi-Seed Ablation Study (SNR={snr}, N={len(seeds)} seeds)...")

        arms = [
            "Baseline_BH",
            AblationArm.ARM_A_FCHMRF_ALONE.value,
            AblationArm.ARM_B_ROI_FCHMRF.value,
            AblationArm.ARM_C_ADAPTIVE_FCHMRF.value,
            AblationArm.ARM_D_ROI_ADAPTIVE_FCHMRF.value,
        ]

        raw_metrics: Dict[str, List[EvaluationMetrics]] = {arm: [] for arm in arms}

        for seed in seeds:
            logger.info(f"Generating synthetic sample with seed={seed} (grid={grid_size}, SNR={snr})...")
            sample = generate_synthetic_spatial_dataset(
                grid_size=grid_size,
                snr=snr,
                num_clusters=2,
                cluster_radius_range=(3, 4),
                seed=seed,
            )
            step_res = self.run_single_ablation_sample(sample=sample, seed=seed)
            for arm in arms:
                raw_metrics[arm].append(step_res[arm])

        # Compute Means & Standard Deviations
        eval_keys = ["tpr", "fdp", "f1_score", "precision", "mcc", "runtime_seconds"]
        mean_metrics: Dict[str, Dict[str, float]] = {arm: {} for arm in arms}
        std_metrics: Dict[str, Dict[str, float]] = {arm: {} for arm in arms}

        for arm in arms:
            for k in eval_keys:
                vals = [getattr(m, k) for m in raw_metrics[arm]]
                mean_metrics[arm][k] = float(np.mean(vals))
                std_metrics[arm][k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # Compute Component Attributions
        attributions: Dict[str, ComponentContribution] = {}
        for k in eval_keys:
            contrib = ComponentContribution(
                metric_name=k,
                baseline_bh=mean_metrics["Baseline_BH"][k],
                arm_a_fchmrf_alone=mean_metrics[AblationArm.ARM_A_FCHMRF_ALONE.value][k],
                arm_b_roi_fchmrf=mean_metrics[AblationArm.ARM_B_ROI_FCHMRF.value][k],
                arm_c_adaptive_fchmrf=mean_metrics[AblationArm.ARM_C_ADAPTIVE_FCHMRF.value][k],
                arm_d_roi_adaptive_fchmrf=mean_metrics[AblationArm.ARM_D_ROI_ADAPTIVE_FCHMRF.value][k],
            )
            contrib.compute_decomposition()
            attributions[k] = contrib

        # Perform Statistical Significance Tests comparing Arm D vs each ablation arm
        stat_tests: List[PairedTestResult] = []
        comp_arms = [
            ("Baseline_BH", "BH-FDR"),
            (AblationArm.ARM_A_FCHMRF_ALONE.value, "Arm A (fcHMRF alone)"),
            (AblationArm.ARM_B_ROI_FCHMRF.value, "Arm B (ROI + fcHMRF)"),
            (AblationArm.ARM_C_ADAPTIVE_FCHMRF.value, "Arm C (Adaptive fcHMRF)"),
        ]

        for base_arm_key, base_arm_label in comp_arms:
            for metric_key in ["tpr", "fdp", "f1_score"]:
                base_vals = [getattr(m, metric_key) for m in raw_metrics[base_arm_key]]
                cand_vals = [getattr(m, metric_key) for m in raw_metrics[AblationArm.ARM_D_ROI_ADAPTIVE_FCHMRF.value]]
                t_res = paired_statistical_comparison(
                    baseline_values=base_vals,
                    candidate_values=cand_vals,
                    metric_name=metric_key,
                    baseline_name=base_arm_label,
                    candidate_name="Arm D (Full Proposed)",
                    alpha=self.nominal_alpha,
                )
                stat_tests.append(t_res)

        # Multiple comparisons correction
        stat_tests = apply_holm_bonferroni(stat_tests, alpha=self.nominal_alpha)

        result = AblationStudyResult(
            snr=snr,
            n_seeds=len(seeds),
            raw_metrics_by_arm=raw_metrics,
            mean_metrics_by_arm=mean_metrics,
            std_metrics_by_arm=std_metrics,
            component_attributions=attributions,
            statistical_tests=stat_tests,
        )
        return result
