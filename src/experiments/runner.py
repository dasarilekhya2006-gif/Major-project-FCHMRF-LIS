"""Core execution engine for automated research experiments."""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import nibabel as nib
import numpy as np
from scipy.stats import norm

from src.core.logging import get_logger
from src.core.seed import set_seed
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.data.loaders import load_nifti
from src.deep_learning.predictor import ROIPriorPredictor
from src.evaluation.metrics import calculate_metrics
from src.fchmrf_lis.fchmrf_model import FCHMRFModel
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.statistical_analysis.classical_fdr import benjamini_hochberg
from src.statistical_analysis.local_fdr import efron_locfdr
from src.adaptive_selection.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
)
from src.adaptive_selection.search_space import create_fchmrf_search_space
from src.adaptive_selection.objectives import (
    ObjectiveType,
    ValidationFDRBenchmarkObjective,
)

from src.experiments.schemas import (
    ExperimentConfig,
    ExperimentResult,
    get_current_system_metadata,
)
from src.experiments.plots import (
    plot_single_experiment_diagnostic,
    plot_suite_summary_figures,
)
from src.experiments.tables import (
    export_suite_summary_csv,
    generate_comparison_latex_table,
    generate_comparison_markdown_table,
)

logger = get_logger(__name__)


class ExperimentRunner:
    """Executes single experiments or suites systematically with strict reproducibility."""

    def __init__(
        self,
        base_output_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> None:
        self.base_output_dir = Path(base_output_dir or "outputs/automated_experiments")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.prior_predictor = ROIPriorPredictor(device=device)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute a single reproducible experiment according to its configuration.

        Records:
          - dataset
          - preprocessing configuration
          - U-Net configuration
          - statistical test
          - fcHMRF parameters
          - adaptive optimization settings
          - random seed
          - evaluation metrics
          - runtime breakdown and total runtime
          - all output artifact locations
        """
        start_total = time.perf_counter()
        runtimes: Dict[str, float] = {}

        # 1. Deterministic Multi-Framework Seed Initialization
        set_seed(config.seed)
        logger.info(f"Starting Experiment: {config.experiment_id} ({config.name}) [seed={config.seed}]")

        exp_dir = Path(config.output_dir) / config.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_locations: Dict[str, str] = {}

        try:
            # -----------------------------------------------------------------
            # Stage 1: Dataset Generation / Loading
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            ground_truth: Optional[np.ndarray] = None
            affine = np.eye(4, dtype=np.float64)

            if config.dataset.dataset_type == "synthetic":
                sample = generate_synthetic_spatial_dataset(
                    grid_size=config.dataset.grid_size,
                    snr=config.dataset.snr,
                    num_clusters=config.dataset.num_clusters,
                    cluster_radius_range=(
                        max(2, config.dataset.cluster_radius - 1),
                        config.dataset.cluster_radius + 1,
                    ),
                    smooth_fwhm_voxels=config.dataset.spatial_smoothing_sigma,
                    seed=config.seed,
                )
                z_scores = sample.z_map.astype(np.float32)
                ground_truth = sample.ground_truth.astype(bool)
                mri_volume = (sample.signal_field + sample.noise_field).astype(np.float32)
            else:
                if not config.dataset.input_path:
                    raise ValueError("NIfTI input path required for real dataset type.")
                loaded = load_nifti(config.dataset.input_path)
                mri_volume = loaded.data.astype(np.float32)
                affine = loaded.affine
                z_scores = mri_volume.copy()
                if config.dataset.ground_truth_path:
                    loaded_gt = load_nifti(config.dataset.ground_truth_path)
                    ground_truth = (loaded_gt.data > 0).astype(bool)

            runtimes["data_loading"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # Stage 2: Modular Preprocessing
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            if config.dataset.dataset_type != "synthetic":
                if config.preprocessing.norm_method == "zscore":
                    mean_val = np.mean(mri_volume)
                    std_val = np.std(mri_volume)
                    z_scores = (mri_volume - mean_val) / (std_val + 1e-8)
                elif config.preprocessing.norm_method == "robust":
                    med_val = np.median(mri_volume)
                    iqr_val = np.percentile(mri_volume, 75) - np.percentile(mri_volume, 25)
                    z_scores = (mri_volume - med_val) / (iqr_val / 1.349 + 1e-8)
            preprocessed_volume = z_scores.copy()
            runtimes["preprocessing"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # Stage 3: Deep Learning 3D U-Net ROI Prior
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            prior_map: Optional[np.ndarray] = None
            roi_mask: Optional[np.ndarray] = None

            if config.unet.enabled:
                prior_map = self.prior_predictor.predict(z_scores)
                roi_mask = (prior_map >= config.unet.roi_threshold).astype(bool)
                # If threshold excludes everything, relax to top 15% active voxels
                if not np.any(roi_mask):
                    thresh = np.percentile(prior_map, 85)
                    roi_mask = (prior_map >= thresh).astype(bool)
            runtimes["unet_inference"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # Stage 4: Voxel-Level Statistical Testing
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            analysis_mask = np.ones_like(z_scores, dtype=bool)
            if config.statistical_test.restrict_to_roi and roi_mask is not None:
                analysis_mask = roi_mask.copy()

            # Generate p-values from standard normal null
            if config.statistical_test.alternative == "greater":
                p_values = 1.0 - norm.cdf(z_scores)
            else:
                p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))
            p_values = np.clip(p_values, 1e-15, 1.0 - 1e-15)
            runtimes["statistical_testing"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # Stage 5: Adaptive Parameter Selection (Optional)
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            tuned_beta = config.fchmrf.beta
            tuned_lambda = config.fchmrf.lambda_dl

            if config.adaptive.enabled and config.adaptive.algorithm == "bayesian_optimization":
                logger.info("Executing adaptive Bayesian Optimization for fcHMRF hyperparameters...")
                search_space = create_fchmrf_search_space()
                objective = ValidationFDRBenchmarkObjective(
                    z_scores_val=z_scores,
                    ground_truth_val=ground_truth,
                    spatial_mask_val=analysis_mask,
                    dl_prior_val=prior_map,
                    target_alpha=config.fchmrf.alpha,
                )
                acq_func = AcquisitionFunction.EXPECTED_IMPROVEMENT
                if config.adaptive.acquisition_function == "upper_confidence_bound":
                    acq_func = AcquisitionFunction.UPPER_CONFIDENCE_BOUND
                elif config.adaptive.acquisition_function == "probability_of_improvement":
                    acq_func = AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT

                opt = BayesianOptimizer(
                    search_space=search_space,
                    objective=objective,
                    acquisition_function=acq_func,
                    seed=config.seed,
                )
                opt_res = opt.optimize(
                    n_iterations=config.adaptive.n_iterations,
                    n_initial=config.adaptive.n_initial,
                )
                tuned_beta = float(opt_res.best_parameters.get("beta", tuned_beta))
                tuned_lambda = float(opt_res.best_parameters.get("lambda_dl", tuned_lambda))
                logger.info(f"Selected parameters via BO: beta={tuned_beta:.3f}, lambda_dl={tuned_lambda:.3f}")
            runtimes["adaptive_optimization"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # Stage 6: fcHMRF-LIS Spatial Inference & FDR Control
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            lis_map: Optional[np.ndarray] = None
            sig_mask = np.zeros_like(z_scores, dtype=bool)
            posterior_prob: Optional[np.ndarray] = None

            mode = config.fchmrf.method_mode.lower()

            if mode == "baseline_bh":
                # Classical Benjamini-Hochberg (BH-FDR)
                bh_sig_vec = benjamini_hochberg(p_values[analysis_mask], alpha=config.fchmrf.alpha)
                sig_mask[analysis_mask] = bh_sig_vec
                lis_map = p_values.copy()
            elif mode == "baseline_locfdr":
                # Efron's Empirical Null Local FDR
                lfdr_res = efron_locfdr(z_scores[analysis_mask], alpha=config.fchmrf.alpha)
                sig_mask[analysis_mask] = lfdr_res.significant_mask
                lis_map = np.ones_like(z_scores)
                lis_map[analysis_mask] = lfdr_res.fdr_values
            else:
                # fcHMRF-LIS Engine (Original or Proposed Adaptive)
                fchmrf_mode = MethodMode.ORIGINAL_FCHMRF
                use_lambda = 0.0
                if "proposed" in mode or "adaptive" in mode:
                    fchmrf_mode = MethodMode.ADAPTIVE_DL_FCHMRF
                    use_lambda = tuned_lambda

                model_params = FCHMRFParameters(
                    method_mode=fchmrf_mode,
                    beta=tuned_beta,
                    sigma_spatial=config.fchmrf.sigma_spatial,
                    sigma_intensity=config.fchmrf.sigma_intensity,
                    lambda_dl=use_lambda,
                    alpha=config.fchmrf.alpha,
                    max_iterations=config.fchmrf.max_iterations,
                    tolerance=config.fchmrf.tolerance,
                    damping=config.fchmrf.damping_tau,
                )
                model = FCHMRFModel(params=model_params)
                fchmrf_out = model.fit_predict(
                    z_scores=z_scores,
                    dl_prior=prior_map if fchmrf_mode == MethodMode.ADAPTIVE_DL_FCHMRF else None,
                    spatial_mask=analysis_mask,
                    affine=affine,
                )
                lis_map = fchmrf_out.lis_map
                sig_mask = fchmrf_out.significant_mask.astype(bool)
                posterior_prob = fchmrf_out.posterior_prob

            runtimes["fchmrf_inference"] = time.perf_counter() - t0

            # -----------------------------------------------------------------
            # Stage 7: Quantitative Performance Evaluation
            # -----------------------------------------------------------------
            total_elapsed = time.perf_counter() - start_total
            metrics_dict: Dict[str, Any] = {}

            if ground_truth is not None:
                eval_m = calculate_metrics(
                    pred_mask=sig_mask,
                    ground_truth=ground_truth,
                    method_name=config.fchmrf.method_mode,
                    nominal_alpha=config.fchmrf.alpha,
                    runtime_seconds=total_elapsed,
                    spatial_mask=analysis_mask,
                )
                metrics_dict = eval_m.to_dict()
                # Ensure alias compatibility
                metrics_dict["empirical_fdr"] = eval_m.fdp
                metrics_dict["tpr"] = eval_m.tpr
                metrics_dict["dice"] = eval_m.f1_score
                metrics_dict["num_discoveries"] = int(np.sum(sig_mask))
            else:
                metrics_dict = {
                    "method_name": config.fchmrf.method_mode,
                    "nominal_alpha": config.fchmrf.alpha,
                    "num_discoveries": int(np.sum(sig_mask)),
                    "total_analyzed_voxels": int(np.sum(analysis_mask)),
                    "runtime_seconds": total_elapsed,
                }

            # -----------------------------------------------------------------
            # Stage 8: Serialization & Publication Artifact Generation
            # -----------------------------------------------------------------
            # 1. Save NIfTI files
            nib.save(nib.Nifti1Image(preprocessed_volume, affine), str(exp_dir / "preprocessed.nii.gz"))
            output_locations["nifti_preprocessed"] = str(exp_dir / "preprocessed.nii.gz")

            if roi_mask is not None:
                nib.save(nib.Nifti1Image(roi_mask.astype(np.uint8), affine), str(exp_dir / "roi_mask.nii.gz"))
                output_locations["nifti_roi_mask"] = str(exp_dir / "roi_mask.nii.gz")

            nib.save(nib.Nifti1Image(z_scores, affine), str(exp_dir / "z_scores.nii.gz"))
            output_locations["nifti_z_scores"] = str(exp_dir / "z_scores.nii.gz")

            if lis_map is not None:
                nib.save(nib.Nifti1Image(lis_map.astype(np.float32), affine), str(exp_dir / "lis_map.nii.gz"))
                output_locations["nifti_lis_map"] = str(exp_dir / "lis_map.nii.gz")

            nib.save(nib.Nifti1Image(sig_mask.astype(np.uint8), affine), str(exp_dir / "significant_mask.nii.gz"))
            output_locations["nifti_significant_mask"] = str(exp_dir / "significant_mask.nii.gz")

            # 2. Save Configurations
            cfg_json = exp_dir / "config.json"
            cfg_yaml = exp_dir / "config.yaml"
            config.save_json(cfg_json)
            config.save_yaml(cfg_yaml)
            output_locations["json_config"] = str(cfg_json)
            output_locations["yaml_config"] = str(cfg_yaml)

            # 3. Save Diagnostic Plot
            diag_plot_path = exp_dir / "diagnostic_summary.png"
            plot_single_experiment_diagnostic(
                result=ExperimentResult(
                    experiment_id=config.experiment_id,
                    config=config,
                    total_runtime=total_elapsed,
                    metrics=metrics_dict,
                ),
                mri_volume=mri_volume,
                roi_mask=prior_map if prior_map is not None else roi_mask,
                z_scores=z_scores,
                lis_map=lis_map,
                sig_mask=sig_mask,
                ground_truth=ground_truth,
                output_path=diag_plot_path,
            )
            output_locations["diagnostic_plot"] = str(diag_plot_path)

            # 4. Save Per-Experiment Metrics CSV
            csv_path = exp_dir / "metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in metrics_dict.items():
                    writer.writerow([k, v])
            output_locations["csv_metrics"] = str(csv_path)

            # Assemble Result
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                status="completed",
                runtime_breakdown=runtimes,
                total_runtime=total_elapsed,
                metrics=metrics_dict,
                output_locations=output_locations,
                system_metadata=get_current_system_metadata(),
            )

            # Save Result JSON
            res_json = exp_dir / "result.json"
            result.save_json(res_json)
            output_locations["json_result"] = str(res_json)

            logger.info(
                f"Completed Experiment {config.experiment_id} in {total_elapsed:.2f}s | "
                f"FDR: {metrics_dict.get('empirical_fdr', 'N/A')} | "
                f"TPR: {metrics_dict.get('tpr', 'N/A')} | "
                f"Dice: {metrics_dict.get('dice', 'N/A')}"
            )
            return result

        except Exception as e:
            total_elapsed = time.perf_counter() - start_total
            logger.error(f"Experiment {config.experiment_id} failed: {e}", exc_info=True)
            res = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                status="failed",
                error=str(e),
                total_runtime=total_elapsed,
                runtime_breakdown=runtimes,
                system_metadata=get_current_system_metadata(),
            )
            res.save_json(exp_dir / "result.json")
            return res

    def run_suite(
        self,
        configs: List[ExperimentConfig],
        suite_output_dir: Optional[Union[str, Path]] = None,
    ) -> List[ExperimentResult]:
        """Execute a batch matrix of research experiments systematically.

        Generates:
          - CSV results spreadsheet: suite_summary.csv
          - JSON suite summary: suite_summary.json
          - Formatted comparison tables: comparison_table.md, comparison_table.tex
          - Comparative plots:
              * suite_metrics_comparison_bar.png
              * suite_runtime_breakdown_bar.png
              * suite_fdr_power_calibration.png
        """
        suite_dir = Path(suite_output_dir or self.base_output_dir)
        suite_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"============================================================")
        logger.info(f"STARTING AUTOMATED EXPERIMENT SUITE: {len(configs)} Configurations")
        logger.info(f"Suite Destination: {suite_dir}")
        logger.info(f"============================================================")

        results: List[ExperimentResult] = []
        for idx, cfg in enumerate(configs, 1):
            logger.info(f"\n--- Running Experiment [{idx}/{len(configs)}]: {cfg.experiment_id} ---")
            cfg.output_dir = str(suite_dir)
            res = self.run_experiment(cfg)
            results.append(res)

        # 1. Export Master Summary CSV
        summary_csv = suite_dir / "suite_summary.csv"
        export_suite_summary_csv(results, summary_csv)
        logger.info(f"Exported Master Suite Summary CSV to {summary_csv}")

        # 2. Export Master Summary JSON
        summary_json = suite_dir / "suite_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        # 3. Export Comparison Tables (Markdown & LaTeX)
        md_table = generate_comparison_markdown_table(results)
        md_path = suite_dir / "comparison_table.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_table + "\n")

        tex_table = generate_comparison_latex_table(results)
        tex_path = suite_dir / "comparison_table.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_table + "\n")

        # 4. Generate Aggregated Comparison Plots
        plots = plot_suite_summary_figures(results, suite_dir)
        for k, p in plots.items():
            logger.info(f"Generated Suite Plot [{k}]: {p}")

        logger.info("\n" + "=" * 90)
        logger.info("AUTOMATED RESEARCH EXPERIMENTS: SUITE SUMMARY TABLE")
        logger.info("=" * 90)
        for line in md_table.split("\n"):
            logger.info(line)
        logger.info("=" * 90)

        return results
