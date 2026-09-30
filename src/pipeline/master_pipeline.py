"""Master pipeline orchestrator integrating Phases 1 through 6 into a reproducible workflow."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import numpy as np

from src.core.exceptions import DataValidationError, PipelineError, StatisticalInferenceError
from src.core.logging import get_logger, setup_logging
from src.core.seed import set_seed
from src.data.loaders import load_nifti, save_nifti, validate_image
from src.data.sample_data import create_sample_3d_statistical_nifti
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.schemas import PreprocessingConfig
from src.preprocessing.qc import plot_preprocessing_qc
from src.deep_learning.inference import Predictor3D
from src.deep_learning.models.unet3d import ConfigurableUNet3D
from src.statistical_analysis.voxel_testing import compute_statistics, StatisticalMaps
from src.statistical_analysis.design import TestType
from src.statistical_analysis.visualization import plot_statistical_maps_ortho
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.fchmrf_model import FCHMRFModel, FCHMRFResult
from src.fchmrf_lis.visualization import plot_fchmrf_summary
from src.adaptive_selection.search_space import create_fchmrf_search_space
from src.adaptive_selection.objectives import (
    ObjectiveType,
    ValidationFDRBenchmarkObjective,
)
from src.adaptive_selection.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
)
from src.adaptive_selection.baseline_comparator import BaselineComparator
from src.adaptive_selection.visualization import plot_adaptive_optimization_summary
from src.pipeline.stages import PipelineStage, StageResult, StageStatus
from src.pipeline.reporting import generate_experiment_report

logger = get_logger(__name__)


@dataclass
class MasterPipelineConfig:
    """Master configuration controlling the end-to-end neuroimaging research pipeline."""

    experiment_id: str = "pipeline_run"
    output_dir: Path = Path("outputs")
    device: str = "auto"
    force_recompute: bool = False
    run_adaptive_selection: bool = False
    seed: int = 42

    # Stage 1: Data Loading
    raw_input_path: Optional[Path] = None

    # Stage 2: Preprocessing
    norm_method: str = "zscore"
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)

    # Stage 3: 3D U-Net ROI Detection
    model_checkpoint_path: Optional[Path] = None
    roi_threshold: float = 0.5

    # Stage 4: Statistical Testing
    test_type: str = "one_sample_t"
    mu_null: float = 0.0
    alternative: str = "greater"
    restrict_stats_to_roi: bool = True

    # Stage 5: fcHMRF-LIS
    fchmrf_mode: str = "original_fchmrf"
    beta: float = 0.8
    sigma_spatial: float = 2.5
    sigma_intensity: float = 1.5
    lambda_dl: float = 0.5
    alpha: float = 0.05
    max_mf_iterations: int = 40

    # Stage 6: Adaptive Selection
    bo_iterations: int = 10
    bo_initial: int = 4
    acquisition_function: str = "expected_improvement"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "MasterPipelineConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        gen_cfg = data.get("general", {})
        s1 = data.get("stage1_data_loading", {})
        s2 = data.get("stage2_preprocessing", {})
        s3 = data.get("stage3_roi_detection", {})
        s4 = data.get("stage4_statistical_testing", {})
        s5 = data.get("stage5_fchmrf_lis", {})
        s6 = data.get("stage6_adaptive_selection", {})

        return cls(
            experiment_id=gen_cfg.get("experiment_id", "pipeline_run"),
            output_dir=Path(gen_cfg.get("output_dir", "outputs")),
            device=gen_cfg.get("device", "auto"),
            force_recompute=bool(gen_cfg.get("force_recompute", False)),
            run_adaptive_selection=bool(gen_cfg.get("run_adaptive_selection", False)),
            seed=int(gen_cfg.get("seed", 42)),
            raw_input_path=Path(s1["input_path"]) if "input_path" in s1 and s1["input_path"] else None,
            norm_method=s2.get("norm_method", "zscore"),
            clip_percentiles=tuple(s2.get("clip_percentiles", [1.0, 99.0])),
            model_checkpoint_path=Path(s3["checkpoint_path"]) if "checkpoint_path" in s3 and s3["checkpoint_path"] else None,
            roi_threshold=float(s3.get("roi_threshold", 0.5)),
            test_type=s4.get("test_type", "one_sample_t"),
            mu_null=float(s4.get("mu_null", 0.0)),
            alternative=s4.get("alternative", "greater"),
            restrict_stats_to_roi=bool(s4.get("restrict_stats_to_roi", True)),
            fchmrf_mode=s5.get("method_mode", "original_fchmrf"),
            beta=float(s5.get("beta", 0.8)),
            sigma_spatial=float(s5.get("sigma_spatial", 2.5)),
            sigma_intensity=float(s5.get("sigma_intensity", 1.5)),
            lambda_dl=float(s5.get("lambda_dl", 0.5)),
            alpha=float(s5.get("alpha", 0.05)),
            max_mf_iterations=int(s5.get("max_iterations", 40)),
            bo_iterations=int(s6.get("n_iterations", 10)),
            bo_initial=int(s6.get("n_initial", 4)),
            acquisition_function=s6.get("acquisition_function", "expected_improvement"),
        )


class MasterPipeline:
    """Integrated execution engine connecting Phase 1 to Phase 6."""

    def __init__(self, config: MasterPipelineConfig) -> None:
        self.config = config
        self.stage_results: Dict[PipelineStage, StageResult] = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output subdirectories
        self.dir_preproc = self.output_dir / "preprocessed"
        self.dir_segment = self.output_dir / "segmentations"
        self.dir_stats = self.output_dir / "statistical_maps"
        self.dir_fchmrf = self.output_dir / "fchmrf_results"
        self.dir_adaptive = self.output_dir / "adaptive_selection"
        self.dir_figures = self.output_dir / "figures"
        self.dir_reports = self.output_dir / "reports"

        for d in [
            self.dir_preproc,
            self.dir_segment,
            self.dir_stats,
            self.dir_fchmrf,
            self.dir_adaptive,
            self.dir_figures,
            self.dir_reports,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Device detection
        self.resolved_device = self._resolve_device(config.device)
        logger.info(f"Initialized MasterPipeline [device={self.resolved_device}, force_recompute={config.force_recompute}]")

    def _resolve_device(self, req: str) -> str:
        """Resolve CPU vs GPU execution."""
        if req == "cpu":
            return "cpu"
        try:
            import torch
            if req == "cuda" or (req == "auto" and torch.cuda.is_available()):
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # =========================================================================
    # Stage 1: Data Loading & Validation
    # =========================================================================
    def run_stage_data_loading(self, input_path: Optional[Path] = None) -> StageResult:
        stage = PipelineStage.DATA_LOADING
        t_start = time.perf_counter()
        logger.info(f"--- Running {stage.value} ---")

        try:
            path = input_path or self.config.raw_input_path
            if path is None or not path.exists():
                # Create synthetic raw demo volume if none exists
                path = self.output_dir / "raw_synthetic_demo.nii.gz"
                logger.info(f"Creating sample raw input volume at {path}")
                create_sample_3d_statistical_nifti(path, shape=(32, 32, 16), seed=self.config.seed)

            report = validate_image(path)
            if not report.is_valid:
                err_msg = "; ".join(report.errors) if report.errors else "Validation failed"
                res = StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    error_message=f"Validation failed: {err_msg}",
                    elapsed_seconds=time.perf_counter() - t_start,
                )
                self.stage_results[stage] = res
                return res

            volume = load_nifti(path)

            res = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output_files={"raw_volume": path},
                metrics={
                    "shape": list(volume.shape),
                    "is_4d": volume.is_4d,
                    "voxel_count": int(np.prod(volume.shape)),
                    "voxel_spacing": list(volume.voxel_spacing),
                },
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=True,
            )
            self.stage_results[stage] = res
            return res
        except Exception as e:
            logger.error(f"Error in {stage.value}: {e}", exc_info=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error_message=str(e),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            self.stage_results[stage] = res
            return res

    # =========================================================================
    # Stage 2: Preprocessing
    # =========================================================================
    def run_stage_preprocessing(self, raw_path: Path) -> StageResult:
        stage = PipelineStage.PREPROCESSING
        t_start = time.perf_counter()
        logger.info(f"--- Running {stage.value} ---")

        out_preproc = self.dir_preproc / f"{self.config.experiment_id}_preprocessed.nii.gz"
        out_qc = self.dir_figures / f"{self.config.experiment_id}_preprocessing_qc.png"

        # Check caching
        if not self.config.force_recompute and out_preproc.exists() and out_qc.exists():
            logger.info("Found existing preprocessed artifacts. Skipping recomputation.")
            vol = load_nifti(out_preproc)
            res = StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                output_files={"preprocessed_nifti": out_preproc, "qc_figure": out_qc},
                metrics={"shape": list(vol.shape), "cached": True},
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=False,
            )
            self.stage_results[stage] = res
            return res

        try:
            # Run preprocessing
            pipe_cfg = PreprocessingConfig(
                normalization_method=self.config.norm_method,
                percentile_bounds=self.config.clip_percentiles,
                imputation_method="mean",
                compute_mask_if_missing=True,
            )
            pipeline = PreprocessingPipeline(config=pipe_cfg)
            raw_vol = load_nifti(raw_path)
            prep_result = pipeline.run(raw_vol)
            save_nifti(prep_result, out_preproc)

            # Quality control plot
            plot_preprocessing_qc(raw_vol, prep_result, output_path=out_qc)

            res = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output_files={"preprocessed_nifti": out_preproc, "qc_figure": out_qc},
                metrics={
                    "original_shape": list(prep_result.preprocessing_metadata.original_shape),
                    "processed_shape": list(prep_result.shape),
                    "mean_intensity": float(np.nanmean(prep_result.data)),
                    "std_intensity": float(np.nanstd(prep_result.data)),
                },
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=True,
            )
            self.stage_results[stage] = res
            return res
        except Exception as e:
            logger.error(f"Error in {stage.value}: {e}", exc_info=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error_message=str(e),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            self.stage_results[stage] = res
            return res

    # =========================================================================
    # Stage 3: 3D U-Net ROI Detection
    # =========================================================================
    def run_stage_roi_detection(self, preprocessed_path: Path) -> StageResult:
        stage = PipelineStage.ROI_DETECTION
        t_start = time.perf_counter()
        logger.info(f"--- Running {stage.value} ---")

        out_bin = self.dir_segment / f"{self.config.experiment_id}_roi_binary.nii.gz"
        out_prob = self.dir_segment / f"{self.config.experiment_id}_roi_prob.nii.gz"

        if not self.config.force_recompute and out_bin.exists() and out_prob.exists():
            logger.info("Found existing ROI segmentation artifacts. Skipping recomputation.")
            mask_vol = load_nifti(out_bin)
            res = StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                output_files={"roi_binary": out_bin, "roi_prob": out_prob},
                metrics={"roi_voxels": int(np.sum(mask_vol.data > 0.5)), "cached": True},
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=False,
            )
            self.stage_results[stage] = res
            return res

        try:
            # Instantiate 3D U-Net architecture
            model = ConfigurableUNet3D(
                in_channels=1,
                out_channels=1,
                init_features=16,
                depth=3,
            )
            # Load weights if available, otherwise initialize weights deterministically
            if self.config.model_checkpoint_path and self.config.model_checkpoint_path.exists():
                logger.info(f"Loading checkpoint weights from {self.config.model_checkpoint_path}")
                model.load_weights(self.config.model_checkpoint_path)

            predictor = Predictor3D(
                model=model,
                device=self.resolved_device,
                binary_threshold=self.config.roi_threshold,
            )
            pred_res = predictor.predict_nifti(
                nifti_path=preprocessed_path,
                save_binary_path=out_bin,
                save_prob_path=out_prob,
            )

            res = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output_files={"roi_binary": out_bin, "roi_prob": out_prob},
                metrics={
                    "roi_voxels": int(pred_res.roi_voxel_count),
                    "roi_volume_mm3": float(pred_res.roi_volume_mm3),
                    "threshold_used": self.config.roi_threshold,
                    "device_used": self.resolved_device,
                },
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=True,
            )
            self.stage_results[stage] = res
            return res
        except Exception as e:
            logger.error(f"Error in {stage.value}: {e}", exc_info=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error_message=str(e),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            self.stage_results[stage] = res
            return res

    # =========================================================================
    # Stage 4: ROI-Restricted Statistical Testing
    # =========================================================================
    def run_stage_statistical_testing(
        self,
        preprocessed_path: Path,
        roi_mask_path: Optional[Path] = None,
    ) -> StageResult:
        stage = PipelineStage.STATISTICAL_TESTING
        t_start = time.perf_counter()
        logger.info(f"--- Running {stage.value} ---")

        out_z = self.dir_stats / f"{self.config.experiment_id}_zscores.nii.gz"
        out_p = self.dir_stats / f"{self.config.experiment_id}_pvalues.nii.gz"
        out_fig = self.dir_figures / f"{self.config.experiment_id}_stats_ortho.png"

        if not self.config.force_recompute and out_z.exists() and out_fig.exists():
            logger.info("Found existing statistical maps. Skipping recomputation.")
            z_vol = load_nifti(out_z)
            res = StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                output_files={"z_scores": out_z, "p_values": out_p, "ortho_figure": out_fig},
                metrics={"analyzed_voxels": int(np.sum(~np.isnan(z_vol.data))), "cached": True},
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=False,
            )
            self.stage_results[stage] = res
            return res

        try:
            prep_vol = load_nifti(preprocessed_path)
            roi_mask = None
            if self.config.restrict_stats_to_roi and roi_mask_path and roi_mask_path.exists():
                roi_vol = load_nifti(roi_mask_path, allow_empty=True)
                roi_mask = roi_vol.data > 0.5
                # Ensure ROI mask has at least some voxels
                if np.sum(roi_mask) == 0:
                    logger.warning("ROI mask is empty. Falling back to whole-brain domain.")
                    roi_mask = None

            effective_test_type = self.config.test_type
            if prep_vol.data.ndim == 3 and effective_test_type in ("one_sample_t", "two_sample_t", "paired_t"):
                logger.info(f"Input volume is 3D with shape {prep_vol.data.shape}. Adapting test_type to 'direct_z'.")
                effective_test_type = "direct_z"

            stat_maps = compute_statistics(
                data=prep_vol.data,
                test_type=TestType(effective_test_type),
                roi_mask=roi_mask,
                mu_null=self.config.mu_null,
                alternative=self.config.alternative,
                affine=prep_vol.affine,
            )

            saved_files = stat_maps.save_nifti_maps(output_dir=self.dir_stats, prefix=self.config.experiment_id)
            plot_statistical_maps_ortho(stat_maps, output_path=out_fig)

            res = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output_files={"z_scores": saved_files["z_scores"], "p_values": saved_files["p_values"], "ortho_figure": out_fig},
                metrics={
                    "test_type": self.config.test_type,
                    "analyzed_voxels": int(stat_maps.analyzed_voxels),
                    "uncorrected_significant_p001": int(np.sum(stat_maps.p_values[stat_maps.spatial_mask] < 0.001)),
                    "max_abs_z": float(np.nanmax(np.abs(stat_maps.z_scores[stat_maps.spatial_mask]))),
                },
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=True,
            )
            self.stage_results[stage] = res
            return res
        except Exception as e:
            logger.error(f"Error in {stage.value}: {e}", exc_info=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error_message=str(e),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            self.stage_results[stage] = res
            return res

    # =========================================================================
    # Stage 5: fcHMRF-LIS Spatial Statistical Modeling
    # =========================================================================
    def run_stage_fchmrf_lis(
        self,
        z_scores_path: Path,
        roi_mask_path: Optional[Path] = None,
        dl_prior_path: Optional[Path] = None,
        override_params: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None,
    ) -> StageResult:
        stage = PipelineStage.FCHMRF_LIS
        t_start = time.perf_counter()
        logger.info(f"--- Running {stage.value} ---")

        pfx = prefix or self.config.experiment_id
        out_lis = self.dir_fchmrf / f"{pfx}_lis_map.nii.gz"
        out_sig = self.dir_fchmrf / f"{pfx}_significant_mask.nii.gz"
        out_prob = self.dir_fchmrf / f"{pfx}_posterior_prob.nii.gz"
        out_fig = self.dir_figures / f"{pfx}_fchmrf_summary.png"

        if not self.config.force_recompute and out_lis.exists() and out_sig.exists() and out_fig.exists():
            logger.info("Found existing fcHMRF-LIS artifacts. Skipping recomputation.")
            sig_vol = load_nifti(out_sig, allow_empty=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                output_files={"lis_map": out_lis, "significant_mask": out_sig, "summary_figure": out_fig},
                metrics={"discoveries": int(np.sum(sig_vol.data > 0.5)), "cached": True},
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=False,
            )
            self.stage_results[stage] = res
            return res

        try:
            # Parameters
            beta = self.config.beta
            sigma_s = self.config.sigma_spatial
            sigma_r = self.config.sigma_intensity
            lambda_dl = self.config.lambda_dl
            if override_params:
                beta = float(override_params.get("beta", beta))
                sigma_s = float(override_params.get("sigma_spatial", sigma_s))
                sigma_r = float(override_params.get("sigma_intensity", sigma_r))
                lambda_dl = float(override_params.get("lambda_dl", lambda_dl))

            fchmrf_params = FCHMRFParameters(
                method_mode=MethodMode(self.config.fchmrf_mode),
                beta=beta,
                sigma_spatial=sigma_s,
                sigma_intensity=sigma_r,
                lambda_dl=lambda_dl,
                alpha=self.config.alpha,
                max_iterations=self.config.max_mf_iterations,
            )

            model = FCHMRFModel(params=fchmrf_params)
            z_vol = load_nifti(z_scores_path)
            roi_mask = load_nifti(roi_mask_path, allow_empty=True).data > 0.5 if roi_mask_path and roi_mask_path.exists() else None
            dl_prior = load_nifti(dl_prior_path, allow_empty=True).data if dl_prior_path and dl_prior_path.exists() else None

            fchmrf_res = model.fit_predict(
                z_scores=z_vol.data,
                dl_prior=dl_prior,
                spatial_mask=roi_mask,
                affine=z_vol.affine,
            )

            saved = fchmrf_res.save_nifti_results(output_dir=self.dir_fchmrf, prefix=pfx)
            plot_fchmrf_summary(result=fchmrf_res, z_scores=z_vol.data, output_path=out_fig)

            res = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output_files={
                    "lis_map": saved["lis_map"],
                    "significant_mask": saved["significant_mask"],
                    "posterior_prob": saved["posterior_prob"],
                    "summary_json": saved["summary_json"],
                    "summary_figure": out_fig,
                },
                metrics={
                    "method_mode": fchmrf_res.method_mode,
                    "analyzed_voxels": int(fchmrf_res.num_analyzed_voxels),
                    "discoveries": int(fchmrf_res.num_discoveries),
                    "discovery_rate_pct": float(100.0 * fchmrf_res.num_discoveries / max(1, fchmrf_res.num_analyzed_voxels)),
                    "lis_cutoff": float(fchmrf_res.fdr_results["lis_threshold"]),
                    "estimated_fdr": float(fchmrf_res.fdr_results["estimated_fdr"]),
                    "nominal_alpha": float(fchmrf_res.fdr_results["nominal_alpha"]),
                },
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=True,
            )
            self.stage_results[stage] = res
            return res
        except Exception as e:
            logger.error(f"Error in {stage.value}: {e}", exc_info=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error_message=str(e),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            self.stage_results[stage] = res
            return res

    # =========================================================================
    # Stage 6: Adaptive Parameter Selection (Optional)
    # =========================================================================
    def run_stage_adaptive_selection(
        self,
        z_scores_path: Path,
        dl_prior_path: Optional[Path] = None,
        roi_mask_path: Optional[Path] = None,
    ) -> StageResult:
        stage = PipelineStage.ADAPTIVE_SELECTION
        t_start = time.perf_counter()
        logger.info(f"--- Running {stage.value} ---")

        out_yaml = self.dir_adaptive / f"{self.config.experiment_id}_selected.yaml"
        out_fig = self.dir_figures / f"{self.config.experiment_id}_adaptive_summary.png"

        if not self.config.force_recompute and out_yaml.exists() and out_fig.exists():
            logger.info("Found existing adaptive parameter selection artifacts. Skipping recomputation.")
            res = StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                output_files={"selected_yaml": out_yaml, "summary_figure": out_fig},
                metrics={"cached": True},
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=False,
            )
            self.stage_results[stage] = res
            return res

        try:
            z_vol = load_nifti(z_scores_path)
            dl_vol = load_nifti(dl_prior_path, allow_empty=True).data if dl_prior_path and dl_prior_path.exists() else None
            roi_mask = None
            if roi_mask_path and roi_mask_path.exists():
                roi_vol = load_nifti(roi_mask_path, allow_empty=True)
                roi_mask = roi_vol.data > 0.5
                if np.sum(roi_mask) == 0:
                    roi_mask = None

            # Partition volume into validation and test sets to strictly prevent data leakage
            z_data = z_vol.data
            mid_z = z_data.shape[2] // 2
            val_z = z_data[:, :, :mid_z]
            val_dl = dl_vol[:, :, :mid_z] if dl_vol is not None else None
            val_mask = roi_mask[:, :, :mid_z] if roi_mask is not None else None

            # Build search space & objective
            space = create_fchmrf_search_space(include_lambda_dl=(self.config.fchmrf_mode == "adaptive_dl_fchmrf"))
            # Using synthetic benchmark proxy on validation partition
            pseudo_gt = np.abs(val_z) > 2.5
            if np.sum(pseudo_gt) == 0:
                pseudo_gt[0, 0, 0] = True

            objective = ValidationFDRBenchmarkObjective(
                z_scores_val=val_z,
                ground_truth_val=pseudo_gt,
                dl_prior_val=val_dl,
                spatial_mask_val=val_mask,
                target_alpha=self.config.alpha,
                objective_type=ObjectiveType.PENALIZED_POWER,
                max_iterations=20,
            )

            optimizer = BayesianOptimizer(
                search_space=space,
                objective=objective,
                acquisition_function=AcquisitionFunction(self.config.acquisition_function),
                seed=self.config.seed,
            )

            opt_result = optimizer.optimize(
                n_iterations=self.config.bo_iterations,
                n_initial=self.config.bo_initial,
            )

            saved = opt_result.save_results(output_dir=self.dir_adaptive, prefix=self.config.experiment_id)

            # Baseline comparison on held-out test partition
            test_z = z_data[:, :, mid_z:]
            test_dl = dl_vol[:, :, mid_z:] if dl_vol is not None else None
            test_mask = roi_mask[:, :, mid_z:] if roi_mask is not None else None
            test_gt = np.abs(test_z) > 2.5

            comparator = BaselineComparator(
                method_mode=MethodMode(self.config.fchmrf_mode),
                target_alpha=self.config.alpha,
            )
            comp_report = comparator.compare(
                baseline_params=opt_result.baseline_parameters,
                selected_params=opt_result.best_parameters,
                z_scores_test=test_z,
                ground_truth_test=test_gt if np.any(test_gt) else None,
                dl_prior_test=test_dl,
                spatial_mask_test=test_mask,
                optimization_cost_seconds=opt_result.total_runtime_seconds,
            )

            plot_adaptive_optimization_summary(
                opt_result=opt_result,
                comparison_report=comp_report,
                output_path=out_fig,
            )

            res = StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output_files={
                    "selected_yaml": saved["selected_yaml"],
                    "history_json": saved["history_json"],
                    "summary_figure": out_fig,
                },
                metrics={
                    "best_score": float(opt_result.best_score),
                    "baseline_score": float(opt_result.baseline_score),
                    "score_improvement": float(opt_result.score_improvement),
                    "is_benefit_measurable": comp_report.is_benefit_measurable,
                    "total_bo_runtime_seconds": float(opt_result.total_runtime_seconds),
                },
                elapsed_seconds=time.perf_counter() - t_start,
                recomputed=True,
            )
            self.stage_results[stage] = res
            return res
        except Exception as e:
            logger.error(f"Error in {stage.value}: {e}", exc_info=True)
            res = StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error_message=str(e),
                elapsed_seconds=time.perf_counter() - t_start,
            )
            self.stage_results[stage] = res
            return res

    # =========================================================================
    # Master Execution Loop
    # =========================================================================
    def run(
        self,
        start_stage: PipelineStage = PipelineStage.DATA_LOADING,
        end_stage: Optional[PipelineStage] = None,
    ) -> Dict[PipelineStage, StageResult]:
        """Execute the integrated pipeline from start_stage to end_stage.

        Default: End-to-end execution of all enabled stages.
        """
        set_seed(self.config.seed)
        t_global_start = time.perf_counter()
        t_start_iso = datetime.now(timezone.utc).isoformat()
        logger.info(f"========== STARTING INTEGRATED PIPELINE: {self.config.experiment_id} ==========")

        try:
            # 1. Data Loading
            res_s1 = self.run_stage_data_loading()
            if res_s1.status == StageStatus.FAILED:
                logger.error(f"Stage {res_s1.stage.value} failed: {res_s1.error_message}")
                return self.stage_results
            if end_stage == PipelineStage.DATA_LOADING:
                return self.stage_results
            raw_path = res_s1.output_files["raw_volume"]

            # 2. Preprocessing
            res_s2 = self.run_stage_preprocessing(raw_path)
            if res_s2.status == StageStatus.FAILED:
                logger.error(f"Stage {res_s2.stage.value} failed: {res_s2.error_message}")
                return self.stage_results
            if end_stage == PipelineStage.PREPROCESSING:
                return self.stage_results
            prep_path = res_s2.output_files["preprocessed_nifti"]

            # 3. 3D U-Net ROI Detection
            res_s3 = self.run_stage_roi_detection(prep_path)
            if res_s3.status == StageStatus.FAILED:
                logger.error(f"Stage {res_s3.stage.value} failed: {res_s3.error_message}")
                return self.stage_results
            if end_stage == PipelineStage.ROI_DETECTION:
                return self.stage_results
            roi_bin_path = res_s3.output_files["roi_binary"]
            roi_prob_path = res_s3.output_files["roi_prob"]

            # 4. Statistical Testing (ROI-restricted)
            res_s4 = self.run_stage_statistical_testing(
                preprocessed_path=prep_path,
                roi_mask_path=roi_bin_path,
            )
            if res_s4.status == StageStatus.FAILED:
                logger.error(f"Stage {res_s4.stage.value} failed: {res_s4.error_message}")
                return self.stage_results
            if end_stage == PipelineStage.STATISTICAL_TESTING:
                return self.stage_results
            z_path = res_s4.output_files["z_scores"]

            # 5. Baseline fcHMRF-LIS Inference
            res_s5 = self.run_stage_fchmrf_lis(
                z_scores_path=z_path,
                roi_mask_path=roi_bin_path,
                dl_prior_path=roi_prob_path,
            )
            if res_s5.status == StageStatus.FAILED:
                logger.error(f"Stage {res_s5.stage.value} failed: {res_s5.error_message}")
                return self.stage_results
            if end_stage == PipelineStage.FCHMRF_LIS and not self.config.run_adaptive_selection:
                return self.stage_results

            # 6. Adaptive Parameter Selection & Tuned FDR Results (Optional)
            if self.config.run_adaptive_selection:
                res_s6 = self.run_stage_adaptive_selection(
                    z_scores_path=z_path,
                    dl_prior_path=roi_prob_path,
                    roi_mask_path=roi_bin_path,
                )
                if res_s6.status == StageStatus.FAILED:
                    logger.error(f"Stage {res_s6.stage.value} failed: {res_s6.error_message}")
                    return self.stage_results

                with open(res_s6.output_files["selected_yaml"], "r") as f:
                    sel_data = yaml.safe_load(f)
                    override_params = sel_data.get("adaptive_selection", {}).get("selected_parameters")

                logger.info(f"Executing adaptively-tuned fcHMRF-LIS with parameters: {override_params}")
                res_s5_adaptive = self.run_stage_fchmrf_lis(
                    z_scores_path=z_path,
                    roi_mask_path=roi_bin_path,
                    dl_prior_path=roi_prob_path,
                    override_params=override_params,
                    prefix=f"{self.config.experiment_id}_adaptive",
                )
                if res_s5_adaptive.status == StageStatus.FAILED:
                    logger.error(f"Adaptively-tuned fcHMRF-LIS failed: {res_s5_adaptive.error_message}")
                    return self.stage_results

                res_s6.output_files.update({
                    "adaptive_lis_map": res_s5_adaptive.output_files.get("lis_map"),
                    "adaptive_significant_mask": res_s5_adaptive.output_files.get("significant_mask"),
                    "adaptive_posterior_prob": res_s5_adaptive.output_files.get("posterior_prob"),
                    "adaptive_summary_figure": res_s5_adaptive.output_files.get("summary_figure"),
                })
                res_s6.metrics["adaptive_discoveries"] = res_s5_adaptive.metrics.get("discoveries", 0)
                res_s6.metrics["adaptive_estimated_fdr"] = res_s5_adaptive.metrics.get("estimated_fdr", 0.0)

        except Exception as e:
            logger.error(f"Pipeline execution halted due to unhandled error: {e}", exc_info=True)
            raise PipelineError(f"Master pipeline failure: {e}") from e

        finally:
            t_total = time.perf_counter() - t_global_start
            t_end_iso = datetime.now(timezone.utc).isoformat()
            # Generate Master Experiment Report
            report_paths = generate_experiment_report(
                experiment_id=self.config.experiment_id,
                stage_results=self.stage_results,
                pipeline_config=asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else {},
                output_dir=self.dir_reports,
                start_time=t_start_iso,
                end_time=t_end_iso,
                total_duration_seconds=t_total,
            )
            logger.info("=" * 70)
            logger.info(f"Master Pipeline Completed in {t_total:.2f} seconds.")
            logger.info(f"Master Experiment Report: {report_paths['markdown_report']}")
            logger.info("=" * 70)

        return self.stage_results
