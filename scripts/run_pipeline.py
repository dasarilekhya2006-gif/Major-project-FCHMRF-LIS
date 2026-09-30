"""Master CLI entry point executing the integrated neuroimaging research pipeline."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.pipeline.stages import PipelineStage
from src.pipeline.master_pipeline import MasterPipeline, MasterPipelineConfig

logger = get_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the Adaptive Deep Learning-Assisted fcHMRF-LIS Integrated Research Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to master pipeline YAML configuration.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input raw 3D or 4D NIfTI file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=[s.value for s in PipelineStage],
        help="Execute a single stage independently.",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default=None,
        choices=[s.value for s in PipelineStage],
        help="Start execution from this stage.",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default=None,
        choices=[s.value for s in PipelineStage],
        help="Stop execution after this stage.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of all stages, bypassing cached artifacts.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable Phase 6 Bayesian Optimization hyperparameter tuning.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Computation device ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique identifier for this experiment run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base destination directory for all pipeline artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(level="INFO")
    args = parse_args()

    # Load configuration
    cfg_p = Path(args.config)
    if cfg_p.exists():
        config = MasterPipelineConfig.from_yaml(cfg_p)
    else:
        logger.warning(f"Config file {cfg_p} not found. Instantiating default pipeline configuration.")
        config = MasterPipelineConfig()

    # Apply CLI overrides
    if args.input:
        config.raw_input_path = Path(args.input)
    if args.force_recompute:
        config.force_recompute = True
    if args.adaptive:
        config.run_adaptive_selection = True
    if args.device:
        config.device = args.device
    if args.experiment_id:
        config.experiment_id = args.experiment_id
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    pipeline = MasterPipeline(config=config)

    # Determine execution mode
    if args.stage:
        target_stage = PipelineStage(args.stage)
        logger.info(f"Executing single stage independently: {target_stage.value}")
        if target_stage == PipelineStage.DATA_LOADING:
            pipeline.run_stage_data_loading()
        elif target_stage == PipelineStage.PREPROCESSING:
            raw_path = config.raw_input_path or (pipeline.output_dir / "raw_synthetic_demo.nii.gz")
            if not raw_path.exists():
                pipeline.run_stage_data_loading()
            pipeline.run_stage_preprocessing(raw_path)
        elif target_stage == PipelineStage.ROI_DETECTION:
            prep_path = pipeline.dir_preproc / f"{config.experiment_id}_preprocessed.nii.gz"
            pipeline.run_stage_roi_detection(prep_path)
        elif target_stage == PipelineStage.STATISTICAL_TESTING:
            prep_path = pipeline.dir_preproc / f"{config.experiment_id}_preprocessed.nii.gz"
            roi_path = pipeline.dir_segment / f"{config.experiment_id}_roi_binary.nii.gz"
            pipeline.run_stage_statistical_testing(prep_path, roi_path)
        elif target_stage == PipelineStage.FCHMRF_LIS:
            z_path = pipeline.dir_stats / f"{config.experiment_id}_zscores.nii.gz"
            roi_path = pipeline.dir_segment / f"{config.experiment_id}_roi_binary.nii.gz"
            dl_path = pipeline.dir_segment / f"{config.experiment_id}_roi_prob.nii.gz"
            pipeline.run_stage_fchmrf_lis(z_path, roi_path, dl_path)
        elif target_stage == PipelineStage.ADAPTIVE_SELECTION:
            z_path = pipeline.dir_stats / f"{config.experiment_id}_zscores.nii.gz"
            dl_path = pipeline.dir_segment / f"{config.experiment_id}_roi_prob.nii.gz"
            roi_path = pipeline.dir_segment / f"{config.experiment_id}_roi_binary.nii.gz"
            pipeline.run_stage_adaptive_selection(z_path, dl_path, roi_path)
    else:
        # End-to-end or ranged execution
        start_st = PipelineStage(args.start_stage) if args.start_stage else PipelineStage.DATA_LOADING
        end_st = PipelineStage(args.end_stage) if args.end_stage else None
        results = pipeline.run(start_stage=start_st, end_stage=end_st)

        # Print console summary
        print("\n" + "=" * 78)
        print(f"PIPELINE EXECUTION SUMMARY: {config.experiment_id}")
        print("=" * 78)
        print(f"{'Stage':<24} | {'Status':<10} | {'Duration':<10} | {'Metrics / Info'}")
        print("-" * 78)
        has_failure = False
        failed_stage = None
        for stg, res in results.items():
            dur_str = f"{res.elapsed_seconds:.2f}s"
            m_list = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(res.metrics.items())[:2]]
            m_str = ", ".join(m_list) if m_list else ""
            if res.error_message:
                m_str = f"ERROR: {res.error_message}"
            print(f"{stg.value:<24} | {res.status.value:<10} | {dur_str:<10} | {m_str}")
            if res.status.value == "failed":
                has_failure = True
                failed_stage = stg

        print("=" * 78)
        if has_failure:
            logger.error(f"Pipeline execution finished with FAILURE in stage: {failed_stage.value}")
            sys.exit(1)
        else:
            logger.info("Pipeline completed all requested stages successfully.")
            sys.exit(0)


if __name__ == "__main__":
    main()
