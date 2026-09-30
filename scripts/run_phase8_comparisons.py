"""CLI script executing Phase 8 rigorous experimental comparisons across baselines."""

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.evaluation.baselines import (
    AdaptiveFCHMRFLISCorrection,
    AdaptiveROIAssistedFCHMRFLISCorrection,
    BenjaminiHochbergCorrection,
    LocalFDRCorrection,
    OriginalFCHMRFLISCorrection,
    ROIAssistedFCHMRFLISCorrection,
)
from src.evaluation.synthetic_benchmarks import generate_monte_carlo_dataset
from src.evaluation.comparative_runner import ComparativeExperimentRunner
from src.evaluation.visualization import plot_comparative_benchmark_summary

logger = get_logger("run_phase8_comparisons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Phase 8 rigorous multi-method benchmark comparison."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_comparison_config.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of Monte Carlo simulation repetitions.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Signal-to-noise ratio of simulated activation regions.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Target nominal spatial FDR threshold (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for tables and reports.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(level="INFO")
    args = parse_args()

    cfg_p = Path(args.config)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {cfg_p} not found. Using defaults.")
        cfg = {}

    exp_cfg = cfg.get("experiment", {})
    methods_cfg = cfg.get("methods", {})
    hp_cfg = cfg.get("hyperparameters", {})
    out_cfg = cfg.get("outputs", {})

    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    num_trials = args.num_trials if args.num_trials is not None else int(exp_cfg.get("num_trials", 5))
    snr = args.snr if args.snr is not None else float(exp_cfg.get("snr", 3.0))
    alpha = args.alpha if args.alpha is not None else float(exp_cfg.get("nominal_alpha", 0.05))
    shape = tuple(exp_cfg.get("volume_shape", [24, 24, 12]))

    tables_dir = Path(out_cfg.get("tables_dir", "outputs/tables"))
    figures_dir = Path(out_cfg.get("figures_dir", "outputs/figures"))
    csv_filename = out_cfg.get("csv_filename", "phase8_comparative_metrics.csv")
    md_filename = out_cfg.get("markdown_filename", "phase8_comparative_metrics.md")
    fig_filename = out_cfg.get("figure_filename", "phase8_comparative_summary.png")

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Methods
    fixed_hp = hp_cfg.get("fixed", {})
    adaptive_hp = hp_cfg.get("adaptive_optimized", {})

    methods = []
    if methods_cfg.get("include_bh_fdr", True):
        methods.append(BenjaminiHochbergCorrection(alpha=alpha, name="Benjamini-Hochberg FDR"))
    if methods_cfg.get("include_local_fdr", True):
        methods.append(LocalFDRCorrection(alpha=alpha))
    if methods_cfg.get("include_original_fchmrf", True):
        methods.append(
            OriginalFCHMRFLISCorrection(
                alpha=alpha,
                beta=float(fixed_hp.get("beta", 0.8)),
                sigma_spatial=float(fixed_hp.get("sigma_spatial", 2.5)),
                sigma_intensity=float(fixed_hp.get("sigma_intensity", 1.5)),
                max_iterations=int(fixed_hp.get("max_iterations", 35)),
                name="Original fcHMRF-LIS",
            )
        )
    if methods_cfg.get("include_roi_fchmrf", True):
        methods.append(
            ROIAssistedFCHMRFLISCorrection(
                alpha=alpha,
                beta=float(fixed_hp.get("beta", 0.8)),
                sigma_spatial=float(fixed_hp.get("sigma_spatial", 2.5)),
                sigma_intensity=float(fixed_hp.get("sigma_intensity", 1.5)),
                lambda_dl=float(fixed_hp.get("lambda_dl", 0.5)),
                max_iterations=int(fixed_hp.get("max_iterations", 35)),
                name="ROI + fcHMRF-LIS",
            )
        )
    if methods_cfg.get("include_adaptive_fchmrf", True):
        adaptive_fchmrf_hp = hp_cfg.get("adaptive_fchmrf", {})
        methods.append(
            AdaptiveFCHMRFLISCorrection(
                alpha=alpha,
                optimized_parameters=adaptive_fchmrf_hp,
                max_iterations=int(adaptive_fchmrf_hp.get("max_iterations", 35)),
                name="Adaptive fcHMRF-LIS",
            )
        )
    if methods_cfg.get("include_adaptive_roi_fchmrf", True):
        methods.append(
            AdaptiveROIAssistedFCHMRFLISCorrection(
                alpha=alpha,
                optimized_parameters=adaptive_hp,
                max_iterations=int(adaptive_hp.get("max_iterations", 35)),
                name="ROI + Adaptive fcHMRF-LIS",
            )
        )

    # 2. Generate Reproducible Monte Carlo Benchmark Dataset
    logger.info(
        f"Generating {num_trials} Monte Carlo benchmark volumes (shape={shape}, SNR={snr:.1f})..."
    )
    dataset = generate_monte_carlo_dataset(
        num_trials=num_trials,
        shape=shape,
        snr=snr,
        base_seed=seed,
    )

    # 3. Run Benchmark
    runner = ComparativeExperimentRunner(nominal_alpha=alpha, methods=methods)
    report = runner.run_benchmark(dataset=dataset)

    # Print publication summary table to console
    print(report.to_ascii_summary())

    # 4. Export Artifacts
    csv_path = report.export_csv(tables_dir / csv_filename)
    md_path = tables_dir / md_filename
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 8 Rigorous Experimental Comparison\n\n")
        f.write(f"**Monte Carlo Trials**: {num_trials}  \n")
        f.write(f"**Nominal Alpha**: {alpha}  \n")
        f.write(f"**Signal-to-Noise Ratio (SNR)**: {snr:.1f}  \n\n")
        f.write(report.to_markdown_table())
        f.write("\n\n---\n*Note: Evaluation conducted under identical volume noise realizations and coordinates.*\n")
    logger.info(f"Saved benchmark Markdown table to {md_path}")

    fig_path = plot_comparative_benchmark_summary(
        report=report,
        output_path=figures_dir / fig_filename,
        dpi=int(out_cfg.get("dpi", 300)),
    )

    logger.info("=" * 80)
    logger.info("PHASE 8 RIGOROUS EXPERIMENT BENCHMARK COMPLETE")
    logger.info(f"  Summary Table (Markdown): {md_path}")
    logger.info(f"  Summary Table (CSV):      {csv_path}")
    logger.info(f"  Publication Figure:       {fig_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
