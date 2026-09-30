"""CLI script executing Phase 9 robustness and uncertainty evaluation sweeps."""

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.robustness.evaluator import RobustnessEvaluator
from src.robustness.visualization import plot_robustness_analysis_summary

logger = get_logger("run_robustness_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Phase 9 robustness and uncertainty evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/robustness_config.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Target nominal spatial FDR threshold (default: 0.05).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of Monte Carlo random seeds for uncertainty analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for tables and plots.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(log_level="INFO")
    args = parse_args()

    cfg_p = Path(args.config)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {cfg_p} not found. Using defaults.")
        cfg = {}

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    alpha = args.alpha if args.alpha is not None else float(cfg.get("nominal_alpha", 0.05))
    sweeps_cfg = cfg.get("sweeps", {})
    fchmrf_cfg = cfg.get("fchmrf_defaults", {})
    out_cfg = cfg.get("outputs", {})

    shape = tuple(sweeps_cfg.get("volume_shape", [20, 20, 10]))
    snr_levels = list(sweeps_cfg.get("snr_levels", [1.5, 2.5, 3.5]))
    pert_sigmas = list(sweeps_cfg.get("perturbation_sigmas", [0.0, 0.15, 0.35]))
    spatial_sigmas = list(sweeps_cfg.get("spatial_smooth_sigmas", [0.0, 1.0, 2.0]))
    lambda_dl_vals = list(sweeps_cfg.get("lambda_dl_values", [0.2, 0.5, 0.8]))
    num_seeds = args.num_seeds if args.num_seeds is not None else int(sweeps_cfg.get("num_multi_seed_runs", 6))

    tables_dir = Path(out_cfg.get("tables_dir", "outputs/tables"))
    figures_dir = Path(out_cfg.get("figures_dir", "outputs/figures"))
    csv_filename = out_cfg.get("csv_filename", "phase9_robustness_metrics.csv")
    md_filename = out_cfg.get("markdown_filename", "phase9_robustness_metrics.md")
    fig_filename = out_cfg.get("figure_filename", "phase9_robustness_summary.png")

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Evaluator
    evaluator = RobustnessEvaluator(
        nominal_alpha=alpha,
        default_beta=float(fchmrf_cfg.get("beta", 0.8)),
        default_sigma_spatial=float(fchmrf_cfg.get("sigma_spatial", 2.5)),
        default_sigma_intensity=float(fchmrf_cfg.get("sigma_intensity", 1.5)),
        default_lambda_dl=float(fchmrf_cfg.get("lambda_dl", 0.5)),
        max_iterations=int(fchmrf_cfg.get("max_iterations", 30)),
    )

    # 2. Run Comprehensive Robustness Sweeps
    report = evaluator.run_full_robustness_evaluation(
        snr_levels=snr_levels,
        perturbation_sigmas=pert_sigmas,
        spatial_sigmas=spatial_sigmas,
        lambda_dl_values=lambda_dl_vals,
        num_seeds=num_seeds,
        shape=shape,
    )

    # 3. Export Artifacts
    csv_path = report.export_csv(tables_dir / csv_filename)
    md_path = tables_dir / md_filename
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 9: Robustness and Uncertainty Evaluation Report\n\n")
        f.write(f"**Nominal Alpha**: {alpha}  \n")
        f.write(f"**Volume Dimensions**: {shape}  \n\n")
        f.write(report.to_markdown_table())
        f.write("\n\n---\n*Scientific Note: Robustness and stability are evaluated across controlled empirical perturbations.*\n")
    logger.info(f"Saved robustness Markdown report to {md_path}")

    fig_path = plot_robustness_analysis_summary(
        report=report,
        output_path=figures_dir / fig_filename,
        dpi=int(out_cfg.get("dpi", 300)),
    )

    logger.info("=" * 80)
    logger.info("PHASE 9 ROBUSTNESS AND UNCERTAINTY EVALUATION COMPLETE")
    logger.info(f"  Summary Table (Markdown): {md_path}")
    logger.info(f"  Summary Table (CSV):      {csv_path}")
    logger.info(f"  Publication Figure:       {fig_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
