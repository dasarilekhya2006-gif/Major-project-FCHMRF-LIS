"""Master CLI entry point for automated research experiments and batch suites."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.experiments.schemas import ExperimentConfig
from src.experiments.matrix import build_experiment_matrix, load_suite_from_yaml
from src.experiments.runner import ExperimentRunner
from src.experiments.reproducibility import reproduce_experiment

logger = get_logger("run_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Research Experiment Runner: Execute multiple configurations systematically."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Path to YAML suite configuration (e.g. configs/experiment_suite.yaml).",
    )
    group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to single experiment JSON or YAML configuration.",
    )
    group.add_argument(
        "--reproduce",
        type=str,
        default=None,
        help="Path to saved config.json to reproduce and verify exactly.",
    )

    parser.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=[1.5, 2.5],
        help="SNR levels to sweep if running matrix without a suite file.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["baseline_bh", "original_fchmrf", "proposed_fchmrf", "proposed_adaptive_fchmrf"],
        help="Methods to include in sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for execution.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nominal False Discovery Rate control level.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/automated_experiments",
        help="Directory where experiment outputs, plots, and CSV summaries are stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # 1. Single Command Reproduction Mode
    if args.reproduce:
        logger.info(f"Reproducing experiment from configuration: {args.reproduce}")
        new_result, is_match, delta_report = reproduce_experiment(
            config_path=args.reproduce,
            output_dir=None,
        )
        if not is_match:
            sys.exit(1)
        sys.exit(0)

    # 2. Single Explicit Configuration Run Mode
    if args.config:
        cfg_p = Path(args.config)
        logger.info(f"Executing single experiment configuration: {cfg_p}")
        if cfg_p.suffix in [".yaml", ".yml"]:
            config = ExperimentConfig.from_yaml(cfg_p)
        else:
            config = ExperimentConfig.from_json(cfg_p)

        runner = ExperimentRunner(base_output_dir=args.output_dir, device=args.device)
        result = runner.run_experiment(config)
        logger.info(f"Experiment {result.experiment_id} status: {result.status}")
        return

    # 3. Systematic Suite Execution Mode
    runner = ExperimentRunner(base_output_dir=args.output_dir, device=args.device)

    if args.suite:
        logger.info(f"Loading experiment suite from: {args.suite}")
        configs = load_suite_from_yaml(args.suite)
    else:
        logger.info(f"Building default systematic experiment matrix...")
        configs = build_experiment_matrix(
            snr_list=args.snr,
            methods=args.methods,
            seeds=[args.seed],
            alpha=args.alpha,
            base_output_dir=args.output_dir,
        )

    for cfg in configs:
        cfg.device = args.device

    results = runner.run_suite(configs, suite_output_dir=args.output_dir)
    num_completed = sum(1 for r in results if r.status == "completed")
    logger.info(f"Completed {num_completed}/{len(results)} experiments successfully.")


if __name__ == "__main__":
    main()
