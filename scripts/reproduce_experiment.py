"""Dedicated CLI tool to reproduce a specific experiment from its configuration file.

Usage:
    python scripts/reproduce_experiment.py --config outputs/automated_experiments/exp_snr2.0_proposed_fchmrf_seed42/config.json
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.experiments.reproducibility import reproduce_experiment

logger = get_logger("reproduce_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-command exact experiment reproduction and verification."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to saved experiment configuration (config.json or config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional destination for reproduced output artifacts. Defaults to a subfolder next to config.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Floating-point numerical tolerance for metric verification.",
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

    logger.info(f"Initiating exact experiment reproduction from: {args.config}")
    new_result, is_match, delta_report = reproduce_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )

    if not is_match:
        logger.error("Reproduction check failed: numerical mismatch detected.")
        sys.exit(1)

    logger.info("Reproduction successfully verified! Bit-for-bit / scientific metrics matched.")
    sys.exit(0)


if __name__ == "__main__":
    main()
