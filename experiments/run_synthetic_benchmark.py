"""Experiment runner: Controlled synthetic spatial benchmark across SNR levels."""

import argparse
from pathlib import Path
from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.reporter import generate_metrics_markdown_table, export_metrics_to_csv

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic spatial FDR benchmark.")
    parser.add_argument("--snr", type=float, default=2.0, help="Signal-to-noise ratio.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nominal FDR level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments", help="Output directory.")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic spatial field (SNR={args.snr}, seed={args.seed})...")
    sample = generate_synthetic_spatial_dataset(grid_size=(48, 48, 24), snr=args.snr, seed=args.seed)

    logger.info(f"Executing benchmark comparison under nominal alpha={args.alpha}...")
    runner = BenchmarkRunner(nominal_alpha=args.alpha, seed=args.seed)
    results = runner.run_single_comparison(sample)

    table_md = generate_metrics_markdown_table(results)
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY:")
    print("=" * 80)
    print(table_md)
    print("=" * 80 + "\n")

    csv_dest = output_dir / f"synthetic_benchmark_snr{args.snr}_alpha{args.alpha}.csv"
    export_metrics_to_csv(results, csv_dest)
    logger.info(f"Benchmark finished successfully. Results saved to {csv_dest}")


if __name__ == "__main__":
    main()
