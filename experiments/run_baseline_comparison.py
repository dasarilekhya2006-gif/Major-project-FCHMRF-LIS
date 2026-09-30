"""Experiment runner: Comparative baseline matrix across nominal alpha levels."""

import argparse
from pathlib import Path
from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.reporter import export_metrics_to_csv

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-alpha baseline comparison.")
    parser.add_argument("--snr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [0.01, 0.05, 0.10]
    sample = generate_synthetic_spatial_dataset(grid_size=(48, 48, 24), snr=args.snr, seed=args.seed)

    print("\n" + "=" * 90)
    print(f"MULTIPLE TESTING BASELINE COMPARISON (SNR = {args.snr})")
    print("=" * 90)
    print(f"{'Method':<25} | {'Alpha':<6} | {'Emp FDR':<8} | {'Power (TPR)':<12} | {'F1':<8} | {'MCC':<8}")
    print("-" * 90)

    for alpha in alphas:
        runner = BenchmarkRunner(nominal_alpha=alpha, seed=args.seed)
        results = runner.run_single_comparison(sample)
        for method, m in results.items():
            print(
                f"{method:<25} | {alpha:<6.2f} | {m.empirical_fdr:<8.4f} | "
                f"{m.tpr:<12.4f} | {m.f1_score:<8.4f} | {m.mcc:<8.4f}"
            )
            csv_path = output_dir / f"baseline_comparison_alpha{alpha}.csv"
            export_metrics_to_csv(results, csv_path)
        print("-" * 90)

    logger.info("Completed baseline comparison successfully.")


if __name__ == "__main__":
    main()
