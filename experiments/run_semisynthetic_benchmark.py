"""Experiment runner: Semi-synthetic benchmark injecting signals into realistic null fields."""

import argparse
from pathlib import Path
import numpy as np

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.data.schemas import SyntheticBenchmarkSample
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.reporter import generate_metrics_markdown_table, export_metrics_to_csv

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semi-synthetic neuroimaging benchmark.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nominal FDR level.")
    parser.add_argument("--snr", type=float, default=2.2, help="Injected activation signal amplitude.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments", help="Output directory.")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up semi-synthetic background noise field...")
    # Simulated resting-state empirical null with complex spatial autocorrelation
    from scipy.ndimage import gaussian_filter
    grid_size = (48, 48, 24)
    raw_noise = np.random.randn(*grid_size).astype(np.float32)
    # Mixture of multi-scale correlations representing physiological vascular noise
    smooth_noise_1 = gaussian_filter(raw_noise, sigma=1.2)
    smooth_noise_2 = gaussian_filter(raw_noise, sigma=3.0)
    resting_noise = 0.7 * smooth_noise_1 + 0.3 * smooth_noise_2
    # Standardize to empirical null
    resting_noise = (resting_noise - np.mean(resting_noise)) / (np.std(resting_noise) + 1e-8)

    # Inject activations in 3 realistic motor/sensory clusters
    cx, cy, cz = 18, 24, 12
    x, y, z = np.ogrid[:grid_size[0], :grid_size[1], :grid_size[2]]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    ground_truth = dist_sq <= (4.5 ** 2)
    signal = args.snr * np.exp(-dist_sq / (2.0 * 2.5 ** 2)) * ground_truth

    z_map = (resting_noise + signal).astype(np.float32)
    sample = SyntheticBenchmarkSample(
        z_map=z_map,
        ground_truth=ground_truth,
        noise_field=resting_noise,
        snr=args.snr,
        cluster_centers=[(cx, cy, cz)],
    )

    logger.info("Executing benchmark comparison on semi-synthetic volume...")
    runner = BenchmarkRunner(nominal_alpha=args.alpha, seed=args.seed)
    results = runner.run_single_comparison(sample)

    table_md = generate_metrics_markdown_table(results)
    print("\n" + "=" * 80)
    print("SEMI-SYNTHETIC BENCHMARK RESULTS:")
    print("=" * 80)
    print(table_md)
    print("=" * 80 + "\n")

    csv_dest = output_dir / f"semisynthetic_benchmark_alpha{args.alpha}.csv"
    export_metrics_to_csv(results, csv_dest)
    logger.info(f"Results saved to {csv_dest}")


if __name__ == "__main__":
    main()
