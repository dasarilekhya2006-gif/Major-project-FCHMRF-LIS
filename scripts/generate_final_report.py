"""CLI Entry Point: Automated Final Research Report Generator (Phase 16)."""

import argparse
from pathlib import Path
import sys

from src.core.logging import setup_logging, get_logger
from src.reporting.data_collector import ResearchDataCollector
from src.reporting.generator import FinalResearchReportGenerator

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an exhaustive, publication-grade research report across all 16 canonical sections."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/final_report",
        help="Target directory to write report artifacts, tables, figures, and manuscripts.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.5],
        help="Signal-to-noise ratio levels to evaluate.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of independent random seeds / Monte Carlo iterations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nominal FDR significance threshold.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=3,
        default=[32, 32, 16],
        help="Spatial dimensions of 3D volume (X Y Z).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Hardware accelerator to use.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging("INFO")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42 + i * 101 for i in range(args.n_seeds)]
    grid_size = tuple(args.grid_size)

    print("\n" + "=" * 80)
    print("PHASE 16: AUTOMATED FINAL RESEARCH REPORT GENERATOR")
    print("=" * 80)
    print(f"Output Directory  : {output_dir}")
    print(f"SNR Levels        : {args.snr}")
    print(f"Seeds ({len(seeds)})      : {seeds}")
    print(f"Nominal Alpha     : {args.alpha}")
    print(f"Grid Dimensions   : {grid_size}")
    print(f"Device            : {args.device}")
    print("=" * 80 + "\n")

    # 1. Collect and synthesize actual experimental data
    collector = ResearchDataCollector(
        output_dir=output_dir,
        nominal_alpha=args.alpha,
        device=args.device,
    )
    report_data = collector.collect_or_run_evaluations(
        snr_levels=args.snr,
        seeds=seeds,
        grid_size=grid_size,
        run_benchmarks=True,
    )

    # 2. Compile publication reports
    generator = FinalResearchReportGenerator(
        data=report_data,
        output_dir=output_dir,
    )
    report_paths = generator.generate_all()

    print("\n" + "=" * 80)
    print("FINAL RESEARCH REPORT GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"  - Master Markdown Report : {report_paths['markdown']}")
    print(f"  - Publication LaTeX File : {report_paths['latex']}")
    print(f"  - Interactive HTML View  : {report_paths['html']}")
    print(f"  - Exported Table CSVs    : {output_dir / 'tables'}")
    print(f"  - Generated Figures      : {output_dir / 'figures'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
