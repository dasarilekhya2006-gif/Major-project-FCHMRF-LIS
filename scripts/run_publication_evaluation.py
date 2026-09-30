"""CLI Runner: Publication-Quality Scientific Evaluation & Ablation Suite."""

import argparse
from pathlib import Path
import sys

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.evaluation.publication_framework import PublicationEvaluationSuite

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run publication-quality scientific evaluation and systematic ablation studies."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/publication_evaluation",
        help="Directory to save publication artifacts, CSVs, LaTeX tables, and figures.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.5],
        help="Signal-to-noise ratio (SNR) conditions to evaluate (e.g. 1.5 2.0 2.5).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of independent random seeds / Monte Carlo iterations per SNR condition.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nominal false discovery rate (FDR) significance threshold.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=3,
        default=[32, 32, 16],
        help="Spatial 3D grid dimensions (X Y Z).",
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
    print("PHASE 14: PUBLICATION-QUALITY SCIENTIFIC EVALUATION & ABLATION STUDY")
    print("=" * 80)
    print(f"SNR Levels      : {args.snr}")
    print(f"Seeds ({len(seeds)})    : {seeds}")
    print(f"Nominal Alpha   : {args.alpha}")
    print(f"Grid Size       : {grid_size}")
    print(f"Device          : {args.device}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80 + "\n")

    suite = PublicationEvaluationSuite(
        nominal_alpha=args.alpha,
        output_dir=output_dir,
        device=args.device,
    )

    results_by_snr = suite.run_multi_snr_ablation_evaluation(
        snr_levels=args.snr,
        seeds=seeds,
        grid_size=grid_size,
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("Exported Publication Artifacts:")
    print(f"  - Metrics CSV        : {output_dir / 'publication_ablation_metrics.csv'}")
    print(f"  - Markdown Table     : {output_dir / 'publication_ablation_table.md'}")
    print(f"  - LaTeX Table        : {output_dir / 'publication_ablation_table.tex'}")
    print(f"  - Attribution Report : {output_dir / 'component_attribution_summary.md'}")
    print(f"  - Full Report        : {output_dir / 'scientific_evaluation_report.md'}")
    print(f"  - Figure 1 (Bars)    : {output_dir / 'fig1_ablation_performance_bars.png'}")
    print(f"  - Figure 2 (Waterfall: {output_dir / 'fig2_component_attribution_waterfall.png'}")
    print(f"  - Figure 3 (Curves)  : {output_dir / 'fig3_snr_sensitivity_curves.png'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
