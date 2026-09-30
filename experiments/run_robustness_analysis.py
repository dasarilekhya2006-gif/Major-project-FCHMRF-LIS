"""Experiment runner: Robustness and sensitivity stress-testing suite."""

import argparse
from pathlib import Path
from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.robustness_analysis.snr_perturbation import evaluate_snr_robustness
from src.robustness_analysis.noise_distribution import evaluate_heavy_tailed_robustness
from src.robustness_analysis.null_misspecification import evaluate_null_misspecification
from src.robustness_analysis.hyperparameter_sensitivity import evaluate_hyperparameter_grid

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness stress-testing suite.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/robustness")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STARTING ROBUSTNESS ANALYSIS SUITE")
    print("=" * 80)

    # 1. SNR Robustness
    logger.info("1/4: Running SNR degradation analysis...")
    snr_results = evaluate_snr_robustness(snr_levels=[0.8, 1.5, 2.5], seed=args.seed)
    print("\n[+] SNR Robustness (Adaptive DL-fcHMRF-LIS Power):")
    for snr, meth_dict in snr_results.items():
        m = meth_dict["Adaptive-DL-fcHMRF-LIS"]
        print(f"    SNR={snr:.1f} -> Emp FDR={m.empirical_fdr:.4f}, TPR={m.tpr:.4f}, F1={m.f1_score:.4f}")

    # 2. Heavy-tailed Student-t Noise
    logger.info("2/4: Running heavy-tailed Student-t stress test...")
    t_results = evaluate_heavy_tailed_robustness(df_levels=[4, 8, 30], seed=args.seed)
    print("\n[+] Heavy-Tailed Null Robustness:")
    for df, meth_dict in t_results.items():
        m = meth_dict["Adaptive-DL-fcHMRF-LIS"]
        print(f"    df={df} -> Emp FDR={m.empirical_fdr:.4f}, TPR={m.tpr:.4f}")

    # 3. Empirical Null Misspecification
    logger.info("3/4: Running empirical null misspecification test...")
    null_results = evaluate_null_misspecification(null_shifts=[-0.3, 0.0, 0.3], variance_scales=[1.0], seed=args.seed)
    print("\n[+] Empirical Null Shift Sensitivity:")
    for (shift, scale), meth_dict in null_results.items():
        m = meth_dict["Adaptive-DL-fcHMRF-LIS"]
        print(f"    shift={shift:+.1f}, scale={scale:.1f} -> Emp FDR={m.empirical_fdr:.4f}, TPR={m.tpr:.4f}")

    # 4. Hyperparameter Grid Sensitivity
    logger.info("4/4: Evaluating hyperparameter sensitivity grid (beta x lambda)...")
    grid_results = evaluate_hyperparameter_grid(
        beta_values=[0.4, 0.8, 1.4],
        lambda_values=[0.25, 0.5, 0.75],
        seed=args.seed,
    )
    print("\n[+] Hyperparameter Sensitivity Grid:")
    for (beta, lam), m in grid_results.items():
        print(f"    beta={beta:.1f}, lambda={lam:.2f} -> Emp FDR={m.empirical_fdr:.4f}, Power={m.tpr:.4f}, F1={m.f1_score:.4f}")

    print("\n" + "=" * 80)
    logger.info("Completed all robustness evaluations successfully.")


if __name__ == "__main__":
    main()
