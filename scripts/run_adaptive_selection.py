"""Command-line runner for Phase 6: adaptive hyperparameter selection for fcHMRF-LIS."""

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.fchmrf_lis.parameters import MethodMode
from src.adaptive_selection.search_space import create_fchmrf_search_space
from src.adaptive_selection.objectives import (
    ObjectiveType,
    ValidationFDRBenchmarkObjective,
    VariationalEvidenceObjective,
)
from src.adaptive_selection.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
)
from src.adaptive_selection.random_search import RandomSearchOptimizer
from src.adaptive_selection.baseline_comparator import BaselineComparator
from src.adaptive_selection.visualization import plot_adaptive_optimization_summary

logger = get_logger("run_adaptive_selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive hyperparameter selection (Bayesian Optimization) for fcHMRF-LIS."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adaptive_config.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["bayesian_optimization", "random_search"],
        help="Optimization algorithm.",
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default=None,
        choices=["expected_improvement", "upper_confidence_bound", "probability_of_improvement"],
        help="Acquisition function for Bayesian Optimization.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Number of guided optimization iterations.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Number of initial random trials.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        choices=["penalized_power", "f1_score", "mcc", "variational_elbo"],
        help="Objective function metric.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for logs and selected parameters.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=None,
        help="Destination directory for diagnostic figures.",
    )
    return parser.parse_args()


def generate_partitioned_neuroimaging_data(seed: int = 42):
    """Generate independent validation and test datasets to strictly prevent data leakage.

    Returns:
        (val_data, test_data) where each is a dict with 'z_scores', 'ground_truth', 'dl_prior'.
    """
    rng = np.random.RandomState(seed)
    shape = (20, 20, 10)

    def create_partition(center_coord, signal_mean=3.2):
        gt = np.zeros(shape, dtype=bool)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if np.linalg.norm(np.array([x, y, z]) - center_coord) <= 3.0:
                        gt[x, y, z] = True

        # Null ~ N(0, 1), Active ~ N(signal_mean, 1)
        z = rng.randn(*shape)
        z[gt] = signal_mean + 1.0 * rng.randn(int(np.sum(gt)))

        # Continuous DL prior with slight spatial blur and noise
        dl_prior = np.zeros(shape, dtype=np.float32)
        dl_prior[gt] = 0.85 + 0.1 * rng.randn(int(np.sum(gt)))
        dl_prior[~gt] = 0.05 + 0.05 * np.abs(rng.randn(int(np.sum(~gt))))
        dl_prior = np.clip(dl_prior, 0.0, 1.0)

        return {"z_scores": z, "ground_truth": gt, "dl_prior": dl_prior}

    # Validation partition (used solely for parameter search)
    val_data = create_partition(center_coord=np.array([7, 7, 5]), signal_mean=3.2)
    # Independent Test partition (used solely for final unbiased evaluation)
    test_data = create_partition(center_coord=np.array([12, 12, 5]), signal_mean=3.0)

    return val_data, test_data


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

    opt_cfg = cfg.get("adaptive_selection", {})
    obj_cfg = opt_cfg.get("objective", {})
    out_cfg = cfg.get("outputs", {})

    # Resolve settings
    seed = int(opt_cfg.get("seed", 42))
    set_seed(seed)

    algo = args.algorithm or opt_cfg.get("algorithm", "bayesian_optimization")
    acq_func = args.acquisition or opt_cfg.get("acquisition_function", "expected_improvement")
    n_iter = args.n_iter if args.n_iter is not None else int(opt_cfg.get("n_iterations", 15))
    n_init = args.n_init if args.n_init is not None else int(opt_cfg.get("n_initial", 5))
    obj_name = args.objective or obj_cfg.get("type", "penalized_power")
    target_alpha = float(obj_cfg.get("target_alpha", 0.05))
    fdr_weight = float(obj_cfg.get("fdr_penalty_weight", 4.0))
    tuning_max_iter = int(obj_cfg.get("tuning_max_iterations", 25))

    out_dir = Path(args.output_dir or out_cfg.get("output_dir", "outputs/adaptive_selection"))
    fig_dir = Path(args.fig_dir or out_cfg.get("figures_dir", "outputs/figures"))
    prefix = out_cfg.get("prefix", "fchmrf_adaptive")

    logger.info("=" * 70)
    logger.info("PHASE 6: ADAPTIVE PARAMETER SELECTION EXPERIMENT")
    logger.info(f"  Algorithm:            {algo}")
    logger.info(f"  Acquisition Function: {acq_func}")
    logger.info(f"  Objective Metric:     {obj_name}")
    logger.info(f"  Iterations:           {n_iter} guided + {n_init} initial")
    logger.info("=" * 70)

    # 1. Generate partitioned data to guarantee zero data leakage
    logger.info("Generating partitioned neuroimaging datasets (Validation vs. Test)...")
    val_data, test_data = generate_partitioned_neuroimaging_data(seed=seed)

    # 2. Setup Search Space
    search_space = create_fchmrf_search_space(
        include_lambda_dl=True,
        include_spatial_bandwidths=True,
    )
    logger.info(f"Search space constructed with {search_space.dimension} hyperparameters: {search_space.names}")

    # 3. Setup Objective Function (evaluated strictly on validation data)
    objective = ValidationFDRBenchmarkObjective(
        z_scores_val=val_data["z_scores"],
        ground_truth_val=val_data["ground_truth"],
        dl_prior_val=val_data["dl_prior"],
        target_alpha=target_alpha,
        objective_type=ObjectiveType(obj_name),
        fdr_penalty_weight=fdr_weight,
        max_iterations=tuning_max_iter,
    )

    # 4. Instantiate Optimizer
    if algo == "bayesian_optimization":
        optimizer = BayesianOptimizer(
            search_space=search_space,
            objective=objective,
            acquisition_function=AcquisitionFunction(acq_func),
            xi=float(opt_cfg.get("xi", 0.01)),
            kappa=float(opt_cfg.get("kappa", 2.0)),
            seed=seed,
        )
    else:
        optimizer = RandomSearchOptimizer(
            search_space=search_space,
            objective=objective,
            seed=seed,
        )

    # 5. Run Controlled Optimization
    opt_result = optimizer.optimize(n_iterations=n_iter, n_initial=n_init)

    # 6. Save selected parameters and trial history
    saved_paths = opt_result.save_results(output_dir=out_dir, prefix=prefix)

    # 7. Strictly Independent Baseline Comparison on Held-Out Test Data
    logger.info("Executing head-to-head comparison on held-out test partition...")
    comparator = BaselineComparator(
        method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
        target_alpha=target_alpha,
    )
    comp_report = comparator.compare(
        baseline_params=opt_result.baseline_parameters,
        selected_params=opt_result.best_parameters,
        z_scores_test=test_data["z_scores"],
        ground_truth_test=test_data["ground_truth"],
        dl_prior_test=test_data["dl_prior"],
        optimization_cost_seconds=opt_result.total_runtime_seconds,
    )

    # 8. Generate Summary Visualization
    fig_path = fig_dir / f"{prefix}_summary.png"
    plot_adaptive_optimization_summary(
        opt_result=opt_result,
        comparison_report=comp_report,
        output_path=fig_path,
        dpi=out_cfg.get("dpi", 300),
    )

    logger.info("=" * 70)
    logger.info("Phase 6 Adaptive Selection Completed Successfully:")
    logger.info(f"  Selected Parameters: {opt_result.best_parameters}")
    logger.info(f"  Validation Score:    {opt_result.best_score:.4f} (baseline: {opt_result.baseline_score:.4f})")
    logger.info(f"  Search Runtime:      {opt_result.total_runtime_seconds:.2f} seconds")
    logger.info(f"  Measurable Benefit:  {comp_report.is_benefit_measurable}")
    logger.info(f"  Figures & Artifacts: {fig_path}, {saved_paths['selected_yaml']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
