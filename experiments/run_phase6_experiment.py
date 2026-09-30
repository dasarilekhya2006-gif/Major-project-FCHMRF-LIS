"""Phase 6 Controlled Experiment: Adaptive vs Fixed Hyperparameter Evaluation.

Executes:
1. Synthetic 3D validation and test volume generation with known ground truth (disjoint seeds).
2. Bayesian Optimization over spatial coupling and bilateral bandwidths on validation partition only.
3. Head-to-head evaluation of fixed baseline vs adaptively selected parameters on held-out test data.
4. Metric extraction: Power (TPR), FDP, F1 score, MCC, inference latency, optimization overhead.
5. Persistence of selected parameters to YAML, trial history to JSON, and summary plot to PNG.
"""

from pathlib import Path
import json
import time
import numpy as np

from optimization import (
    AcquisitionFunction,
    BaselineComparator,
    BayesianOptimizer,
    ObjectiveType,
    ValidationFDRBenchmarkObjective,
    create_fchmrf_search_space,
    plot_adaptive_optimization_summary,
)


def run_phase6_experiment(output_dir: Path = Path("experiments/outputs/phase6")):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 75)
    print("PHASE 6 CONTROLLED EXPERIMENT: ADAPTIVE BAYESIAN OPTIMIZATION VS FIXED BASELINE")
    print("=" * 75)

    # 1. Ground truth geometry: 14x14x6 volume with spherical activation cluster
    shape = (14, 14, 6)
    center = np.array([7, 7, 3])
    ground_truth = np.zeros(shape, dtype=bool)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if np.linalg.norm(np.array([x, y, z]) - center) <= 2.8:
                    ground_truth[x, y, z] = True

    n_active = int(np.sum(ground_truth))
    total_voxels = int(np.prod(shape))
    print(f"Domain Shape: {shape} ({total_voxels} voxels)")
    print(f"Active Cluster: {n_active} voxels ({100.0 * n_active / total_voxels:.1f}% of volume)")

    # 2. Validation volume (Seed 42)
    np.random.seed(42)
    z_val = np.random.randn(*shape)
    z_val[ground_truth] = 3.2 + 0.8 * np.random.randn(n_active)

    # 3. Independent Test volume (Seed 999 - different noise realization, slightly lower SNR)
    np.random.seed(999)
    z_test = np.random.randn(*shape)
    z_test[ground_truth] = 3.0 + 0.8 * np.random.randn(n_active)

    # 4. Search space definition
    search_space = create_fchmrf_search_space(
        include_lambda_dl=False,
        include_spatial_bandwidths=True,
    )
    print(f"Tunable Parameters: {search_space.names}")
    for p in search_space.parameters:
        print(f"  - {p.name}: bounds={p.bounds}, default={p.default_val}")

    # 5. Validation objective (Supervised Penalized Power with target FDR alpha=0.05)
    objective = ValidationFDRBenchmarkObjective(
        z_scores_val=z_val,
        ground_truth_val=ground_truth,
        objective_type=ObjectiveType.PENALIZED_POWER,
        target_alpha=0.05,
        max_iterations=20,
    )

    # 6. Bayesian Optimization
    print("\nExecuting Bayesian Optimization on Validation partition...")
    optimizer = BayesianOptimizer(
        search_space=search_space,
        objective=objective,
        acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
        xi=0.01,
        n_candidates=200,
        seed=42,
    )

    t0 = time.perf_counter()
    opt_result = optimizer.optimize(n_iterations=5, n_initial=3)
    bo_duration = time.perf_counter() - t0

    print(f"Bayesian Optimization finished in {bo_duration:.2f}s ({len(opt_result.trials)} total trials).")
    print(f"  Baseline Validation Score: {opt_result.baseline_score:.4f}")
    print(f"  Best Validation Score:     {opt_result.best_score:.4f} (improvement: {opt_result.score_improvement:+.4f})")
    print(f"  Selected Parameters:       {opt_result.best_parameters}")

    # 7. Persistence
    saved_files = opt_result.save_results(output_dir=output_dir, prefix="phase6_bo")

    # 8. Head-to-head comparison on Independent Test Data (zero data leakage)
    print("\nEvaluating on Independent Test Partition (zero data leakage)...")
    comparator = BaselineComparator(target_alpha=0.05)
    comp_report = comparator.compare(
        baseline_params=opt_result.baseline_parameters,
        selected_params=opt_result.best_parameters,
        z_scores_test=z_test,
        ground_truth_test=ground_truth,
        optimization_cost_seconds=opt_result.total_runtime_seconds,
    )

    comp_report.print_report()

    # 9. Save comparison report JSON
    comp_json_path = output_dir / "phase6_comparison_report.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comp_report.to_dict(), f, indent=2)

    # 10. Generate comparison plot
    fig_path = output_dir / "phase6_adaptive_vs_baseline.png"
    plot_adaptive_optimization_summary(
        opt_result=opt_result,
        comparison_report=comp_report,
        output_path=fig_path,
        dpi=120,
    )

    print(f"\nSaved Report: {comp_json_path}")
    print(f"Saved Plot:   {fig_path}")

    # Print clean summary table for reporting
    b_m = comp_report.baseline_metrics
    s_m = comp_report.selected_metrics
    print("\n" + "=" * 60)
    print("EXACT CONTROLLED EXPERIMENT METRICS")
    print("=" * 60)
    print(f"Metric                        | Fixed Baseline | Adaptive (BO)")
    print("-" * 60)
    print(f"Discoveries                   | {b_m.num_discoveries:<14} | {s_m.num_discoveries}")
    print(f"Statistical Power (TPR)       | {b_m.power:<14.4f} | {s_m.power:.4f}")
    print(f"Empirical FDP                 | {b_m.fdp:<14.4f} | {s_m.fdp:.4f}")
    print(f"F1 Score                      | {b_m.f1_score:<14.4f} | {s_m.f1_score:.4f}")
    print(f"Matthews Corr. Coeff. (MCC)   | {b_m.mcc:<14.4f} | {s_m.mcc:.4f}")
    print(f"Inference Latency (s)         | {b_m.inference_runtime_seconds:<14.3f} | {s_m.inference_runtime_seconds:.3f}")
    print(f"Optimization Overhead (s)     | {'0.000':<14} | {opt_result.total_runtime_seconds:.3f}")
    print(f"Measurable Benefit Detected   | {'N/A':<14} | {comp_report.is_benefit_measurable}")
    print("=" * 60)


if __name__ == "__main__":
    run_phase6_experiment()
