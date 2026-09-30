"""Random Search hyperparameter optimizer for comparative baselines."""

import time
from typing import List
import numpy as np

from src.core.logging import get_logger
from src.adaptive_selection.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
)
from src.adaptive_selection.objectives import BaseObjective
from src.adaptive_selection.search_space import SearchSpace

logger = get_logger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """Uniform random search optimizer baseline for fcHMRF-LIS tuning."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective: BaseObjective,
        seed: int = 42,
    ) -> None:
        super().__init__(search_space=search_space, objective=objective, seed=seed)

    def optimize(
        self,
        n_iterations: int = 20,
        n_initial: int = 0,
    ) -> OptimizationResult:
        """Execute random search iterations."""
        t_global_start = time.perf_counter()
        total_trials = n_iterations + n_initial
        logger.info(f"Initiating Random Search: total_trials={total_trials}")

        trials: List[OptimizationTrial] = []
        y_history: List[float] = []

        # Step 0: Baseline Evaluation
        default_params = self.search_space.get_default_parameters()
        baseline_res = self.objective.evaluate(default_params)
        baseline_trial = OptimizationTrial(
            trial_idx=1,
            parameters=default_params,
            score=baseline_res.score,
            metrics=baseline_res.metrics,
            elapsed_seconds=baseline_res.elapsed_seconds,
        )
        trials.append(baseline_trial)
        y_history.append(baseline_res.score)

        # Random iterations
        for i in range(total_trials):
            trial_idx = len(trials) + 1
            rand_params = self.search_space.sample_random(rng=self.rng)
            obj_res = self.objective.evaluate(rand_params)

            trial = OptimizationTrial(
                trial_idx=trial_idx,
                parameters=rand_params,
                score=obj_res.score,
                metrics=obj_res.metrics,
                elapsed_seconds=obj_res.elapsed_seconds,
            )
            trials.append(trial)
            y_history.append(obj_res.score)
            logger.info(
                f"[RandomSearch {i + 1}/{total_trials}] Trial {trial_idx}: "
                f"score={obj_res.score:.4f} (best={max(y_history):.4f})"
            )

        total_elapsed = time.perf_counter() - t_global_start
        best_trial_idx = int(np.argmax(y_history))
        best_trial = trials[best_trial_idx]

        return OptimizationResult(
            optimizer_name="RandomSearch",
            best_parameters=best_trial.parameters,
            best_score=float(best_trial.score),
            baseline_parameters=baseline_trial.parameters,
            baseline_score=float(baseline_trial.score),
            trials=trials,
            total_runtime_seconds=total_elapsed,
        )
