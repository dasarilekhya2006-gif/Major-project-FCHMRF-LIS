"""Bayesian Optimization engine for adaptive fcHMRF-LIS hyperparameter selection."""

from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.core.logging import get_logger
from src.adaptive_selection.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
)
from src.adaptive_selection.objectives import BaseObjective
from src.adaptive_selection.search_space import SearchSpace

logger = get_logger(__name__)


class AcquisitionFunction(str, Enum):
    """Supported Bayesian Optimization acquisition functions."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using Gaussian Process surrogate and acquisition functions.

    =============================================================================
    MATHEMATICAL FORMULATION:
    =============================================================================
    1. SURROGATE MODEL:
       The black-box objective f(x) is modeled as a Gaussian Process:
           f(x) ~ GP(m(x), k(x, x'))
       where k(x, x') is a Matern-5/2 kernel with automated automatic relevance
       determination (ARD) length-scales and observation noise variance:
           k(x, x') = sigma_f^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r) + sigma_n^2

    2. ACQUISITION FUNCTIONS:
       Let f^+ = max_{i} y_i be the best observed objective value so far.
       - Expected Improvement (EI):
           EI(x) = (mu(x) - f^+ - xi) * Phi(Z) + sigma(x) * phi(Z)
           where Z = (mu(x) - f^+ - xi) / sigma(x)
       - Upper Confidence Bound (UCB):
           UCB(x) = mu(x) + kappa * sigma(x)
       - Probability of Improvement (PI):
           PI(x) = Phi(Z)

    3. EXPLORATION VS EXPLOITATION:
       Controlled by hyperparameter xi (for EI/PI) or kappa (for UCB).
    =============================================================================
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: BaseObjective,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        xi: float = 0.01,
        kappa: float = 2.0,
        n_candidates: int = 1000,
        seed: int = 42,
    ) -> None:
        """Initialize Bayesian Optimizer.

        Args:
            search_space: Domain bounds and parameter mappings.
            objective: Evaluator providing scalar score and metrics on validation data.
            acquisition_function: EI, UCB, or PI.
            xi: Exploration bonus for EI/PI.
            kappa: Exploration weight for UCB.
            n_candidates: Number of random candidate points evaluated during acquisition search.
            seed: Random seed for reproducibility.
        """
        super().__init__(search_space=search_space, objective=objective, seed=seed)
        self.acq_type = AcquisitionFunction(acquisition_function)
        self.xi = xi
        self.kappa = kappa
        self.n_candidates = n_candidates
        self.gp: Optional[GaussianProcessRegressor] = None

    def _build_gp(self) -> GaussianProcessRegressor:
        """Instantiate a robust Matern-5/2 Gaussian Process with ARD length scales."""
        D = self.search_space.dimension
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(D),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.seed,
        )

    def _compute_acquisition(
        self,
        X_norm: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        """Evaluate acquisition values across candidate points in [0, 1]^D."""
        if self.gp is None:
            return np.zeros(len(X_norm))

        try:
            mu, sigma = self.gp.predict(X_norm, return_std=True)
        except Exception:
            return np.zeros(len(X_norm))

        sigma = np.maximum(sigma, 1e-8)

        if self.acq_type == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            improvement = mu - y_best - self.xi
            Z = improvement / sigma
            ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            return np.maximum(0.0, ei)

        elif self.acq_type == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return mu + self.kappa * sigma

        elif self.acq_type == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
            Z = (mu - y_best - self.xi) / sigma
            return stats.norm.cdf(Z)

        return mu

    def _acquire_next_point(
        self,
        X_observed: np.ndarray,
        y_observed: np.ndarray,
    ) -> np.ndarray:
        """Find the next point in [0, 1]^D maximizing the acquisition function."""
        D = self.search_space.dimension
        if self.gp is None:
            return self.rng.uniform(0.0, 1.0, size=D)

        y_best = float(np.max(y_observed))

        # 1. Random candidate pool across unit hypercube [0, 1]^D
        candidates = self.rng.uniform(0.0, 1.0, size=(self.n_candidates, D))
        acq_values = self._compute_acquisition(candidates, y_best=y_best)
        best_candidate_idx = int(np.argmax(acq_values))
        x_init = candidates[best_candidate_idx]

        # 2. Local gradient-based refinement via L-BFGS-B
        def neg_acq(x_vec: np.ndarray) -> float:
            x_2d = x_vec.reshape(1, -1)
            val = self._compute_acquisition(x_2d, y_best=y_best)[0]
            return -float(val)

        bounds = [(0.0, 1.0) for _ in range(D)]
        try:
            res = minimize(neg_acq, x0=x_init, bounds=bounds, method="L-BFGS-B")
            x_next = res.x if res.success else x_init
        except Exception:
            x_next = x_init

        return np.clip(x_next, 0.0, 1.0)

    def optimize(
        self,
        n_iterations: int = 20,
        n_initial: int = 5,
    ) -> OptimizationResult:
        """Run controlled Bayesian Optimization loop."""
        t_global_start = time.perf_counter()
        logger.info(
            f"Initiating Bayesian Optimization ({self.acq_type.value}): "
            f"n_init={n_initial}, n_iter={n_iterations}, D={self.search_space.dimension}"
        )

        trials: List[OptimizationTrial] = []
        X_history: List[np.ndarray] = []
        y_history: List[float] = []

        # -------------------------------------------------------------
        # Step 0: Baseline Evaluation (Predefined Defaults)
        # -------------------------------------------------------------
        default_params = self.search_space.get_default_parameters()
        logger.info("Evaluating baseline fixed parameters...")
        baseline_res = self.objective.evaluate(default_params)
        baseline_trial = OptimizationTrial(
            trial_idx=1,
            parameters=default_params,
            score=baseline_res.score,
            metrics=baseline_res.metrics,
            elapsed_seconds=baseline_res.elapsed_seconds,
        )
        trials.append(baseline_trial)
        X_history.append(self.search_space.to_normalized_vector(default_params))
        y_history.append(baseline_res.score)

        # -------------------------------------------------------------
        # Step 1: Initial Random Sampling (Exploration warm-up)
        # -------------------------------------------------------------
        for init_idx in range(n_initial):
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
            X_history.append(self.search_space.to_normalized_vector(rand_params))
            y_history.append(obj_res.score)
            logger.info(
                f"[Init {init_idx + 1}/{n_initial}] Trial {trial_idx}: "
                f"score={obj_res.score:.4f} (elapsed: {obj_res.elapsed_seconds:.2f}s)"
            )

        # -------------------------------------------------------------
        # Step 2: Sequential Bayesian Optimization Iterations
        # -------------------------------------------------------------
        for bo_idx in range(n_iterations):
            trial_idx = len(trials) + 1
            X_arr = np.array(X_history, dtype=np.float64)
            y_arr = np.array(y_history, dtype=np.float64)

            # Fit GP surrogate
            self.gp = self._build_gp()
            try:
                self.gp.fit(X_arr, y_arr)
            except Exception as e:
                logger.warning(f"GP fit encountered warning: {e}. Falling back to random candidate.")
                self.gp = None

            # Maximize acquisition function
            x_next_norm = self._acquire_next_point(X_arr, y_arr)
            candidate_params = self.search_space.from_normalized_vector(x_next_norm)

            # Evaluate objective on candidate parameters
            obj_res = self.objective.evaluate(candidate_params)

            trial = OptimizationTrial(
                trial_idx=trial_idx,
                parameters=candidate_params,
                score=obj_res.score,
                metrics=obj_res.metrics,
                elapsed_seconds=obj_res.elapsed_seconds,
            )
            trials.append(trial)
            X_history.append(x_next_norm)
            y_history.append(obj_res.score)

            best_so_far = max(y_history)
            logger.info(
                f"[BO {bo_idx + 1}/{n_iterations}] Trial {trial_idx}: score={obj_res.score:.4f} "
                f"(best={best_so_far:.4f}, elapsed={obj_res.elapsed_seconds:.2f}s)"
            )

        total_elapsed = time.perf_counter() - t_global_start

        # Determine best trial
        best_trial_idx = int(np.argmax(y_history))
        best_trial = trials[best_trial_idx]

        return OptimizationResult(
            optimizer_name="BayesianOptimization",
            best_parameters=best_trial.parameters,
            best_score=float(best_trial.score),
            baseline_parameters=baseline_trial.parameters,
            baseline_score=float(baseline_trial.score),
            trials=trials,
            total_runtime_seconds=total_elapsed,
        )
