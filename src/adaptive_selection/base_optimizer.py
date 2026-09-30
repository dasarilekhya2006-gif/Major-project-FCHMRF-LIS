"""Base optimizer interfaces, trial logs, and result structures for adaptive parameter selection."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml

from src.core.logging import get_logger
from src.adaptive_selection.objectives import BaseObjective, ObjectiveResult
from src.adaptive_selection.search_space import SearchSpace

logger = get_logger(__name__)


@dataclass
class OptimizationTrial:
    """Record of a single hyperparameter evaluation trial.

    Attributes:
        trial_idx: 1-based sequential trial index.
        parameters: Hyperparameter dictionary evaluated.
        score: Primary scalar objective score achieved.
        metrics: Detailed secondary metrics (e.g. power, fdp, elapsed time).
        elapsed_seconds: Computation duration in seconds.
        timestamp: ISO format evaluation timestamp.
    """

    trial_idx: int
    parameters: Dict[str, Any]
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class OptimizationResult:
    """Consolidated summary of an entire hyperparameter optimization experiment.

    Attributes:
        optimizer_name: Name of optimization algorithm used (e.g. 'BayesianOptimization', 'RandomSearch').
        best_parameters: Highest-scoring hyperparameter dictionary discovered.
        best_score: Highest objective score achieved.
        baseline_parameters: Predefined baseline fixed parameters.
        baseline_score: Objective score achieved by baseline parameters.
        trials: Full sequential history of all optimization trials.
        total_runtime_seconds: Total wall-clock time for optimization experiment.
    """

    optimizer_name: str
    best_parameters: Dict[str, Any]
    best_score: float
    baseline_parameters: Dict[str, Any]
    baseline_score: float
    trials: List[OptimizationTrial] = field(default_factory=list)
    total_runtime_seconds: float = 0.0

    @property
    def score_improvement(self) -> float:
        """Absolute improvement over fixed baseline (positive indicates gain)."""
        return self.best_score - self.baseline_score

    @property
    def percentage_improvement(self) -> float:
        """Percentage improvement over baseline."""
        if abs(self.baseline_score) < 1e-8:
            return 0.0
        return float(100.0 * (self.best_score - self.baseline_score) / abs(self.baseline_score))

    def save_results(self, output_dir: Union[str, Path], prefix: str = "adaptive_params") -> Dict[str, Path]:
        """Save best selected parameters as YAML and full trial history as JSON."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        # 1. Save best selected parameters to YAML (ready for downstream fcHMRF inference)
        yaml_path = out_p / f"{prefix}_selected.yaml"
        payload = {
            "adaptive_selection": {
                "optimizer": self.optimizer_name,
                "best_score": float(self.best_score),
                "baseline_score": float(self.baseline_score),
                "score_improvement": float(self.score_improvement),
                "percentage_improvement": float(self.percentage_improvement),
                "total_runtime_seconds": float(self.total_runtime_seconds),
                "selected_parameters": self.best_parameters,
            }
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
        paths["selected_yaml"] = yaml_path

        # 2. Save full trial history to JSON
        json_path = out_p / f"{prefix}_history.json"
        history_payload = {
            "optimizer_name": self.optimizer_name,
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "baseline_parameters": self.baseline_parameters,
            "baseline_score": self.baseline_score,
            "score_improvement": self.score_improvement,
            "total_runtime_seconds": self.total_runtime_seconds,
            "trials": [asdict(t) for t in self.trials],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)
        paths["history_json"] = json_path

        logger.info(f"Saved optimization results to {yaml_path} and {json_path}")
        return paths


class BaseOptimizer(ABC):
    """Abstract interface for all hyperparameter optimizers."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective: BaseObjective,
        seed: int = 42,
    ) -> None:
        self.search_space = search_space
        self.objective = objective
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def optimize(
        self,
        n_iterations: int = 20,
        n_initial: int = 5,
    ) -> OptimizationResult:
        """Run the optimization experiment and return the consolidated results."""
        pass
