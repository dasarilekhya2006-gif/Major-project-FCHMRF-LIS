"""Robustness analysis under misspecified or shifted empirical null distributions."""

from typing import Dict, List, Tuple
import numpy as np

from src.core.logging import get_logger
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.metrics import EvaluationMetrics

logger = get_logger(__name__)


def evaluate_null_misspecification(
    null_shifts: List[float] = [-0.4, -0.2, 0.0, 0.2, 0.4],
    variance_scales: List[float] = [0.8, 1.0, 1.25],
    seed: int = 42,
) -> Dict[Tuple[float, float], Dict[str, EvaluationMetrics]]:
    """Test resilience against unmodeled scanner drift, global blood flow, or correlated noise.

    Injects artificial shift (mu0) and scale (sigma0) into the synthetic null field:
        z_perturbed = (z - signal) * scale + shift + signal

    Args:
        null_shifts: Array of mean offsets to inject into the null background.
        variance_scales: Multipliers for null variance.
        seed: Random seed.

    Returns:
        Dict[Tuple[float, float], Dict[str, EvaluationMetrics]]: (shift, scale) -> metrics.
    """
    logger.info("Evaluating sensitivity to empirical null misspecification.")
    base_sample = generate_synthetic_spatial_dataset(
        grid_size=(48, 48, 24),
        snr=2.5,
        seed=seed,
    )
    runner = BenchmarkRunner(seed=seed)
    results: Dict[Tuple[float, float], Dict[str, EvaluationMetrics]] = {}

    signal_part = np.where(base_sample.ground_truth, base_sample.z_map - base_sample.noise_field, 0.0)

    for shift in null_shifts:
        for scale in variance_scales:
            perturbed_noise = base_sample.noise_field * scale + shift
            perturbed_z = (perturbed_noise + signal_part).astype(np.float32)

            perturbed_sample = base_sample
            perturbed_sample.z_map = perturbed_z

            metrics_dict = runner.run_single_comparison(perturbed_sample)
            results[(shift, scale)] = metrics_dict

    return results
