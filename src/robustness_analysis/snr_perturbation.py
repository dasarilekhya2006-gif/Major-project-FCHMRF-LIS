"""Robustness evaluation under varying Signal-to-Noise Ratios (SNR)."""

from typing import Dict, List
import numpy as np

from src.core.logging import get_logger
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.metrics import EvaluationMetrics

logger = get_logger(__name__)


def evaluate_snr_robustness(
    snr_levels: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
    num_replications: int = 5,
    seed: int = 42,
) -> Dict[float, Dict[str, EvaluationMetrics]]:
    """Evaluate empirical FDR and TPR across degrading signal-to-noise ratios.

    Args:
        snr_levels: List of SNR amplitudes to evaluate.
        num_replications: Number of synthetic volumes per SNR level.
        seed: Random seed.

    Returns:
        Dict[float, Dict[str, EvaluationMetrics]]: SNR -> method -> aggregated metrics.
    """
    logger.info(f"Starting SNR robustness analysis across levels: {snr_levels}")
    runner = BenchmarkRunner(seed=seed)
    results_by_snr: Dict[float, Dict[str, EvaluationMetrics]] = {}

    for snr in snr_levels:
        logger.info(f"Testing SNR={snr}...")
        sample = generate_synthetic_spatial_dataset(
            grid_size=(48, 48, 24),
            snr=snr,
            seed=seed,
        )
        metrics_dict = runner.run_single_comparison(sample)
        results_by_snr[snr] = metrics_dict

    return results_by_snr
