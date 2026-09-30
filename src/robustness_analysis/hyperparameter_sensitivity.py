"""Hyperparameter sensitivity analysis for spatial coupling beta and DL prior weight lambda."""

from typing import Dict, List, Tuple
import numpy as np

from src.core.logging import get_logger
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.evaluation.metrics import EvaluationMetrics, compute_spatial_fdr_metrics
from src.preprocessing.spatial_graph import build_knn_spatial_adjacency
from src.statistical_analysis.mixture_models import fit_two_group_mixture_em
from src.fchmrf_lis.graph_potentials import compute_unary_potentials, compute_bilateral_weights
from src.fchmrf_lis.mean_field import run_mean_field_inference
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis
from src.deep_learning.predictor import ROIPriorPredictor

logger = get_logger(__name__)


def evaluate_hyperparameter_grid(
    beta_values: List[float] = [0.1, 0.4, 0.8, 1.2, 1.8],
    lambda_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    nominal_alpha: float = 0.05,
    seed: int = 42,
) -> Dict[Tuple[float, float], EvaluationMetrics]:
    """Sweep beta and lambda_dl on a benchmark volume to analyze stability."""
    logger.info("Executing hyperparameter sensitivity grid sweep.")
    sample = generate_synthetic_spatial_dataset(
        grid_size=(48, 48, 24),
        snr=2.0,
        seed=seed,
    )
    z_1d = sample.z_map.ravel()
    gt_1d = sample.ground_truth.ravel()

    coords = np.argwhere(np.ones(sample.z_map.shape, dtype=bool)).astype(np.float32)
    indices, dists = build_knn_spatial_adjacency(coords, k_neighbors=18)
    diffs = np.abs(z_1d[:, None] - z_1d[indices])
    weights = compute_bilateral_weights(dists, diffs)

    mixture = fit_two_group_mixture_em(z_1d, fix_null=True)
    predictor = ROIPriorPredictor()
    dl_prior = predictor.predict(sample.z_map).ravel()

    results: Dict[Tuple[float, float], EvaluationMetrics] = {}

    for beta in beta_values:
        for lam in lambda_values:
            u0, u1 = compute_unary_potentials(z_1d, mixture, dl_prior, lambda_dl=lam)
            q1, _, _ = run_mean_field_inference(
                u0, u1, indices, weights, beta=beta, max_iterations=20
            )
            lis = compute_lis(q1)
            rejected, _, _ = control_spatial_fdr_lis(lis, alpha=nominal_alpha)
            metrics = compute_spatial_fdr_metrics(rejected, gt_1d)
            results[(beta, lam)] = metrics

    return results
