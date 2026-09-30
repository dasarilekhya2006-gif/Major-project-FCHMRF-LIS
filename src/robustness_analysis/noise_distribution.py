"""Robustness analysis against non-Gaussian and heavy-tailed null noise fields."""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter

from src.core.logging import get_logger
from src.core.seed import set_seed
from src.data.schemas import SyntheticBenchmarkSample
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.metrics import EvaluationMetrics

logger = get_logger(__name__)


def generate_heavy_tailed_sample(
    grid_size=(48, 48, 24),
    degrees_of_freedom: int = 4,
    snr: float = 2.0,
    seed: Optional[int] = None,
) -> SyntheticBenchmarkSample:
    """Generate a synthetic volume with heavy-tailed Student's t background noise.

    Standard Gaussian assumptions fail in heavy-tailed settings, leading to potential
    false discovery inflation if not properly handled by empirical null estimation.
    """
    if seed is not None:
        set_seed(seed)

    # Student-t noise: t_df / sqrt(df / (df - 2)) for unit variance
    raw_t = stats.t.rvs(df=degrees_of_freedom, size=grid_size).astype(np.float32)
    # Spatial correlation
    smooth_t = gaussian_filter(raw_t, sigma=1.5)
    noise_field = (smooth_t - np.mean(smooth_t)) / (np.std(smooth_t) + 1e-8)

    # Activate center cluster
    cx, cy, cz = grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    z = np.arange(grid_size[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    radius = 5.0

    ground_truth = dist_sq <= (radius ** 2)
    signal = snr * np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2)) * ground_truth

    z_map = (noise_field + signal).astype(np.float32)

    return SyntheticBenchmarkSample(
        z_map=z_map,
        ground_truth=ground_truth,
        noise_field=noise_field,
        snr=snr,
        cluster_centers=[(cx, cy, cz)],
    )


def evaluate_heavy_tailed_robustness(
    df_levels: List[int] = [3, 4, 8, 30],
    seed: int = 42,
) -> Dict[int, Dict[str, EvaluationMetrics]]:
    """Test performance across degrees of freedom df for Student-t background noise."""
    logger.info(f"Evaluating heavy-tailed null robustness across degrees of freedom: {df_levels}")
    runner = BenchmarkRunner(seed=seed)
    results: Dict[int, Dict[str, EvaluationMetrics]] = {}

    for df in df_levels:
        logger.info(f"Testing Student-t noise with df={df}...")
        sample = generate_heavy_tailed_sample(degrees_of_freedom=df, seed=seed)
        results[df] = runner.run_single_comparison(sample)

    return results
