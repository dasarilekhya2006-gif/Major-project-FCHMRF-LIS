"""Synthetic spatial random field and cluster signal generator for benchmark simulation."""

from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from src.core.logging import get_logger
from src.core.seed import set_seed
from src.data.schemas import SyntheticBenchmarkSample

logger = get_logger(__name__)


def generate_synthetic_spatial_dataset(
    grid_size: Tuple[int, int, int] = (64, 64, 32),
    snr: float = 2.0,
    num_clusters: int = 4,
    cluster_radius_range: Tuple[int, int] = (3, 7),
    smooth_fwhm_voxels: float = 3.0,
    seed: Optional[int] = None,
) -> SyntheticBenchmarkSample:
    """Generate a reproducible synthetic 3D spatial field with ground truth activations.

    The model follows:
        Z_i = S_i + epsilon_i
    where epsilon_i is a zero-mean spatially correlated Gaussian random field with variance 1,
    and S_i is an activation signal with positive amplitudes located in geometric clusters.

    Args:
        grid_size: Dimensions of the 3D grid (X, Y, Z).
        snr: Peak signal-to-noise ratio in active clusters.
        num_clusters: Number of active spatial foci/clusters to seed.
        cluster_radius_range: Min and max radius of spherical active regions in voxels.
        smooth_fwhm_voxels: Spatial smoothing kernel FWHM for generating spatial noise correlation.
        seed: Random seed for exact reproducibility.

    Returns:
        SyntheticBenchmarkSample containing the simulated z_map, ground_truth mask, and noise field.
    """
    if seed is not None:
        set_seed(seed)

    logger.debug(f"Generating synthetic 3D volume of shape {grid_size} with SNR={snr}")

    # 1. Generate spatially correlated background noise (Gaussian Random Field)
    white_noise = np.random.randn(*grid_size).astype(np.float32)
    # sigma = FWHM / sqrt(8 * ln(2)) ~= FWHM / 2.35482
    sigma = smooth_fwhm_voxels / 2.35482
    smooth_noise = gaussian_filter(white_noise, sigma=sigma)
    # Standardize smoothed noise to mean 0, variance 1 (standard normal empirical null)
    noise_field = (smooth_noise - np.mean(smooth_noise)) / (np.std(smooth_noise) + 1e-8)

    # 2. Generate ground truth active clusters
    ground_truth = np.zeros(grid_size, dtype=bool)
    signal_field = np.zeros(grid_size, dtype=np.float32)
    cluster_centers: List[Tuple[int, int, int]] = []

    # Coordinates meshgrid
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    z = np.arange(grid_size[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Seed clusters away from the outer boundaries with bounds check for small grids
    margin = max(cluster_radius_range[1] + 2, 5)
    for _ in range(num_clusters):
        low_x = min(margin, max(0, (grid_size[0] - 2) // 2))
        high_x = max(low_x + 1, grid_size[0] - low_x)
        cx = np.random.randint(low_x, high_x)

        low_y = min(margin, max(0, (grid_size[1] - 2) // 2))
        high_y = max(low_y + 1, grid_size[1] - low_y)
        cy = np.random.randint(low_y, high_y)

        low_z = min(margin, max(0, (grid_size[2] - 2) // 2))
        high_z = max(low_z + 1, grid_size[2] - low_z)
        cz = np.random.randint(low_z, high_z)
        cluster_centers.append((cx, cy, cz))

        radius = np.random.uniform(cluster_radius_range[0], cluster_radius_range[1])
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        cluster_mask = dist_sq <= (radius ** 2)

        # Smooth signal profile within cluster (decaying towards boundary)
        cluster_signal = snr * np.exp(-dist_sq / (2 * (radius / 2.0) ** 2))
        signal_field = np.maximum(signal_field, cluster_signal * cluster_mask)
        ground_truth = ground_truth | cluster_mask

    # 3. Combine noise and activation signal
    z_map = (noise_field + signal_field).astype(np.float32)

    logger.info(
        f"Synthetic volume created: active voxels = {np.sum(ground_truth)} / {np.prod(grid_size)} "
        f"({100 * np.mean(ground_truth):.2f}%), max z-score = {np.max(z_map):.2f}"
    )

    return SyntheticBenchmarkSample(
        z_map=z_map,
        ground_truth=ground_truth,
        noise_field=noise_field,
        snr=snr,
        cluster_centers=cluster_centers,
    )
