"""Synthetic neuroimaging benchmark generator with known ground-truth activation regions."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage, stats

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticBenchmarkVolume:
    """A simulated 3D neuroimaging volume with known ground-truth activation status.

    Attributes:
        z_scores: 3D float array of voxel test statistics.
        p_values: 3D float array of two-sided unadjusted p-values.
        ground_truth: 3D boolean array indicating true active voxels.
        dl_prior: 3D float array in [0, 1] representing continuous 3D U-Net prior.
        spatial_mask: 3D boolean array indicating valid brain parenchyma voxels.
        snr: Signal-to-noise ratio of activation mean.
        trial_id: Integer identifier for this Monte Carlo trial.
        shape: Dimensions (X, Y, Z).
    """

    z_scores: np.ndarray
    p_values: np.ndarray
    ground_truth: np.ndarray
    dl_prior: np.ndarray
    spatial_mask: np.ndarray
    snr: float
    trial_id: int
    shape: Tuple[int, int, int]

    @property
    def num_active_voxels(self) -> int:
        return int(np.sum(self.ground_truth))

    @property
    def active_fraction(self) -> float:
        total = int(np.sum(self.spatial_mask))
        return float(self.num_active_voxels / max(1, total))


def generate_synthetic_benchmark_volume(
    shape: Tuple[int, int, int] = (24, 24, 12),
    snr: float = 3.0,
    spatial_smooth_sigma: float = 1.0,
    cluster_centers: Optional[List[Tuple[int, int, int]]] = None,
    cluster_radii: Optional[List[float]] = None,
    dl_accuracy: float = 0.85,
    trial_id: int = 0,
    seed: Optional[int] = None,
) -> SyntheticBenchmarkVolume:
    """Generate a reproducible synthetic neuroimaging volume with known ground truth.

    Models:
      1. Spatially correlated background noise via Gaussian-filtered white noise.
      2. Multi-focal ground-truth activation clusters with specified radii and SNR.
      3. Continuous 3D U-Net localization prior with realistic spatial boundary uncertainty.

    Args:
        shape: 3D volume dimensions (X, Y, Z).
        snr: Signal-to-noise ratio (mean activation intensity).
        spatial_smooth_sigma: Bandwidth for background spatial correlation.
        cluster_centers: List of (x, y, z) cluster centroids.
        cluster_radii: List of cluster radii in voxel units.
        dl_accuracy: Expected Dice-like localization fidelity of simulated 3D U-Net prior.
        trial_id: Simulation trial index.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticBenchmarkVolume instance.
    """
    rng = np.random.RandomState(seed if seed is not None else (42 + trial_id))

    # 1. Define default multi-focal cluster centers and radii
    if cluster_centers is None:
        cluster_centers = [
            (int(shape[0] * 0.35), int(shape[1] * 0.35), int(shape[2] * 0.5)),
            (int(shape[0] * 0.65), int(shape[1] * 0.65), int(shape[2] * 0.5)),
        ]
    if cluster_radii is None:
        cluster_radii = [3.2, 2.5]

    # 2. Construct Ground Truth Active Regions
    gt = np.zeros(shape, dtype=bool)
    grid_x, grid_y, grid_z = np.ogrid[:shape[0], :shape[1], :shape[2]]

    for center, radius in zip(cluster_centers, cluster_radii):
        dist_sq = (
            (grid_x - center[0]) ** 2
            + (grid_y - center[1]) ** 2
            + (grid_z - center[2]) ** 2
        )
        gt |= dist_sq <= (radius ** 2)

    # 3. Brain spatial mask (ellipsoidal parenchyma)
    center_vol = np.array(shape) / 2.0
    axes = np.array(shape) * 0.44
    ellipsoid = (
        ((grid_x - center_vol[0]) / axes[0]) ** 2
        + ((grid_y - center_vol[1]) / axes[1]) ** 2
        + ((grid_z - center_vol[2]) / axes[2]) ** 2
    )
    spatial_mask = ellipsoid <= 1.0

    # Ensure ground truth is strictly inside brain mask
    gt &= spatial_mask

    # 4. Generate Spatially Correlated Background Noise ~ N(0, 1)
    raw_noise = rng.randn(*shape)
    if spatial_smooth_sigma > 0:
        smoothed_noise = ndimage.gaussian_filter(raw_noise, sigma=spatial_smooth_sigma)
        # Standardize smoothed noise to zero mean and unit variance
        noise = (smoothed_noise - np.mean(smoothed_noise)) / np.std(smoothed_noise)
    else:
        noise = raw_noise

    # 5. Inject Statistical Activation Signal
    z_scores = noise.copy()
    num_gt = int(np.sum(gt))
    if num_gt > 0:
        signal = snr + 0.5 * rng.randn(num_gt)
        z_scores[gt] += signal

    # Two-sided p-values from standard normal
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))
    p_values = np.clip(p_values, 1e-15, 1.0)

    # 6. Simulate Realistic 3D U-Net Prior pi_i^DL
    # Smooth ground truth to mimic probabilistic network outputs with boundary gradients
    smoothed_gt = ndimage.gaussian_filter(gt.astype(float), sigma=1.2)
    max_sm = np.max(smoothed_gt) if np.max(smoothed_gt) > 0 else 1.0
    dl_prior = smoothed_gt / max_sm

    # Add realistic calibration noise
    prob_noise = 0.08 * rng.randn(*shape)
    dl_prior = np.clip(dl_prior * dl_accuracy + prob_noise, 0.01, 0.99)
    # Background voxels have low prior
    dl_prior[~spatial_mask] = 0.0

    return SyntheticBenchmarkVolume(
        z_scores=z_scores.astype(np.float32),
        p_values=p_values.astype(np.float32),
        ground_truth=gt,
        dl_prior=dl_prior.astype(np.float32),
        spatial_mask=spatial_mask,
        snr=snr,
        trial_id=trial_id,
        shape=shape,
    )


def generate_monte_carlo_dataset(
    num_trials: int = 5,
    shape: Tuple[int, int, int] = (24, 24, 12),
    snr: float = 3.0,
    base_seed: int = 42,
) -> List[SyntheticBenchmarkVolume]:
    """Generate a batch of independent Monte Carlo benchmark volumes under identical parameters."""
    dataset = []
    for i in range(num_trials):
        seed = base_seed + i * 100
        vol = generate_synthetic_benchmark_volume(
            shape=shape,
            snr=snr,
            trial_id=i,
            seed=seed,
        )
        dataset.append(vol)
    return dataset
