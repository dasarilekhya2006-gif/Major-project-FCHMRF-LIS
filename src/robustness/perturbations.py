"""Perturbation generators and controlled parameter sweeps for neuroimaging robustness evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger
from src.evaluation.synthetic_benchmarks import (
    SyntheticBenchmarkVolume,
    generate_synthetic_benchmark_volume,
)

logger = get_logger(__name__)


def apply_additive_voxel_perturbation(
    volume: SyntheticBenchmarkVolume,
    noise_sigma: float = 0.2,
    seed: Optional[int] = None,
) -> SyntheticBenchmarkVolume:
    """Apply zero-mean Gaussian jitter to voxel test statistics:

    z_perturbed = z_orig + N(0, noise_sigma^2)
    """
    if noise_sigma <= 0:
        return volume

    rng = np.random.RandomState(seed if seed is not None else 42)
    jitter = rng.randn(*volume.shape).astype(np.float32) * noise_sigma
    jitter[~volume.spatial_mask] = 0.0

    z_pert = volume.z_scores + jitter

    from scipy import stats
    p_pert = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_pert)))
    p_pert = np.clip(p_pert, 1e-15, 1.0).astype(np.float32)

    return SyntheticBenchmarkVolume(
        z_scores=z_pert,
        p_values=p_pert,
        ground_truth=volume.ground_truth,
        dl_prior=volume.dl_prior,
        spatial_mask=volume.spatial_mask,
        snr=volume.snr,
        trial_id=volume.trial_id,
        shape=volume.shape,
    )


def apply_prior_misalignment_perturbation(
    volume: SyntheticBenchmarkVolume,
    translation_voxels: Tuple[int, int, int] = (1, 1, 0),
) -> SyntheticBenchmarkVolume:
    """Simulate spatial registration error or U-Net mislocalization by shifting the DL prior."""
    from scipy import ndimage

    shifted_prior = ndimage.shift(
        volume.dl_prior,
        shift=translation_voxels,
        order=1,
        mode="constant",
        cval=0.0,
    )
    shifted_prior[~volume.spatial_mask] = 0.0

    return SyntheticBenchmarkVolume(
        z_scores=volume.z_scores,
        p_values=volume.p_values,
        ground_truth=volume.ground_truth,
        dl_prior=shifted_prior.astype(np.float32),
        spatial_mask=volume.spatial_mask,
        snr=volume.snr,
        trial_id=volume.trial_id,
        shape=volume.shape,
    )


def generate_snr_sweep_dataset(
    snr_levels: List[float] = [1.5, 2.0, 2.5, 3.0, 4.0],
    num_repetitions_per_level: int = 5,
    shape: Tuple[int, int, int] = (24, 24, 12),
    base_seed: int = 42,
) -> Dict[float, List[SyntheticBenchmarkVolume]]:
    """Generate datasets across varying Signal-to-Noise Ratio (SNR) levels."""
    sweep_dict: Dict[float, List[SyntheticBenchmarkVolume]] = {}
    for snr in snr_levels:
        vols = []
        for rep in range(num_repetitions_per_level):
            seed = base_seed + int(snr * 100) + rep * 17
            vols.append(
                generate_synthetic_benchmark_volume(
                    shape=shape,
                    snr=snr,
                    trial_id=rep,
                    seed=seed,
                )
            )
        sweep_dict[snr] = vols
    return sweep_dict


def generate_spatial_correlation_sweep_dataset(
    smooth_sigmas: List[float] = [0.0, 0.8, 1.4, 2.0],
    num_repetitions: int = 5,
    snr: float = 3.0,
    shape: Tuple[int, int, int] = (24, 24, 12),
    base_seed: int = 42,
) -> Dict[float, List[SyntheticBenchmarkVolume]]:
    """Generate datasets across varying spatial autocorrelation bandwidths."""
    sweep_dict: Dict[float, List[SyntheticBenchmarkVolume]] = {}
    for sigma in smooth_sigmas:
        vols = []
        for rep in range(num_repetitions):
            seed = base_seed + int(sigma * 100) + rep * 23
            vols.append(
                generate_synthetic_benchmark_volume(
                    shape=shape,
                    snr=snr,
                    spatial_smooth_sigma=sigma,
                    trial_id=rep,
                    seed=seed,
                )
            )
        sweep_dict[sigma] = vols
    return sweep_dict


def generate_roi_size_sweep_dataset(
    radius_multipliers: List[float] = [0.6, 1.0, 1.4, 1.8],
    num_repetitions: int = 5,
    snr: float = 3.0,
    shape: Tuple[int, int, int] = (24, 24, 12),
    base_seed: int = 42,
) -> Dict[float, List[SyntheticBenchmarkVolume]]:
    """Generate datasets across varying cluster radii / ROI sizes."""
    base_radii = [3.2, 2.5]
    sweep_dict: Dict[float, List[SyntheticBenchmarkVolume]] = {}

    for mult in radius_multipliers:
        scaled_radii = [r * mult for r in base_radii]
        vols = []
        for rep in range(num_repetitions):
            seed = base_seed + int(mult * 100) + rep * 31
            vols.append(
                generate_synthetic_benchmark_volume(
                    shape=shape,
                    snr=snr,
                    cluster_radii=scaled_radii,
                    trial_id=rep,
                    seed=seed,
                )
            )
        sweep_dict[mult] = vols
    return sweep_dict
