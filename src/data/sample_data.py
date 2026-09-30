"""Utilities for generating and managing synthetic test NIfTI datasets.

Allows reproducible testing and pipeline evaluation without relying on private medical data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter

from src.core.logging import get_logger
from src.core.seed import set_seed
from src.data.schemas import NeuroimagingVolume
from src.data.loaders import save_nifti

logger = get_logger(__name__)


def create_sample_3d_statistical_nifti(
    output_path: Union[str, Path],
    shape: Tuple[int, int, int] = (32, 32, 16),
    voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    snr: float = 2.5,
    num_clusters: int = 2,
    seed: int = 42,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """Generate a realistic 3D statistical z-map NIfTI file with embedded active clusters.

    Args:
        output_path: Destination path (.nii or .nii.gz).
        shape: 3D spatial dimensions (nx, ny, nz).
        voxel_size: Spatial resolution in mm (dx, dy, dz).
        snr: Signal-to-noise ratio for activations.
        num_clusters: Number of active regions.
        seed: Random seed for reproducibility.
        affine: Optional 4x4 coordinate affine matrix.

    Returns:
        Path: Path to the generated NIfTI file.
    """
    set_seed(seed)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Background standard normal noise with spatial autocorrelation
    raw_noise = np.random.randn(*shape).astype(np.float32)
    smooth_noise = gaussian_filter(raw_noise, sigma=1.5)
    noise_field = (smooth_noise - np.mean(smooth_noise)) / (np.std(smooth_noise) + 1e-8)

    # 2. Add spherical activation clusters
    active_mask = np.zeros(shape, dtype=bool)
    signal_field = np.zeros(shape, dtype=np.float32)

    cx_centers = np.linspace(shape[0] * 0.3, shape[0] * 0.7, num_clusters)
    cy_centers = np.linspace(shape[1] * 0.3, shape[1] * 0.7, num_clusters)
    cz_center = shape[2] // 2
    radius = min(shape) * 0.12

    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]

    for i in range(num_clusters):
        cx, cy = cx_centers[i], cy_centers[i]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz_center) ** 2
        cluster_mask = dist_sq <= (radius**2)
        active_mask |= cluster_mask
        signal_field += snr * np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2)) * cluster_mask

    z_map = (noise_field + signal_field).astype(np.float32)

    # 3. Create canonical affine matrix (RAS+ coordinates with specified voxel size)
    if affine is None:
        affine_mat = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
        affine_mat[:3, 3] = [-shape[0] * voxel_size[0] / 2.0, -shape[1] * voxel_size[1] / 2.0, -shape[2] * voxel_size[2] / 2.0]
    else:
        affine_mat = affine.copy()

    vol = NeuroimagingVolume(
        data=z_map,
        affine=affine_mat,
        voxel_dimensions=voxel_size,
        mask=np.ones(shape, dtype=bool),
        subject_id=out_p.stem.replace(".nii", ""),
    )

    save_nifti(vol, out_p)
    logger.info(f"Created sample 3D statistical NIfTI at {out_p} with shape {shape}")
    return out_p


def create_sample_4d_fmri_nifti(
    output_path: Union[str, Path],
    spatial_shape: Tuple[int, int, int] = (24, 24, 12),
    num_timepoints: int = 15,
    tr: float = 2.0,
    voxel_size: Tuple[float, float, float] = (3.0, 3.0, 3.5),
    seed: int = 42,
) -> Path:
    """Generate a realistic 4D fMRI timeseries NIfTI file with temporal autocorrelation.

    Args:
        output_path: Destination path (.nii or .nii.gz).
        spatial_shape: 3D spatial dimensions (nx, ny, nz).
        num_timepoints: Number of temporal scans (TRs).
        tr: Repetition time in seconds.
        voxel_size: Spatial resolution in mm.
        seed: Random seed.

    Returns:
        Path: Path to the generated 4D NIfTI file.
    """
    set_seed(seed)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    full_shape = (*spatial_shape, num_timepoints)
    # Baseline mean anatomical signal
    mean_signal = 800.0 + 200.0 * np.random.rand(*spatial_shape).astype(np.float32)
    mean_signal = gaussian_filter(mean_signal, sigma=2.0)

    # AR(1) temporal noise
    ar_coef = 0.4
    timeseries = np.zeros(full_shape, dtype=np.float32)
    noise_prev = np.random.randn(*spatial_shape).astype(np.float32) * 15.0

    for t in range(num_timepoints):
        innov = np.random.randn(*spatial_shape).astype(np.float32) * 15.0
        noise_t = ar_coef * noise_prev + np.sqrt(1 - ar_coef**2) * innov
        timeseries[..., t] = mean_signal + noise_t
        noise_prev = noise_t

    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    zooms = (voxel_size[0], voxel_size[1], voxel_size[2], tr)

    vol = NeuroimagingVolume(
        data=timeseries,
        affine=affine,
        voxel_dimensions=zooms,
        subject_id=out_p.stem.replace(".nii", ""),
    )

    save_nifti(vol, out_p)
    logger.info(f"Created sample 4D fMRI NIfTI at {out_p} with shape {full_shape}")
    return out_p


def create_empty_nifti(
    output_path: Union[str, Path],
    shape: Tuple[int, int, int] = (16, 16, 8),
) -> Path:
    """Create an empty (all-zeros) NIfTI file for validation and error-handling testing."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    vol = NeuroimagingVolume(
        data=np.zeros(shape, dtype=np.float32),
        affine=np.eye(4),
        voxel_dimensions=(2.0, 2.0, 2.0),
    )
    save_nifti(vol, out_p)
    return out_p


def create_corrupt_nifti(output_path: Union[str, Path]) -> Path:
    """Create a truncated/corrupted binary file disguised as a .nii.gz for testing."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        # Write arbitrary invalid bytes that cannot be decoded as valid NIfTI or gzip
        f.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xffTRUNCATED_CORRUPT_NIFTI_PAYLOAD_BYTE_STREAM")
    return out_p


def create_nan_infested_nifti(
    output_path: Union[str, Path],
    shape: Tuple[int, int, int] = (16, 16, 8),
    nan_ratio: float = 0.5,
    seed: int = 42,
) -> Path:
    """Create a NIfTI volume containing a high proportion of NaN values."""
    set_seed(seed)
    out_p = Path(output_path)
    data = np.random.randn(*shape).astype(np.float32)
    nan_mask = np.random.rand(*shape) < nan_ratio
    data[nan_mask] = np.nan

    vol = NeuroimagingVolume(
        data=data,
        affine=np.eye(4),
        voxel_dimensions=(2.0, 2.0, 2.0),
    )
    save_nifti(vol, out_p)
    return out_p


def create_singular_affine_nifti(
    output_path: Union[str, Path],
    shape: Tuple[int, int, int] = (16, 16, 8),
) -> Path:
    """Create a NIfTI volume with a degenerate/singular affine matrix (det = 0)."""
    out_p = Path(output_path)
    data = np.random.randn(*shape).astype(np.float32)
    # Degenerate matrix: near-zero determinant (< 1e-6)
    singular_affine = np.diag([2.0, 2.0, 1e-8, 1.0]).astype(np.float64)

    vol = NeuroimagingVolume(
        data=data,
        affine=singular_affine,
        voxel_dimensions=(2.0, 2.0, 1e-8),
    )
    save_nifti(vol, out_p)
    return out_p


def initialize_sample_test_datasets(
    destination_dir: Union[str, Path] = "data/synthetic",
) -> Dict[str, Path]:
    """Populate a directory with a suite of standard synthetic NIfTI benchmark files.

    Returns:
        Dict[str, Path]: Mapping of sample dataset identifiers to generated file paths.
    """
    dest = Path(destination_dir)
    dest.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample_3d_zmap": create_sample_3d_statistical_nifti(dest / "sample_zmap_3d.nii.gz", snr=2.5),
        "sample_4d_fmri": create_sample_4d_fmri_nifti(dest / "sample_fmri_4d.nii.gz", num_timepoints=10),
    }
    logger.info(f"Initialized standard synthetic NIfTI suite in {dest}")
    return samples
