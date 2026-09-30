"""Dense bilateral kernel computation for fully connected Hidden Markov Random Field (fcHMRF).

Computes all-to-all voxel interactions without sparse graph truncation, preserving dense
spatial and feature affinity dependencies across the entire analysis domain.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from src.core.logging import get_logger

logger = get_logger(__name__)


def compute_dense_bilateral_matrix(
    coordinates: np.ndarray,
    z_scores: np.ndarray,
    sigma_spatial: float = 2.5,
    sigma_intensity: float = 1.5,
    w_spatial: float = 1.0,
    w_pure_spatial: float = 0.2,
    sigma_pure_spatial: float = 4.0,
) -> np.ndarray:
    """Compute the full dense N x N bilateral affinity kernel matrix K across all voxels.

    Kernel definition:
        K_{ij} = w_1 * exp(- ||p_i - p_j||^2 / (2 * sigma_s^2) - (z_i - z_j)^2 / (2 * sigma_r^2))
               + w_2 * exp(- ||p_i - p_j||^2 / (2 * sigma_pure^2))
        with K_{ii} = 0 (no self-loops).

    Args:
        coordinates: (N, 3) spatial coordinates of active voxels.
        z_scores: (N,) 1D array of statistical z-scores.
        sigma_spatial: Bandwidth for spatial Euclidean distance term.
        sigma_intensity: Bandwidth for radiometric / statistical feature term.
        w_spatial: Weight of bilateral contrast-sensitive term.
        w_pure_spatial: Weight of isotropic spatial smoothness term.
        sigma_pure_spatial: Bandwidth for pure spatial term.

    Returns:
        np.ndarray: (N, N) dense affinity matrix with zero diagonal.
    """
    N = len(z_scores)
    coords = np.asarray(coordinates, dtype=np.float64)
    z = np.asarray(z_scores, dtype=np.float64).ravel()

    if len(coords) != N:
        raise ValueError(f"Coordinates length ({len(coords)}) does not match z_scores length ({N})")

    # Pairwise spatial squared Euclidean distances
    dist_sq = cdist(coords, coords, metric="sqeuclidean")

    # Pairwise feature absolute squared differences
    z_diff_sq = (z[:, np.newaxis] - z[np.newaxis, :]) ** 2

    # Bilateral component: spatial proximity + statistical affinity
    k_bilateral = np.exp(
        -(dist_sq / (2.0 * max(sigma_spatial, 1e-4) ** 2))
        -(z_diff_sq / (2.0 * max(sigma_intensity, 1e-4) ** 2))
    )

    # Pure spatial component: smoothness across boundary
    k_spatial = np.exp(-(dist_sq / (2.0 * max(sigma_pure_spatial, 1e-4) ** 2)))

    # Composite dense kernel
    K = (w_spatial * k_bilateral + w_pure_spatial * k_spatial).astype(np.float32)

    # Enforce zero self-loops (no voxel passes messages to itself)
    np.fill_diagonal(K, 0.0)

    return K


def dense_bilateral_message_passing(
    coordinates: np.ndarray,
    z_scores: np.ndarray,
    belief_diff: np.ndarray,
    sigma_spatial: float = 2.5,
    sigma_intensity: float = 1.5,
    w_spatial: float = 1.0,
    w_pure_spatial: float = 0.2,
    sigma_pure_spatial: float = 4.0,
    chunk_size: int = 4096,
    normalize: bool = False,
) -> np.ndarray:
    """Compute dense message passing m = K * s across all voxel pairs.

    For moderate N (<= 8,000) when chunk_size >= N, computes K directly in memory.
    Otherwise, evaluates K in contiguous blocks of rows to maintain exact all-to-all
    connectivity without allocating massive N x N matrices in RAM (zero sparse truncation).

    Args:
        coordinates: (N, 3) spatial coordinates.
        z_scores: (N,) statistical features.
        belief_diff: (N,) belief difference vector s = 2 * q - 1 in [-1, 1].
        sigma_spatial: Spatial bandwidth.
        sigma_intensity: Feature bandwidth.
        w_spatial: Bilateral component weight.
        w_pure_spatial: Pure spatial component weight.
        sigma_pure_spatial: Pure spatial bandwidth.
        chunk_size: Block size for chunked evaluation on large domains.
        normalize: Whether to row-normalize affinity weights to sum to 1.

    Returns:
        np.ndarray: (N,) aggregated message vector m.
    """
    N = len(z_scores)
    s = np.asarray(belief_diff, dtype=np.float32)

    # Direct in-memory matrix-vector product for standard domains when not explicitly chunking
    if N <= 8000 and chunk_size >= N:
        K = compute_dense_bilateral_matrix(
            coordinates=coordinates,
            z_scores=z_scores,
            sigma_spatial=sigma_spatial,
            sigma_intensity=sigma_intensity,
            w_spatial=w_spatial,
            w_pure_spatial=w_pure_spatial,
            sigma_pure_spatial=sigma_pure_spatial,
        )
        if normalize:
            row_sums = np.sum(K, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            K = K / row_sums
        return (K @ s).astype(np.float32)

    # Blocked row-chunking for whole-brain domains: guarantees O(1) memory without sparse truncation
    coords = np.asarray(coordinates, dtype=np.float64)
    z = np.asarray(z_scores, dtype=np.float64).ravel()
    messages = np.zeros(N, dtype=np.float32)

    var_s = 2.0 * max(sigma_spatial, 1e-4) ** 2
    var_r = 2.0 * max(sigma_intensity, 1e-4) ** 2
    var_pure = 2.0 * max(sigma_pure_spatial, 1e-4) ** 2

    # Dynamically adjust chunk_size to avoid MemoryError on large N
    # We want (chunk_size * N * 8 bytes) to be around 1 GB (10^9 bytes)
    max_chunk_size = max(1, int(1e9 / (N * 8)))
    effective_chunk_size = min(chunk_size, max_chunk_size)

    for start_idx in range(0, N, effective_chunk_size):
        end_idx = min(start_idx + effective_chunk_size, N)
        coords_chunk = coords[start_idx:end_idx]
        z_chunk = z[start_idx:end_idx]

        # Compute pairwise distances from chunk rows to all N columns
        dist_sq_chunk = cdist(coords_chunk, coords, metric="sqeuclidean")
        z_diff_sq_chunk = (z_chunk[:, np.newaxis] - z[np.newaxis, :]) ** 2

        K_chunk = (
            w_spatial * np.exp(-(dist_sq_chunk / var_s) - (z_diff_sq_chunk / var_r))
            + w_pure_spatial * np.exp(-(dist_sq_chunk / var_pure))
        ).astype(np.float32)

        # Zero out diagonal elements in this chunk
        for i in range(start_idx, end_idx):
            K_chunk[i - start_idx, i] = 0.0

        # Normalize rows if requested
        if normalize:
            row_sums = np.sum(K_chunk, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            K_chunk = K_chunk / row_sums

        messages[start_idx:end_idx] = K_chunk @ s

    return messages
