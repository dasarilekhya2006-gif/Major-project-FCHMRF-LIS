"""Spatial graph construction and coordinate representation for HMRF and fcHMRF models."""

from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def extract_masked_coordinates(
    mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract grid coordinates (i, j, k) and physical mm coordinates for all in-mask voxels.

    Args:
        mask: 3D boolean mask array.
        affine: Optional 4x4 affine matrix. If None, uses identity.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - grid_coords: (N, 3) integer grid indices.
            - physical_coords: (N, 3) continuous world coordinates in mm.
    """
    if mask.ndim != 3:
        raise PreprocessingError(f"Mask must be 3D, got shape {mask.shape}")

    grid_indices = np.argwhere(mask)  # shape (N, 3)
    N = len(grid_indices)

    if affine is None:
        physical_coords = grid_indices.astype(np.float32)
    else:
        # Homogeneous coordinate transform: [x, y, z, 1] * affine^T
        homo = np.hstack([grid_indices, np.ones((N, 1), dtype=np.float32)])
        physical_coords = (homo @ affine.T)[:, :3].astype(np.float32)

    logger.debug(f"Extracted coordinates for {N} in-mask voxels.")
    return grid_indices, physical_coords


def build_knn_spatial_adjacency(
    coords: np.ndarray,
    k_neighbors: int = 26,
    max_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build spatial neighbor indices and Euclidean distances using a k-d tree.

    Args:
        coords: (N, 3) array of voxel coordinates.
        k_neighbors: Number of nearest neighbors to query for each voxel.
        max_radius: Optional maximum spatial cutoff distance in voxels or mm.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - neighbor_indices: (N, k) array of neighbor indices.
            - neighbor_distances: (N, k) array of Euclidean distances.
    """
    tree = cKDTree(coords)
    # Query k + 1 because neighbor 0 is the voxel itself with distance 0
    dists, indices = tree.query(coords, k=k_neighbors + 1)

    # Exclude self-loop at index 0
    neighbor_indices = indices[:, 1:]
    neighbor_distances = dists[:, 1:]

    if max_radius is not None:
        # Mask out neighbors beyond max_radius by setting distance to inf
        invalid = neighbor_distances > max_radius
        neighbor_distances[invalid] = np.inf

    logger.debug(f"Built spatial adjacency with {k_neighbors} neighbors for {len(coords)} nodes.")
    return neighbor_indices, neighbor_distances
