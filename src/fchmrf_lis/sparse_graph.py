"""Memory-bounded local lattice neighborhood graph for fcHMRF-LIS spatial inference.

This module constructs a sparse voxel-adjacency graph using integer lattice connectivity
(6-face, 18-edge, or 26-vertex neighbors) rather than a floating-point radius cKDTree.

Scientific rationale
--------------------
Mean-field variational inference on a Hidden Markov Random Field requires a pairwise
affinity matrix K where K_ij > 0 only for spatially interacting pairs. The original
dense formulation uses a Gaussian kernel with bandwidth sigma_s and truncates at 3*sigma_s.
For sigma_s = 2.5 voxels this gives a cutoff of 7.5 voxels, which (in 3D) includes
4/3 * pi * 7.5^3 ≈ 1770 neighbors per voxel → O(1770 * N) edges → std::bad_alloc
for large N.

The lattice local neighborhood replaces this with explicit integer voxel offsets:
  - connectivity=1 (6-face):   6  neighbors per voxel → 6 * N  edges
  - connectivity=2 (18-edge): 18  neighbors per voxel → 18 * N edges
  - connectivity=3 (26-vertex):26  neighbors per voxel → 26 * N edges

The bilateral Gaussian affinity weights are computed identically to the dense kernel
using actual Euclidean distances between voxel coordinates and z-score differences.
Only the graph topology is sparse; the weight computation is unchanged.

For a brain volume with N=544,528 valid voxels:
  - 26-connectivity: max 14.2M edges → ~455 MB float32 sparse  (fits in RAM)
  - Dense query at k=128: 69.7M edges → ~2.2 GB float32 dense  (OOM)

Memory guarantee: no N×N matrix is ever allocated. The CSR sparse matrix stores
exactly |E| nonzero values where |E| ≤ connectivity_degree * N.
"""

from __future__ import annotations

from typing import Dict, Literal, NamedTuple, Tuple
import numpy as np
from scipy.sparse import csr_matrix, diags

from src.core.logging import get_logger

logger = get_logger(__name__)

# Precomputed 3D integer offset templates for each connectivity level
_FACE_OFFSETS = np.array(
    [
        (+1, 0, 0), (-1, 0, 0),
        (0, +1, 0), (0, -1, 0),
        (0, 0, +1), (0, 0, -1),
    ],
    dtype=np.int32,
)  # 6-connectivity

_EDGE_OFFSETS = np.array(
    [d for d in np.ndindex(3, 3, 3) if 1 <= sum(abs(np.array(d) - 1) > 0) <= 2],
    dtype=np.int32,
) - 1  # 18-connectivity

_VERTEX_OFFSETS = np.array(
    [d for d in np.ndindex(3, 3, 3) if not all(np.array(d) == 1)],
    dtype=np.int32,
) - 1  # 26-connectivity


class SparseNeighborGraph(NamedTuple):
    """Sparse voxel neighborhood graph in CSR format.

    Attributes:
        K: (N, N) sparse CSR affinity matrix with bilateral Gaussian weights.
        num_edges: Total number of directed edges stored (nnz of K).
        connectivity: Integer label (6, 18, or 26) describing the neighborhood.
        valid_voxels: Number of active voxels N.
        estimated_memory_bytes: Estimated RAM for K (3 arrays of nnz float32/int32 elements).
        graph_stats: Dictionary of diagnostic statistics.
    """

    K: csr_matrix
    num_edges: int
    connectivity: int
    valid_voxels: int
    estimated_memory_bytes: int
    graph_stats: Dict


def build_lattice_neighbor_graph(
    voxel_indices: np.ndarray,
    z_scores: np.ndarray,
    volume_shape: Tuple[int, int, int],
    sigma_spatial: float = 2.5,
    sigma_intensity: float = 1.5,
    w_spatial: float = 1.0,
    w_pure_spatial: float = 0.2,
    sigma_pure_spatial: float = 4.0,
    connectivity: Literal[6, 18, 26] = 26,
    normalize_rows: bool = True,
) -> SparseNeighborGraph:
    """Build a sparse bilateral affinity graph using local lattice voxel adjacency.

    This is the memory-bounded replacement for cKDTree.query_pairs / cKDTree.query
    with large k. Instead of querying all voxels within a radius in floating-point
    space, neighbors are defined as integer-offset lattice neighbors (6, 18, or 26-
    connected). Each neighbor pair's affinity is computed with the same bilateral
    Gaussian kernel used in the original dense formulation.

    NO N×N dense matrix is allocated at any point. The result is a CSR sparse matrix
    with at most ``connectivity × N`` nonzero entries.

    Args:
        voxel_indices: (N, 3) integer voxel index coordinates (i, j, k) in the volume.
        z_scores: (N,) statistical z-scores aligned with voxel_indices.
        volume_shape: (I, J, K) spatial shape of the full 3D volume.
        sigma_spatial: Bandwidth for spatial Euclidean distance in the bilateral kernel.
        sigma_intensity: Bandwidth for z-score difference in the bilateral kernel.
        w_spatial: Weight for the bilateral (spatial × feature) kernel component.
        w_pure_spatial: Weight for the pure spatial smoothness kernel component.
        sigma_pure_spatial: Bandwidth for the pure spatial kernel component.
        connectivity: Lattice connectivity — 6 (face), 18 (edge), or 26 (vertex).
        normalize_rows: Whether to row-normalize K so each row sums to 1 (recommended).

    Returns:
        SparseNeighborGraph: Named tuple containing sparse CSR matrix K, edge count,
            connectivity label, memory estimate, and diagnostic statistics.

    Raises:
        ValueError: If connectivity is not 6, 18, or 26.
    """
    if connectivity == 6:
        offsets = _FACE_OFFSETS
    elif connectivity == 18:
        offsets = _EDGE_OFFSETS
    elif connectivity == 26:
        offsets = _VERTEX_OFFSETS
    else:
        raise ValueError(f"connectivity must be 6, 18, or 26; got {connectivity}")

    N = len(z_scores)
    vi = np.asarray(voxel_indices, dtype=np.int32)  # (N, 3)
    z = np.asarray(z_scores, dtype=np.float32)

    # Build a fast lookup: voxel index tuple → position in valid voxel array
    # We use a 3D label volume: valid_voxel_label[i, j, k] = position in [0..N-1]
    # -1 means this voxel is not in the active mask.
    label_vol = np.full(volume_shape, -1, dtype=np.int32)
    label_vol[vi[:, 0], vi[:, 1], vi[:, 2]] = np.arange(N, dtype=np.int32)

    I_list = []
    J_list = []

    for offset in offsets:
        neighbor_vi = vi + offset[np.newaxis, :]  # (N, 3)

        # Boundary check: all three coordinates must be within volume bounds
        in_bounds = (
            (neighbor_vi[:, 0] >= 0) & (neighbor_vi[:, 0] < volume_shape[0]) &
            (neighbor_vi[:, 1] >= 0) & (neighbor_vi[:, 1] < volume_shape[1]) &
            (neighbor_vi[:, 2] >= 0) & (neighbor_vi[:, 2] < volume_shape[2])
        )

        # Look up neighbor positions in label volume
        # np.where needed to avoid out-of-bounds indexing on clamped coords
        nvi_clamped = neighbor_vi.copy()
        nvi_clamped[:, 0] = np.clip(nvi_clamped[:, 0], 0, volume_shape[0] - 1)
        nvi_clamped[:, 1] = np.clip(nvi_clamped[:, 1], 0, volume_shape[1] - 1)
        nvi_clamped[:, 2] = np.clip(nvi_clamped[:, 2], 0, volume_shape[2] - 1)

        j_positions = label_vol[nvi_clamped[:, 0], nvi_clamped[:, 1], nvi_clamped[:, 2]]

        # Keep only pairs where:
        #  1. The neighbor coordinate is within the volume bounds
        #  2. The neighbor voxel is in the active mask (j_positions >= 0)
        valid = in_bounds & (j_positions >= 0)

        i_valid = np.where(valid)[0].astype(np.int32)
        j_valid = j_positions[valid]

        I_list.append(i_valid)
        J_list.append(j_valid)

    if not I_list:
        K_empty = csr_matrix((N, N), dtype=np.float32)
        return SparseNeighborGraph(
            K=K_empty, num_edges=0, connectivity=connectivity,
            valid_voxels=N, estimated_memory_bytes=0,
            graph_stats={"warning": "no_edges_found"},
        )

    i_all = np.concatenate(I_list)
    j_all = np.concatenate(J_list)

    # --- Compute bilateral affinity weights ---
    # Euclidean distance between actual float coordinates (same formula as dense kernel)
    # Since coordinates are integer voxel indices, dist² is sum of squared offset integers
    diff = vi[i_all].astype(np.float32) - vi[j_all].astype(np.float32)
    dist_sq = np.sum(diff ** 2, axis=1)  # actual squared voxel-space distances
    z_diff_sq = (z[i_all] - z[j_all]) ** 2

    var_s = 2.0 * max(sigma_spatial, 1e-4) ** 2
    var_r = 2.0 * max(sigma_intensity, 1e-4) ** 2
    var_pure = 2.0 * max(sigma_pure_spatial, 1e-4) ** 2

    k_vals = (
        w_spatial * np.exp(-dist_sq / var_s - z_diff_sq / var_r)
        + w_pure_spatial * np.exp(-dist_sq / var_pure)
    ).astype(np.float32)

    # Build CSR matrix — no N×N dense allocation
    K = csr_matrix((k_vals, (i_all, j_all)), shape=(N, N), dtype=np.float32)

    # Row-normalize so each row's message weights sum to 1
    if normalize_rows:
        row_sums = np.array(K.sum(axis=1), dtype=np.float32).ravel()
        row_sums[row_sums == 0] = 1.0
        K = diags(1.0 / row_sums, format="csr").dot(K)

    num_edges = K.nnz
    # Memory estimate: three CSR arrays (data, indices, indptr) all float32 or int32
    estimated_bytes = num_edges * (4 + 4) + (N + 1) * 4  # data+col_idx + row_ptr

    min_deg = int(np.diff(K.indptr).min()) if N > 0 else 0
    max_deg = int(np.diff(K.indptr).max()) if N > 0 else 0
    mean_deg = float(num_edges / N) if N > 0 else 0.0

    graph_stats = {
        "connectivity": connectivity,
        "valid_voxels": N,
        "num_edges": num_edges,
        "min_degree": min_deg,
        "max_degree": max_deg,
        "mean_degree": round(mean_deg, 2),
        "estimated_memory_mb": round(estimated_bytes / 1e6, 2),
    }

    logger.info(
        f"Lattice neighbor graph built: N={N:,}, connectivity={connectivity}, "
        f"edges={num_edges:,}, mean_degree={mean_deg:.1f}, "
        f"memory≈{estimated_bytes / 1e6:.1f} MB"
    )

    return SparseNeighborGraph(
        K=K,
        num_edges=num_edges,
        connectivity=connectivity,
        valid_voxels=N,
        estimated_memory_bytes=estimated_bytes,
        graph_stats=graph_stats,
    )
