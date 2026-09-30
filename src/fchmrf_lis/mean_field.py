"""Mean-Field variational inference engine for fully connected Hidden Markov Random Field (fcHMRF).

Implements memory-bounded coordinate-ascent mean-field variational inference using a
sparse local lattice neighborhood graph (6/18/26-connectivity) to approximate posterior
marginal probabilities Q(theta_i = 1) across all active voxels.

Key design change (memory safety)
----------------------------------
The previous implementation used ``cKDTree.query(k=neighbor_limit)`` with k=128,
yielding up to N*128 edge entries. For N=544,528 that is 69.7M pairs → std::bad_alloc.

The new implementation uses ``build_lattice_neighbor_graph`` from sparse_graph.py, which
constructs a CSR sparse matrix by iterating over integer voxel offsets (26 offsets for
26-connectivity). For N=544,528 and 26-connectivity the maximum edge count is
26 * 544,528 = 14.2M, stored as a CSR float32 sparse matrix (~115 MB). No N×N dense
matrix is ever allocated.

The bilateral affinity weights are computed with the same Gaussian kernel formula as the
original dense formulation, ensuring mathematical equivalence within the local neighborhood.

Mean-field update (unchanged):
    s_j = 2 * q_j(1) - 1   (belief difference, in [-1, 1])
    m_i = sum_{j in N(i)} K_ij * s_j   =   [K @ s]_i   (sparse matvec)
    delta_i = psi_i(0) - psi_i(1) + beta * m_i
    q_i^new(1) = sigmoid(delta_i)
    q_i^(t+1) = (1 - damping) * q_i^(t) + damping * q_i^new(1)
"""

from typing import Dict, Optional, Tuple
import numpy as np

from src.core.exceptions import MemoryLimitExceededError, ModelConvergenceError
from src.core.logging import get_logger
from src.fchmrf_lis.sparse_graph import build_lattice_neighbor_graph

logger = get_logger(__name__)


def run_dense_mean_field_inference(
    unary_0: np.ndarray,
    unary_1: np.ndarray,
    coordinates: np.ndarray,
    z_scores: np.ndarray,
    beta: float = 0.8,
    sigma_spatial: float = 2.5,
    sigma_intensity: float = 1.5,
    w_spatial: float = 1.0,
    w_pure_spatial: float = 0.2,
    sigma_pure_spatial: float = 4.0,
    max_iterations: int = 40,
    tolerance: float = 1e-4,
    damping: float = 0.5,
    neighbor_connectivity: int = 26,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    max_valid_voxels: int = 2000000,
    max_graph_edges: int = 100000000,
    # Legacy parameter — accepted for backward compatibility but not used
    neighbor_limit: Optional[int] = None,
) -> Tuple[np.ndarray, int, float, Dict]:
    """Execute sparse lattice Mean-Field variational inference.

    =============================================================================
    MATHEMATICAL DERIVATION (unchanged from dense formulation):
    =============================================================================
    Under the mean-field product distribution:
        Q(theta) = prod_{i=1}^N q_i(theta_i)

    The coordinate ascent update minimizing KL(Q || P) is:
        log (q_i(1) / q_i(0)) = psi_i(0) - psi_i(1) + beta * sum_{j in N(i)} K_ij * (q_j(1) - q_j(0))

    Since q_j(0) = 1 - q_j(1), the belief difference is s_j = 2 * q_j(1) - 1 in [-1, 1].
    With a sparse neighborhood N(i), message aggregation is:
        m_i = [K @ s]_i      (sparse matrix-vector product, O(|E|) time)

    The fixed-point step:
        Delta_i = psi_i(0) - psi_i(1) + beta * m_i
        q_i^new(1) = sigmoid(Delta_i)

    Damped update for numerical stability:
        q_i^(t+1) = (1 - damping) * q_i^(t) + damping * q_i^new(1)
    =============================================================================

    Graph construction:
    -------------------
    Neighbors are defined by the LOCAL LATTICE: for each voxel, we examine the
    (2^connectivity_dim - 1) integer-offset neighbors in the 3x3x3 cube. Only
    neighbors that are (a) within the volume bounds AND (b) also active in the
    spatial mask are included as edges. Affinity weights use the same bilateral
    Gaussian kernel as the original formulation.

    Memory bound:
        |E| <= neighbor_connectivity * N   (26 * N for full 3x3x3 minus center)
        CSR storage ≈ 8 * |E| bytes  (4B data + 4B col_idx) + (N+1)*4B row_ptr

    For N=544,528:  26 * 544,528 = 14.2M edges → ~113 MB sparse CSR.

    Args:
        unary_0: 1D array of null unary energies (N,).
        unary_1: 1D array of active unary energies (N,).
        coordinates: (N, 3) spatial coordinates of voxels. MUST be integer voxel
            indices (i, j, k) for lattice graph construction.
        z_scores: (N,) statistical z-scores.
        beta: Spatial coupling parameter.
        sigma_spatial: Spatial distance kernel bandwidth.
        sigma_intensity: Radiometric contrast kernel bandwidth.
        w_spatial: Weight of contrast-sensitive bilateral kernel.
        w_pure_spatial: Weight of pure spatial proximity kernel.
        sigma_pure_spatial: Bandwidth of pure spatial kernel.
        max_iterations: Maximum iterations before termination.
        tolerance: Stopping threshold for max absolute change in marginals.
        damping: Damping factor in (0.0, 1.0].
        neighbor_connectivity: Lattice connectivity — 6, 18, or 26.
        volume_shape: (I, J, K) shape of the full volume. If None, inferred from
            the maximum coordinate values + 1.
        max_valid_voxels: Safety ceiling on N.
        max_graph_edges: Safety ceiling on total edges.
        neighbor_limit: Ignored (legacy k-NN parameter kept for API compatibility).

    Returns:
        Tuple[np.ndarray, int, float, Dict]:
            - q1: (N,) posterior activation marginal probabilities Q(theta_i = 1).
            - iterations: Total iterations executed.
            - final_delta: Final maximum absolute change in marginal probabilities.
            - graph_stats: Dictionary with neighborhood diagnostics (edge count, memory, etc.)
    """
    N = len(unary_0)
    if N == 0:
        return np.array([], dtype=np.float64), 0, 0.0, {}

    # --- Pre-flight safety check (final guard, not primary solution) ---
    if N > max_valid_voxels:
        raise MemoryLimitExceededError(
            "fcHMRF-LIS: valid voxel count exceeds maximum configured limit.",
            {
                "valid_voxels": N,
                "max_valid_voxels": max_valid_voxels,
                "estimated_edges": neighbor_connectivity * N,
                "max_graph_edges": max_graph_edges,
                "neighbor_connectivity": neighbor_connectivity,
                "note": "Increase max_valid_voxels or reduce the brain mask.",
            },
        )

    estimated_edges = neighbor_connectivity * N
    if estimated_edges > max_graph_edges:
        raise MemoryLimitExceededError(
            "fcHMRF-LIS: estimated edge count exceeds configured maximum.",
            {
                "valid_voxels": N,
                "max_valid_voxels": max_valid_voxels,
                "estimated_edges": estimated_edges,
                "max_graph_edges": max_graph_edges,
                "neighbor_connectivity": neighbor_connectivity,
                "note": "Reduce neighbor_connectivity (26 → 18 → 6) or the brain mask.",
            },
        )

    # --- Infer volume shape if not provided ---
    voxel_indices = np.round(coordinates).astype(np.int32)
    if volume_shape is None:
        max_coords = voxel_indices.max(axis=0) + 1
        volume_shape = (int(max_coords[0]), int(max_coords[1]), int(max_coords[2]))

    # --- Report pre-construction graph statistics ---
    estimated_memory_mb = (estimated_edges * 8 + (N + 1) * 4) / 1e6
    logger.info(
        f"fcHMRF-LIS graph pre-report: N={N:,} valid voxels, "
        f"connectivity={neighbor_connectivity} (3D lattice), "
        f"max_edges={estimated_edges:,}, "
        f"estimated_memory≈{estimated_memory_mb:.1f} MB, "
        f"volume_shape={volume_shape}"
    )

    # --- Step 0: Initialize beliefs from independent Bayes posterior ---
    log_odds_init = unary_0 - unary_1
    q1 = 1.0 / (1.0 + np.exp(-np.clip(log_odds_init, -30.0, 30.0)))

    # --- Step 1: Build sparse local lattice neighbor graph ---
    # This is the core memory fix: no N×N dense matrix, no cKDTree.query_pairs,
    # no large radius search. Edge count is bounded by connectivity * N.
    graph = build_lattice_neighbor_graph(
        voxel_indices=voxel_indices,
        z_scores=z_scores,
        volume_shape=volume_shape,
        sigma_spatial=sigma_spatial,
        sigma_intensity=sigma_intensity,
        w_spatial=w_spatial,
        w_pure_spatial=w_pure_spatial,
        sigma_pure_spatial=sigma_pure_spatial,
        connectivity=neighbor_connectivity,
        normalize_rows=True,
    )

    K = graph.K  # CSR sparse matrix, |E| nonzeros

    # --- Step 2: Damped coordinate-ascent mean-field iterations ---
    final_iter = 0
    final_delta = 1.0

    for it in range(max_iterations):
        final_iter = it + 1
        s = (2.0 * q1 - 1.0).astype(np.float32)

        # Sparse message passing: m = K @ s — O(|E|) not O(N^2)
        m = K @ s

        # Total log-odds evidence
        delta_logits = (unary_0 - unary_1) + beta * m
        delta_logits_clamped = np.clip(delta_logits, -30.0, 30.0)

        # Sigmoid coordinate update
        q1_new = 1.0 / (1.0 + np.exp(-delta_logits_clamped))

        # Damped belief update
        q1_updated = (1.0 - damping) * q1 + damping * q1_new
        final_delta = float(np.max(np.abs(q1_updated - q1)))
        q1 = q1_updated

        if final_delta < tolerance:
            logger.debug(
                f"Sparse lattice Mean-Field converged at iteration {final_iter} "
                f"(delta={final_delta:.2e})"
            )
            break

    return q1.astype(np.float64), final_iter, final_delta, graph.graph_stats


def run_mean_field_inference(
    unary_0: np.ndarray,
    unary_1: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    beta: float = 0.8,
    max_iterations: int = 30,
    tolerance: float = 1e-4,
    damping: float = 0.5,
) -> Tuple[np.ndarray, int, float]:
    """Execute sparse Mean-Field variational inference with predefined neighbor graph.

    This function is retained for backward compatibility with the explicit
    neighbor_indices / neighbor_weights interface used in unit tests.
    """
    N = len(unary_0)
    log_diff = unary_0 - unary_1
    q1 = 1.0 / (1.0 + np.exp(-np.clip(log_diff, -30.0, 30.0)))

    row_sums = np.sum(neighbor_weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_weights = neighbor_weights / row_sums

    final_iter = 0
    delta = 1.0

    for it in range(max_iterations):
        final_iter = it + 1
        q0 = 1.0 - q1

        neighbor_q1 = q1[neighbor_indices]
        neighbor_q0 = q0[neighbor_indices]

        spatial_field_1 = beta * np.sum(norm_weights * neighbor_q1, axis=1)
        spatial_field_0 = beta * np.sum(norm_weights * neighbor_q0, axis=1)

        eff_1 = -unary_1 + spatial_field_1
        eff_0 = -unary_0 + spatial_field_0

        max_eff = np.maximum(eff_0, eff_1)
        exp_0 = np.exp(eff_0 - max_eff)
        exp_1 = np.exp(eff_1 - max_eff)
        q1_new = exp_1 / (exp_0 + exp_1)

        q1_updated = (1.0 - damping) * q1 + damping * q1_new
        delta = float(np.max(np.abs(q1_updated - q1)))
        q1 = q1_updated

        if delta < tolerance:
            break

    return q1, final_iter, delta
