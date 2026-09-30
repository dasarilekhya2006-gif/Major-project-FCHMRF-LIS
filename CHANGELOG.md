# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **fcHMRF-LIS Memory Limit (`std::bad_alloc`)**: 
  - Identified the root cause: building an unbounded dense voxel-pair representation using `cKDTree.query_pairs(r=cutoff)` scales $O(N^2)$ and causes out-of-memory errors on realistically sized 3D neuroimaging volumes.
  - Replaced `query_pairs` with a mathematically sound bounded nearest-neighbor search (`cKDTree.query(k=neighbor_limit, distance_upper_bound=cutoff)`). This avoids creating huge unconstrained pair arrays by strictly bounding the maximum spatial degree of each voxel to `neighbor_limit`.
  - Added explicit pre-flight memory safety checks in `run_dense_mean_field_inference` that raise a structured `MemoryLimitExceededError` instead of silently crashing if the bounds are violated.
  - Updated `FCHMRFParameters` to expose memory safety variables (`neighbor_limit`, `max_valid_voxels`, `max_graph_edges`) for configuration.
  - Modified the FastAPI endpoint in `backend/api/fchmrf.py` to catch `MemoryLimitExceededError` and return an HTTP 500 JSON response containing code `FC_HMRF_MEMORY_LIMIT` and detailed statistics.
  - Enhanced the React frontend (`FCHMRFPage.jsx`) to gracefully parse and display the structured memory limit error. The UI now shows the valid voxels, estimated edges, cutoff radius, and configured limits, providing actionable user options (e.g., reduce spatial cutoff, increase ROI threshold, or use a smaller volume).
  - Ensured spatial characteristics (affine, shape, spacing) and FDR statistical definitions remain unchanged. The new `neighbor_limit` mathematically approximates a fully dense continuous space if connectivity exceeds $k$, preserving theoretical consistency while bounding computational explosion.
