"""Comparative benchmarking framework across spatial and non-spatial FDR methods."""

from typing import Dict, List, Optional
import numpy as np

from src.core.logging import get_logger
from src.core.seed import set_seed
from src.data.schemas import SyntheticBenchmarkSample
from src.evaluation.metrics import EvaluationMetrics, compute_spatial_fdr_metrics
from src.preprocessing.normalization import convert_pvalues_to_zscores
from src.preprocessing.spatial_graph import build_knn_spatial_adjacency
from src.statistical_analysis.classical_fdr import benjamini_hochberg
from src.statistical_analysis.storey_qvalue import storey_qvalues
from src.statistical_analysis.local_fdr import efron_locfdr
from src.statistical_analysis.mixture_models import fit_two_group_mixture_em
from src.fchmrf_lis.graph_potentials import compute_unary_potentials, compute_bilateral_weights
from src.fchmrf_lis.mean_field import run_mean_field_inference
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis
from src.deep_learning.predictor import ROIPriorPredictor

logger = get_logger(__name__)


class BenchmarkRunner:
    """Executes side-by-side performance comparisons across statistical FDR methods."""

    def __init__(
        self,
        nominal_alpha: float = 0.05,
        k_neighbors: int = 18,
        seed: int = 42,
    ) -> None:
        self.nominal_alpha = nominal_alpha
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.prior_predictor = ROIPriorPredictor()

    def run_single_comparison(
        self,
        sample: SyntheticBenchmarkSample,
    ) -> Dict[str, EvaluationMetrics]:
        """Run all candidate methods on a single synthetic neuroimaging sample.

        Methods compared:
          1. Benjamini-Hochberg (BH-FDR)
          2. Storey's q-value (pFDR)
          3. Efron's empirical null locfdr
          4. Standard HMRF-LIS (spatial only, no DL prior: lambda_dl = 0.0)
          5. Adaptive DL-Assisted fcHMRF-LIS (combining DL prior + fcHMRF + LIS)

        Args:
            sample: SyntheticBenchmarkSample containing z_map and ground_truth.

        Returns:
            Dict[str, EvaluationMetrics]: Method name -> computed evaluation metrics.
        """
        set_seed(self.seed)
        z_map = sample.z_map
        gt = sample.ground_truth
        N = z_map.size

        # Flattened vectors
        z_1d = z_map.ravel()
        gt_1d = gt.ravel()

        # Compute 1-tailed p-values: p_i = 1 - Phi(z_i)
        from scipy import stats
        p_vals = 1.0 - stats.norm.cdf(z_1d)
        p_vals = np.clip(p_vals, 1e-15, 1.0)

        results: Dict[str, EvaluationMetrics] = {}

        # 1. Benjamini-Hochberg
        bh_rejected, _, _ = benjamini_hochberg(p_vals, alpha=self.nominal_alpha)
        results["BH-FDR"] = compute_spatial_fdr_metrics(bh_rejected, gt_1d)

        # 2. Storey's q-value
        storey_rej, _, _ = storey_qvalues(p_vals, alpha=self.nominal_alpha)
        results["Storey-q"] = compute_spatial_fdr_metrics(storey_rej, gt_1d)

        # 3. Efron locfdr
        locfdr_rej, _, _ = efron_locfdr(z_1d, threshold=self.nominal_alpha * 2)
        results["Efron-locfdr"] = compute_spatial_fdr_metrics(locfdr_rej, gt_1d)

        # Build coordinate grid for spatial methods
        grid_shape = z_map.shape
        coords = np.argwhere(np.ones(grid_shape, dtype=bool)).astype(np.float32)
        neighbor_indices, neighbor_dists = build_knn_spatial_adjacency(
            coords, k_neighbors=self.k_neighbors
        )

        # Compute bilateral weights
        # Intensity diffs
        intensity_diffs = np.abs(z_1d[:, None] - z_1d[neighbor_indices])
        weights = compute_bilateral_weights(
            distances=neighbor_dists,
            intensity_diffs=intensity_diffs,
            sigma_spatial=2.0,
            sigma_intensity=1.5,
        )

        # Fit two-group mixture emission
        mixture_params = fit_two_group_mixture_em(z_1d, fix_null=True)

        # 4. Standard HMRF-LIS (lambda_dl = 0.0, no DL assistance)
        u0_std, u1_std = compute_unary_potentials(
            z_scores=z_1d,
            mixture_params=mixture_params,
            dl_prior_p1=None,
            lambda_dl=0.0,
        )
        q1_std, _, _ = run_mean_field_inference(
            unary_0=u0_std,
            unary_1=u1_std,
            neighbor_indices=neighbor_indices,
            neighbor_weights=weights,
            beta=0.8,
        )
        lis_std = compute_lis(q1_std)
        std_hmrf_rej, _, _ = control_spatial_fdr_lis(lis_std, alpha=self.nominal_alpha)
        results["Standard-HMRF-LIS"] = compute_spatial_fdr_metrics(std_hmrf_rej, gt_1d)

        # 5. Adaptive DL-Assisted fcHMRF-LIS
        dl_prior_map = self.prior_predictor.predict(z_map)
        dl_prior_1d = dl_prior_map.ravel()

        u0_dl, u1_dl = compute_unary_potentials(
            z_scores=z_1d,
            mixture_params=mixture_params,
            dl_prior_p1=dl_prior_1d,
            lambda_dl=0.5,
        )
        q1_dl, _, _ = run_mean_field_inference(
            unary_0=u0_dl,
            unary_1=u1_dl,
            neighbor_indices=neighbor_indices,
            neighbor_weights=weights,
            beta=0.8,
        )
        lis_dl = compute_lis(q1_dl)
        dl_fchmrf_rej, _, _ = control_spatial_fdr_lis(lis_dl, alpha=self.nominal_alpha)
        results["Adaptive-DL-fcHMRF-LIS"] = compute_spatial_fdr_metrics(dl_fchmrf_rej, gt_1d)

        logger.info(f"Completed benchmark comparison across {len(results)} methods.")
        return results
