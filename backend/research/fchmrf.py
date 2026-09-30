"""Research adapter for fcHMRF-LIS spatial modeling and FDR control."""

from typing import Any, Dict, Tuple
import numpy as np

from src.fchmrf_lis.fchmrf_model import FCHMRFModel
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.core.logging import get_logger

logger = get_logger(__name__)


def execute_fchmrf_lis(
    z_scores: np.ndarray,
    dl_prior: np.ndarray,
    brain_mask: np.ndarray,
    nominal_alpha: float = 0.05,
    beta: float = 0.8,
    lambda_dl: float = 0.5,
    max_iterations: int = 25,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Execute fully connected Hidden Markov Random Field with Local Index of Significance.

    Returns:
        (lis_map, significant_mask, model_summary)
    """
    logger.info("Executing fcHMRF-LIS spatial modeling...")
    params = FCHMRFParameters(
        method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
        beta=beta,
        sigma_spatial=2.5,
        sigma_intensity=1.5,
        lambda_dl=lambda_dl,
        alpha=nominal_alpha,
        max_iterations=max_iterations,
    )
    model = FCHMRFModel(params=params)
    result = model.fit_predict(
        z_scores=z_scores,
        dl_prior=dl_prior,
        spatial_mask=brain_mask,
    )

    summary = {
        "method_mode": result.method_mode,
        "num_analyzed_voxels": result.num_analyzed_voxels,
        "num_discoveries": result.num_discoveries,
        "fdr_results": result.fdr_results,
        "lis_threshold": result.fdr_results.get("lis_threshold", 0.0),
    }

    return result.lis_map.astype(np.float32), result.significant_mask.astype(bool), summary
