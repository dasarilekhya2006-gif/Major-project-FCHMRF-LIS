"""Fully Connected Hidden Markov Random Field with Local Index of Significance (fcHMRF-LIS).

Master model orchestrator implementing dense spatial statistical inference, variational mean-field
state estimation, LIS computation, and exact spatial False Discovery Rate control.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import nibabel as nib
import numpy as np

from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.core.logging import get_logger
from src.fchmrf_lis.dense_kernel import compute_dense_bilateral_matrix
from src.fchmrf_lis.graph_potentials import compute_unary_potentials
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.mean_field import run_dense_mean_field_inference
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis
from src.statistical_analysis.mixture_models import (
    TwoGroupMixtureParameters,
    fit_two_group_mixture_em,
)

logger = get_logger(__name__)


@dataclass
class FCHMRFResult:
    """Encapsulates the complete outputs from fcHMRF-LIS spatial inference.

    Attributes:
        lis_map: 3D spatial volume of Local Index of Significance values in [0, 1].
        lis_vector: 1D vector of in-mask LIS values.
        significant_mask: 3D binary volume of FDR-controlled discoveries.
        significant_vector: 1D boolean vector of in-mask discoveries.
        posterior_prob: 3D volume of active state posterior marginals Q(theta_i = 1).
        fdr_results: Dictionary containing nominal_alpha, estimated_fdr, num_discoveries, lis_threshold.
        model_parameters: Dictionary containing fitted mixture parameters and inference metrics.
        spatial_mask: 3D boolean mask of active domain analyzed.
        affine: 4x4 spatial transformation matrix for NIfTI coordinate preservation.
        method_mode: Execution mode ('original_fchmrf' or 'adaptive_dl_fchmrf').
    """

    lis_map: np.ndarray
    lis_vector: np.ndarray
    significant_mask: np.ndarray
    significant_vector: np.ndarray
    posterior_prob: np.ndarray
    fdr_results: Dict[str, Any]
    model_parameters: Dict[str, Any]
    spatial_mask: np.ndarray
    affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    method_mode: str = "original_fchmrf"

    @property
    def num_discoveries(self) -> int:
        return int(np.sum(self.significant_vector))

    @property
    def num_analyzed_voxels(self) -> int:
        return len(self.lis_vector)

    @property
    def iterations_to_converge(self) -> int:
        return self.model_parameters.get("mean_field_iterations", 0)

    def save_nifti_results(
        self,
        output_dir: Union[str, Path],
        prefix: str = "fchmrf_lis",
    ) -> Dict[str, Path]:
        """Save spatial results as coordinate-aligned NIfTI files and export metadata JSON.

        Args:
            output_dir: Directory path where output volumes will be stored.
            prefix: Filename prefix for outputs.

        Returns:
            Dict[str, Path]: Mapping of result names to saved file paths.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        # 1. Save LIS Map
        lis_path = out_p / f"{prefix}_lis_map.nii.gz"
        nib.save(nib.Nifti1Image(self.lis_map.astype(np.float32), self.affine), str(lis_path))
        paths["lis_map"] = lis_path

        # 2. Save Significant Voxel Binary Mask
        sig_path = out_p / f"{prefix}_significant_mask.nii.gz"
        nib.save(nib.Nifti1Image(self.significant_mask.astype(np.uint8), self.affine), str(sig_path))
        paths["significant_mask"] = sig_path

        # 3. Save Posterior Probability Q(theta_i = 1) Map
        post_path = out_p / f"{prefix}_posterior_prob.nii.gz"
        nib.save(nib.Nifti1Image(self.posterior_prob.astype(np.float32), self.affine), str(post_path))
        paths["posterior_prob"] = post_path

        # 4. Save Metadata & FDR Summary JSON
        summary_path = out_p / f"{prefix}_summary.json"
        summary_payload = {
            "method_mode": self.method_mode,
            "num_analyzed_voxels": self.num_analyzed_voxels,
            "num_discoveries": self.num_discoveries,
            "discovery_percentage": float(100.0 * self.num_discoveries / max(1, self.num_analyzed_voxels)),
            "fdr_results": self.fdr_results,
            "model_parameters": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.model_parameters.items()
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        paths["summary_json"] = summary_path

        logger.info(f"Saved fcHMRF-LIS NIfTI and metadata outputs to {out_p}")
        return paths


class FCHMRFModel:
    """Fully Connected Hidden Markov Random Field with Local Index of Significance."""

    def __init__(self, params: Optional[FCHMRFParameters] = None) -> None:
        """Initialize the fcHMRF-LIS model.

        Args:
            params: Optional configuration hyperparameters. If None, default parameters are used.
        """
        self.params = params or FCHMRFParameters()

    def fit_predict(
        self,
        z_scores: Union[np.ndarray, Dict[str, Any]],
        coordinates: Optional[np.ndarray] = None,
        dl_prior: Optional[np.ndarray] = None,
        spatial_mask: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        mixture_params: Optional[TwoGroupMixtureParameters] = None,
    ) -> FCHMRFResult:
        """Execute full fcHMRF-LIS spatial modeling pipeline.

        Accepts either:
          - A dictionary formatted by Phase 4's StatisticalMaps.to_fchmrf_inputs(), OR
          - A 3D numpy array of z-scores, OR
          - A 1D in-mask z-scores vector with matching (N, 3) coordinates.

        Pipeline Steps:
          1. Data ingestion & active domain coordinate extraction.
          2. Two-group emission mixture estimation via EM (f_0 and f_1).
          3. Unary energy computation (distinguishing original vs DL-adaptive prior).
          4. Dense bilateral mean-field variational inference across all voxel pairs.
          5. Local Index of Significance (LIS) computation: LIS_i = 1 - Q(theta_i = 1).
          6. Step-up spatial False Discovery Rate control (Sun & Cai 2009).
          7. Spatial reconstruction into 3D NIfTI maps preserving original affine.

        Args:
            z_scores: Input data (dict from Phase 4, 3D array, or 1D array).
            coordinates: Optional (N, 3) spatial coordinates if z_scores is 1D.
            dl_prior: Optional 3D or 1D deep-learning prior probabilities pi_i in [0, 1].
            spatial_mask: Optional 3D boolean mask of active voxels.
            affine: 4x4 spatial transformation matrix.
            mixture_params: Pre-fitted TwoGroupMixtureParameters (optional).

        Returns:
            FCHMRFResult: Complete spatial maps and FDR-controlled discoveries.
        """
        # Step 1: Parse and validate inputs
        if isinstance(z_scores, dict):
            # Ingestion from Phase 4 StatisticalMaps.to_fchmrf_inputs()
            z_vec = np.asarray(z_scores["z_scores"], dtype=np.float64)
            coords = np.asarray(z_scores["voxel_coordinates"], dtype=np.float64)
            s_mask = np.asarray(z_scores["spatial_mask"], dtype=bool)
            vol_affine = z_scores.get("affine", affine if affine is not None else np.eye(4))
            spatial_shape = s_mask.shape
            dl_prior_vec = None
            if dl_prior is not None:
                if dl_prior.shape == spatial_shape:
                    dl_prior_vec = dl_prior[s_mask]
                else:
                    dl_prior_vec = np.asarray(dl_prior, dtype=np.float64)
        elif z_scores.ndim == 3:
            spatial_shape = z_scores.shape
            vol_affine = affine if affine is not None else np.eye(4)
            if spatial_mask is not None:
                s_mask = np.asarray(spatial_mask, dtype=bool)
            else:
                s_mask = np.isfinite(z_scores) & (z_scores != 0.0)

            z_vec = z_scores[s_mask]
            # Construct voxel coordinates grid
            grid_indices = np.argwhere(s_mask)
            coords = grid_indices.astype(np.float64)

            dl_prior_vec = None
            if dl_prior is not None:
                if dl_prior.shape == spatial_shape:
                    dl_prior_vec = dl_prior[s_mask]
                else:
                    dl_prior_vec = np.asarray(dl_prior, dtype=np.float64)
        elif z_scores.ndim == 1:
            z_vec = z_scores.astype(np.float64)
            if coordinates is None:
                raise DataValidationError("1D z-scores requires explicitly providing coordinates (N, 3).")
            coords = np.asarray(coordinates, dtype=np.float64)
            vol_affine = affine if affine is not None else np.eye(4)
            if spatial_mask is not None:
                s_mask = np.asarray(spatial_mask, dtype=bool)
                spatial_shape = s_mask.shape
            else:
                # Default minimal bounding box shape
                max_coords = np.max(coords.astype(int), axis=0) + 1
                spatial_shape = tuple(max_coords)
                s_mask = np.zeros(spatial_shape, dtype=bool)
                for c in coords.astype(int):
                    s_mask[c[0], c[1], c[2]] = True

            dl_prior_vec = np.asarray(dl_prior, dtype=np.float64) if dl_prior is not None else None
        else:
            raise DataValidationError(f"Unsupported z_scores array shape: {z_scores.shape}")

        N = len(z_vec)
        if N == 0:
            raise StatisticalInferenceError("Empty active domain for fcHMRF analysis.")

        logger.info(
            f"Starting fcHMRF-LIS spatial inference: domain={N:,} voxels, "
            f"mode={self.params.method_mode.value}, alpha={self.params.alpha}"
        )

        # Step 2: Fit two-group emission mixture density if not provided
        if mixture_params is None:
            mixture_params = fit_two_group_mixture_em(
                z_vec,
                fix_null=self.params.fix_null,
            )
            logger.info(
                f"Fitted emission mixture: p0={mixture_params.p0:.3f}, "
                f"mu0={mixture_params.mu0:.2f}, sigma0={mixture_params.sigma0:.2f}, "
                f"mu1={mixture_params.mu1:.2f}, sigma1={mixture_params.sigma1:.2f}"
            )

        # Step 3: Compute unary energy potentials
        unary_0, unary_1 = compute_unary_potentials(
            z_scores=z_vec,
            mixture_params=mixture_params,
            dl_prior_p1=dl_prior_vec,
            lambda_dl=self.params.lambda_dl,
            method_mode=self.params.method_mode,
        )

        # Step 4: Execute sparse lattice Mean-Field variational inference
        posterior_q1, mf_iters, mf_delta, graph_stats = run_dense_mean_field_inference(
            unary_0=unary_0,
            unary_1=unary_1,
            coordinates=coords,
            z_scores=z_vec,
            beta=self.params.beta,
            sigma_spatial=self.params.sigma_spatial,
            sigma_intensity=self.params.sigma_intensity,
            w_spatial=self.params.w_spatial,
            w_pure_spatial=self.params.w_pure_spatial,
            sigma_pure_spatial=self.params.sigma_spatial_pure,
            max_iterations=self.params.max_iterations,
            tolerance=self.params.tolerance,
            damping=self.params.damping,
            neighbor_connectivity=self.params.neighbor_connectivity,
            volume_shape=spatial_shape,
            max_valid_voxels=self.params.max_valid_voxels,
            max_graph_edges=self.params.max_graph_edges,
        )

        # Step 5: Compute Local Index of Significance (LIS)
        lis_vec = compute_lis(posterior_q1)

        # Step 6: Exact Spatial FDR control via step-up LIS ranking
        significant_vec, lis_thresh, est_fdr = control_spatial_fdr_lis(
            lis_values=lis_vec,
            alpha=self.params.alpha,
        )

        # Step 7: Spatial reconstruction back to 3D volumes
        lis_map = np.ones(spatial_shape, dtype=np.float64)
        lis_map[s_mask] = lis_vec

        significant_mask = np.zeros(spatial_shape, dtype=bool)
        significant_mask[s_mask] = significant_vec

        posterior_prob = np.zeros(spatial_shape, dtype=np.float64)
        posterior_prob[s_mask] = posterior_q1

        # Package complete outputs
        fdr_summary = {
            "nominal_alpha": float(self.params.alpha),
            "estimated_fdr": float(est_fdr),
            "num_discoveries": int(np.sum(significant_vec)),
            "lis_threshold": float(lis_thresh),
        }

        model_params_summary = {
            "p0": float(mixture_params.p0),
            "mu0": float(mixture_params.mu0),
            "sigma0": float(mixture_params.sigma0),
            "mu1": float(mixture_params.mu1),
            "sigma1": float(mixture_params.sigma1),
            "beta": float(self.params.beta),
            "sigma_spatial": float(self.params.sigma_spatial),
            "sigma_intensity": float(self.params.sigma_intensity),
            "lambda_dl": float(self.params.lambda_dl),
            "mean_field_iterations": int(mf_iters),
            "mean_field_final_delta": float(mf_delta),
            "neighbor_connectivity": int(self.params.neighbor_connectivity),
            "graph_stats": graph_stats,
        }

        return FCHMRFResult(
            lis_map=lis_map,
            lis_vector=lis_vec,
            significant_mask=significant_mask,
            significant_vector=significant_vec,
            posterior_prob=posterior_prob,
            fdr_results=fdr_summary,
            model_parameters=model_params_summary,
            spatial_mask=s_mask,
            affine=vol_affine,
            method_mode=self.params.method_mode.value,
        )
