"""Voxel-level statistical hypothesis testing engine for neuroimaging data.

Computes voxel-wise test statistics, p-values, effect sizes, and converted z-scores
under specified experimental designs (One-Sample t, Two-Sample t, Paired t, and GLM).
Restricts analysis to configured spatial ROIs and outputs spatially aligned NIfTI maps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.special import ndtri

from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.core.logging import get_logger
from src.data.loaders import save_nifti
from src.data.schemas import NeuroimagingVolume
from src.statistical_analysis.design import (
    AlternativeHypothesis,
    ExperimentalDesign,
    TestType,
)
from src.statistical_analysis.validation import validate_statistical_inputs

logger = get_logger(__name__)


@dataclass
class StatisticalMaps:
    """Container holding voxel-level statistical fields and spatial coordinate metadata.

    Attributes:
        test_statistic: 3D array of test statistics (e.g. t, F, or z-values).
        p_values: 3D array of p-values in [0, 1].
        z_scores: 3D array of standard normal z-scores suitable for fcHMRF-LIS.
        effect_sizes: 3D array of effect sizes (e.g. Cohen's d or contrast beta).
        neg_log10_p: 3D array of -log10(p-value) for significance visualization.
        dof: Degrees of freedom (scalar or 3D array).
        mask: 3D boolean mask of tested in-ROI voxels.
        affine: 4x4 coordinate affine matrix.
        test_type: String description of statistical procedure used.
        alternative: String directionality ('greater', 'two_sided', 'less').
    """

    test_statistic: np.ndarray
    p_values: np.ndarray
    z_scores: np.ndarray
    effect_sizes: np.ndarray
    neg_log10_p: np.ndarray
    dof: Union[float, np.ndarray]
    mask: np.ndarray
    affine: np.ndarray
    test_type: str = "one_sample_t"
    alternative: str = "greater"

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return self.test_statistic.shape[:3]

    @property
    def num_active_voxels(self) -> int:
        return int(np.sum(self.mask))

    @property
    def analyzed_voxels(self) -> int:
        return self.num_active_voxels

    @property
    def spatial_mask(self) -> np.ndarray:
        return self.mask

    def save_nifti_maps(
        self,
        output_dir: Union[str, Path],
        prefix: str = "stat",
    ) -> Dict[str, Path]:
        """Save statistical results as spatially aligned NIfTI volumes preserving affine matrix.

        Args:
            output_dir: Target directory.
            prefix: Filename prefix.

        Returns:
            Dict[str, Path]: Mapping of map names to saved file paths.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Path] = {}

        maps_to_save = {
            "test_stat": (self.test_statistic.astype(np.float32), f"{prefix}_test_statistic.nii.gz"),
            "p_values": (self.p_values.astype(np.float32), f"{prefix}_pvalues.nii.gz"),
            "z_scores": (self.z_scores.astype(np.float32), f"{prefix}_zscores.nii.gz"),
            "effect_sizes": (self.effect_sizes.astype(np.float32), f"{prefix}_effectsizes.nii.gz"),
            "neg_log10_p": (self.neg_log10_p.astype(np.float32), f"{prefix}_neglog10p.nii.gz"),
            "analysis_mask": (self.mask.astype(np.float32), f"{prefix}_analysis_mask.nii.gz"),
        }

        for key, (arr, fname) in maps_to_save.items():
            vol = NeuroimagingVolume(
                data=arr,
                affine=self.affine,
                voxel_dimensions=tuple(np.abs(np.diag(self.affine)[:3])),
                subject_id=prefix,
            )
            dest = out_p / fname
            save_nifti(vol, dest)
            saved[key] = dest

        logger.info(f"Saved {len(saved)} statistical NIfTI maps to {out_p}")
        return saved

    def to_fchmrf_inputs(self, restrict_to_mask: bool = True) -> Dict[str, Any]:
        """Format outputs directly for the downstream fcHMRF-LIS inference engine.

        Args:
            restrict_to_mask: If True, vectors are 1D arrays of in-mask voxels.
                             If False, returns full 3D spatial grids.

        Returns:
            Dict[str, Any]: Inputs ready for fcHMRF-LIS.
        """
        if restrict_to_mask:
            in_mask_idx = self.mask
            z_vec = self.z_scores[in_mask_idx]
            p_vec = self.p_values[in_mask_idx]
            eff_vec = self.effect_sizes[in_mask_idx]
            coords = np.argwhere(in_mask_idx)
        else:
            z_vec = self.z_scores
            p_vec = self.p_values
            eff_vec = self.effect_sizes
            coords = np.argwhere(np.ones_like(self.mask, dtype=bool))

        return {
            "z_scores": z_vec,
            "p_values": p_vec,
            "effect_sizes": eff_vec,
            "spatial_mask": self.mask,
            "voxel_coordinates": coords,
            "affine": self.affine,
            "spatial_shape": self.spatial_shape,
            "dof": self.dof,
        }


def _p_to_z(p_values: np.ndarray, test_stats: np.ndarray, alternative: str = "greater") -> np.ndarray:
    """Convert p-values to signed z-scores via inverse normal CDF with tail-safe bounds."""
    # Clip p-values to prevent inf/nan from machine precision limits
    p_safe = np.clip(p_values, 1e-16, 1.0 - 1e-16)

    if alternative == "greater":
        # Upper tail: z = Phi^-1(1 - p)
        z = ndtri(1.0 - p_safe)
    elif alternative == "less":
        # Lower tail: z = -Phi^-1(1 - p)
        z = -ndtri(1.0 - p_safe)
    else:  # two_sided
        # Signed according to test statistic
        z_mag = ndtri(1.0 - p_safe / 2.0)
        z = np.sign(test_stats) * z_mag

    # Replace any non-finite with 0.0
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def one_sample_t_test(
    data_4d: np.ndarray,
    popmean: float = 0.0,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Perform voxel-wise one-sample Student's t-test across samples (axis 3).

    Args:
        data_4d: Array of shape (X, Y, Z, N_samples).
        popmean: Null hypothesis population mean.
        alternative: 'greater', 'two_sided', or 'less'.
        mask: Boolean mask indicating valid analysis voxels.

    Returns:
        Tuple: (t_stat, p_val, z_score, cohens_d, dof)
    """
    alt = str(alternative.value if isinstance(alternative, AlternativeHypothesis) else alternative).lower()
    x_dim, y_dim, z_dim, n_samples = data_4d.shape
    dof = float(n_samples - 1)

    t_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    p_map = np.ones((x_dim, y_dim, z_dim), dtype=np.float64)
    d_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)

    active_mask = mask if mask is not None else np.ones((x_dim, y_dim, z_dim), dtype=bool)

    if not np.any(active_mask):
        z_map = np.zeros_like(t_map)
        return t_map, p_map, z_map, d_map, dof

    # Extract in-mask samples: shape (V, N)
    Y = data_4d[active_mask, :]  # shape: (V_active, N)
    sample_mean = np.mean(Y, axis=1)
    sample_std = np.std(Y, axis=1, ddof=1)

    # Prevent zero-division on constant background
    safe_std = np.where(sample_std > 1e-12, sample_std, np.nan)
    se = safe_std / np.sqrt(n_samples)

    diff = sample_mean - popmean
    t_vals = diff / se
    d_vals = diff / safe_std

    # Replace zero variance voxels with 0
    t_vals = np.nan_to_num(t_vals, nan=0.0)
    d_vals = np.nan_to_num(d_vals, nan=0.0)

    # Compute p-values based on alternative hypothesis
    if alt == "greater":
        p_vals = stats.t.sf(t_vals, df=dof)
    elif alt == "less":
        p_vals = stats.t.cdf(t_vals, df=dof)
    else:  # two_sided
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof)

    # Insert back into 3D grids
    t_map[active_mask] = t_vals
    p_map[active_mask] = np.clip(p_vals, 0.0, 1.0)
    d_map[active_mask] = d_vals

    z_map = _p_to_z(p_map, t_map, alternative=alt)
    z_map[~active_mask] = 0.0

    return t_map, p_map, z_map, d_map, dof


def two_sample_t_test(
    group1_4d: np.ndarray,
    group2_4d: np.ndarray,
    equal_var: bool = False,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """Perform voxel-wise independent two-sample t-test (Welch's or Student's) comparing Group 1 > Group 2."""
    alt = str(alternative.value if isinstance(alternative, AlternativeHypothesis) else alternative).lower()
    x_dim, y_dim, z_dim, n1 = group1_4d.shape
    _, _, _, n2 = group2_4d.shape

    t_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    p_map = np.ones((x_dim, y_dim, z_dim), dtype=np.float64)
    d_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    dof_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)

    active_mask = mask if mask is not None else np.ones((x_dim, y_dim, z_dim), dtype=bool)

    if not np.any(active_mask):
        z_map = np.zeros_like(t_map)
        return t_map, p_map, z_map, d_map, float(n1 + n2 - 2)

    Y1 = group1_4d[active_mask, :]  # (V, N1)
    Y2 = group2_4d[active_mask, :]  # (V, N2)

    m1 = np.mean(Y1, axis=1)
    m2 = np.mean(Y2, axis=1)
    v1 = np.var(Y1, axis=1, ddof=1)
    v2 = np.var(Y2, axis=1, ddof=1)

    diff = m1 - m2

    if equal_var:
        # Pooled variance
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        se = np.where(se > 1e-12, se, np.nan)
        t_vals = np.nan_to_num(diff / se, nan=0.0)
        cohens_d = np.nan_to_num(diff / np.sqrt(np.where(sp2 > 1e-12, sp2, np.nan)), nan=0.0)
        dof = float(n1 + n2 - 2)
        dof_vals = np.full_like(t_vals, dof)
    else:
        # Welch's unequal variance t-test
        se1 = v1 / n1
        se2 = v2 / n2
        se = np.sqrt(se1 + se2)
        se = np.where(se > 1e-12, se, np.nan)
        t_vals = np.nan_to_num(diff / se, nan=0.0)
        # Welch-Satterthwaite degrees of freedom
        dof_num = (se1 + se2) ** 2
        dof_den = (se1**2) / (n1 - 1) + (se2**2) / (n2 - 1)
        dof_den = np.where(dof_den > 1e-12, dof_den, np.nan)
        dof_vals = np.nan_to_num(dof_num / dof_den, nan=float(n1 + n2 - 2))
        # Cohen's d using pooled SD approximation
        sp = np.sqrt((v1 + v2) / 2.0)
        cohens_d = np.nan_to_num(diff / np.where(sp > 1e-12, sp, np.nan), nan=0.0)
        dof = dof_vals

    if alt == "greater":
        p_vals = stats.t.sf(t_vals, df=dof_vals)
    elif alt == "less":
        p_vals = stats.t.cdf(t_vals, df=dof_vals)
    else:
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof_vals)

    t_map[active_mask] = t_vals
    p_map[active_mask] = np.clip(p_vals, 0.0, 1.0)
    d_map[active_mask] = cohens_d
    if isinstance(dof, np.ndarray):
        dof_map[active_mask] = dof_vals
        dof_out = dof_map
    else:
        dof_out = dof

    z_map = _p_to_z(p_map, t_map, alternative=alt)
    z_map[~active_mask] = 0.0

    return t_map, p_map, z_map, d_map, dof_out


def glm_voxelwise(
    data_4d: np.ndarray,
    design_matrix: np.ndarray,
    contrast: np.ndarray,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Perform voxel-wise Ordinary Least Squares (OLS) GLM estimation and contrast t-test.

    Model: Y = X * Beta + Epsilon
    Contrast hypothesis: c^T * Beta = 0 vs c^T * Beta > 0 (or != 0)
    """
    alt = str(alternative.value if isinstance(alternative, AlternativeHypothesis) else alternative).lower()
    x_dim, y_dim, z_dim, n_samples = data_4d.shape
    X = np.asarray(design_matrix, dtype=np.float64)
    c = np.asarray(contrast, dtype=np.float64).ravel()

    num_regressors = X.shape[1]
    dof = float(n_samples - num_regressors)

    t_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    p_map = np.ones((x_dim, y_dim, z_dim), dtype=np.float64)
    eff_map = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)

    active_mask = mask if mask is not None else np.ones((x_dim, y_dim, z_dim), dtype=bool)

    if not np.any(active_mask):
        z_map = np.zeros_like(t_map)
        return t_map, p_map, z_map, eff_map, dof

    # Invert (X^T * X) once for all voxels
    XtX = X.T @ X
    inv_XtX = np.linalg.pinv(XtX)
    pinv_X = inv_XtX @ X.T  # Shape: (P, N)

    # In-mask voxel data: Y is (N, V)
    Y = data_4d[active_mask, :].T  # Shape: (N, V_active)

    # Beta estimates: (P, V_active)
    betas = pinv_X @ Y

    # Contrast estimate: gamma = c^T * Beta -> shape (V_active,)
    contrast_est = c @ betas

    # Residuals and variance
    Y_hat = X @ betas
    residuals = Y - Y_hat
    rss = np.sum(residuals**2, axis=0)  # (V_active,)
    sigma_sq = rss / dof

    # Contrast variance: Var(c^T * Beta) = sigma^2 * (c^T * (X^T X)^-1 * c)
    c_cov_c = float(c.T @ inv_XtX @ c)
    contrast_var = sigma_sq * c_cov_c
    contrast_se = np.sqrt(np.where(contrast_var > 1e-12, contrast_var, np.nan))

    t_vals = np.nan_to_num(contrast_est / contrast_se, nan=0.0)

    if alt == "greater":
        p_vals = stats.t.sf(t_vals, df=dof)
    elif alt == "less":
        p_vals = stats.t.cdf(t_vals, df=dof)
    else:
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof)

    t_map[active_mask] = t_vals
    p_map[active_mask] = np.clip(p_vals, 0.0, 1.0)
    eff_map[active_mask] = contrast_est

    z_map = _p_to_z(p_map, t_map, alternative=alt)
    z_map[~active_mask] = 0.0

    return t_map, p_map, z_map, eff_map, dof


def compute_statistics(
    data: Union[np.ndarray, List[NeuroimagingVolume], Path, str],
    design_matrix: Optional[np.ndarray] = None,
    contrast: Optional[np.ndarray] = None,
    roi_mask: Optional[Union[np.ndarray, Path, str]] = None,
    test_type: Union[TestType, str] = TestType.ONE_SAMPLE_T,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    affine: Optional[np.ndarray] = None,
    popmean: float = 0.0,
    equal_var: bool = False,
    nan_policy: str = "mask_out",
    mu_null: Optional[float] = None,
    design: Optional[ExperimentalDesign] = None,
) -> StatisticalMaps:
    """Master entry point for voxel-level neuroimaging hypothesis testing.

    Args:
        data: 3D/4D numpy array, list of NeuroimagingVolume objects, or path to 4D NIfTI.
        design_matrix: Optional (N, P) design matrix.
        contrast: Optional (P,) contrast vector.
        roi_mask: Optional 3D ROI mask (from Phase 3 or reference atlas).
        test_type: 'one_sample_t', 'two_sample_t', 'paired_t', 'glm', or 'direct_z'.
        alternative: 'greater', 'two_sided', or 'less'.
        affine: Optional 4x4 coordinate affine matrix.
        popmean: Null mean for one-sample t-test.
        equal_var: Equal variance assumption for two-sample t-test.
        nan_policy: Action on non-finite values ('mask_out', 'zero', 'raise').
        mu_null: Alias for popmean.
        design: Optional ExperimentalDesign object encapsulating design matrix, contrast, and test type.

    Returns:
        StatisticalMaps: Container with test_statistic, p_values, z_scores, effect_sizes, mask, and affine.
    """
    if design is not None:
        test_type = design.test_type
        if design.design_matrix is not None:
            design_matrix = design.design_matrix
        if design.contrast is not None:
            contrast = design.contrast
        alternative = design.alternative
        equal_var = design.equal_var
        popmean = design.popmean

    if mu_null is not None:
        popmean = mu_null
    from src.data.loaders import load_nifti

    # 1. Ingest data
    if isinstance(data, (str, Path)):
        vol = load_nifti(data, canonical_reorient=True, validate=True)
        raw_data = vol.data
        coord_affine = vol.affine if affine is None else affine
    elif isinstance(data, list):
        # List of 3D NeuroimagingVolume objects or paths
        loaded_vols = [load_nifti(item) if isinstance(item, (str, Path)) else item for item in data]
        raw_data = np.stack([v.data[..., 0] if v.is_4d else v.data for v in loaded_vols], axis=-1)
        coord_affine = loaded_vols[0].affine if affine is None else affine
    elif isinstance(data, np.ndarray):
        raw_data = data
        coord_affine = np.eye(4) if affine is None else affine
    else:
        raise DataValidationError(f"Unsupported input data type: {type(data)}")

    # 2. Ingest ROI mask
    if roi_mask is not None:
        if isinstance(roi_mask, (str, Path)):
            mask_vol = load_nifti(roi_mask, canonical_reorient=True)
            mask_arr = (mask_vol.data[..., 0] if mask_vol.is_4d else mask_vol.data) > 0.5
        else:
            mask_arr = (np.asarray(roi_mask) > 0.5)
    else:
        mask_arr = None

    # 3. Numerical validation
    cleaned_data, valid_mask = validate_statistical_inputs(
        raw_data, design_matrix=design_matrix, roi_mask=mask_arr, nan_policy=nan_policy
    )

    t_type = TestType(test_type.lower() if isinstance(test_type, str) else test_type)
    alt = AlternativeHypothesis(alternative.lower() if isinstance(alternative, str) else alternative)

    logger.info(
        f"Computing voxel-level statistics: test='{t_type.value}', alt='{alt.value}', "
        f"data_shape={cleaned_data.shape}, active_voxels={int(np.sum(valid_mask)):,}"
    )

    # 4. Dispatch to selected statistical test
    if t_type == TestType.ONE_SAMPLE_T:
        if cleaned_data.ndim == 3:
            # Single volume provided: treat as pseudo-replicate or raise informative error
            raise StatisticalInferenceError(
                "One-sample t-test requires a 4D volume with multiple samples along axis 3. "
                "If input is already a contrast z-score map, use test_type='direct_z'."
            )
        t_stat, p_val, z_score, eff_size, dof = one_sample_t_test(
            cleaned_data, popmean=popmean, alternative=alt, mask=valid_mask
        )

    elif t_type == TestType.TWO_SAMPLE_T:
        if design_matrix is None:
            raise StatisticalInferenceError(
                "Two-sample t-test requires an experimental design matrix defining group indicators "
                "(e.g., column 0 = group 1, column 1 = group 2). Missing required design information; "
                "cannot assume an unsupported design."
            )
        # Identify group partitions from 2-column design matrix
        g1_idx = design_matrix[:, 0] > 0.5
        g2_idx = design_matrix[:, 1] > 0.5
        t_stat, p_val, z_score, eff_size, dof = two_sample_t_test(
            cleaned_data[..., g1_idx],
            cleaned_data[..., g2_idx],
            equal_var=equal_var,
            alternative=alt,
            mask=valid_mask,
        )

    elif t_type == TestType.PAIRED_T:
        if design_matrix is not None and design_matrix.shape[1] == 2:
            # Paired difference from 2 condition indices
            c1_idx = design_matrix[:, 0] > 0.5
            c2_idx = design_matrix[:, 1] > 0.5
            diff_data = cleaned_data[..., c1_idx] - cleaned_data[..., c2_idx]
        else:
            diff_data = cleaned_data
        t_stat, p_val, z_score, eff_size, dof = one_sample_t_test(
            diff_data, popmean=0.0, alternative=alt, mask=valid_mask
        )

    elif t_type == TestType.GLM:
        if design_matrix is None or contrast is None:
            raise StatisticalInferenceError(
                "GLM requires both an experimental design matrix and a contrast vector. "
                "Missing required design information; cannot assume an unsupported design."
            )
        t_stat, p_val, z_score, eff_size, dof = glm_voxelwise(
            cleaned_data, design_matrix=design_matrix, contrast=contrast, alternative=alt, mask=valid_mask
        )

    elif t_type == TestType.DIRECT_Z:
        # Data is already z-scores: pass through and compute p-values
        z_score = cleaned_data[..., 0] if cleaned_data.ndim == 4 else cleaned_data
        t_stat = z_score.astype(np.float32)
        eff_size = z_score.astype(np.float32)
        dof = np.inf
        p_val = stats.norm.sf(z_score) if alt == AlternativeHypothesis.GREATER else 2.0 * stats.norm.sf(np.abs(z_score))
        p_val[~valid_mask] = 1.0
        z_score[~valid_mask] = 0.0

    else:
        raise StatisticalInferenceError(f"Unsupported test_type: '{t_type.value}'")

    # Compute -log10(p) for significance plotting
    neg_log10_p = -np.log10(np.clip(p_val, 1e-16, 1.0))
    neg_log10_p[~valid_mask] = 0.0

    return StatisticalMaps(
        test_statistic=t_stat,
        p_values=p_val,
        z_scores=z_score,
        effect_sizes=eff_size,
        neg_log10_p=neg_log10_p.astype(np.float32),
        dof=dof,
        mask=valid_mask,
        affine=coord_affine,
        test_type=t_type.value,
        alternative=alt.value,
    )


# Alias for explicit voxel-level statistics naming
compute_voxel_statistics = compute_statistics
