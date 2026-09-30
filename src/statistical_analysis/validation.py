"""Validation and diagnostic routines for voxel-level neuroimaging statistical testing."""

from typing import Optional, Tuple
import numpy as np

from src.core.exceptions import DataValidationError, StatisticalInferenceError
from src.core.logging import get_logger

logger = get_logger(__name__)


def validate_statistical_inputs(
    data: np.ndarray,
    design_matrix: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    nan_policy: str = "mask_out",
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate volumetric data, design matrix, and ROI mask for numerical sanity.

    Args:
        data: Neuroimaging data array (3D or 4D).
        design_matrix: Optional 2D design matrix (N_samples, P_regressors).
        roi_mask: Optional 3D boolean/binary mask.
        nan_policy: Action when non-finite values encountered:
            - 'mask_out': Remove non-finite voxels from active analysis mask (default).
            - 'zero': Replace non-finite voxels with 0.0.
            - 'raise': Raise DataValidationError immediately.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Cleaned data array.
            - Valid boolean 3D spatial analysis mask.

    Raises:
        DataValidationError: If dimensions are incompatible, data is empty, or NaNs exist under 'raise' policy.
        StatisticalInferenceError: If design matrix is rank-deficient or has mismatched rows.
    """
    if not isinstance(data, np.ndarray):
        raise DataValidationError(f"Expected numpy.ndarray, got {type(data)}")

    if data.ndim not in (3, 4):
        raise DataValidationError(
            f"Voxel-level testing expects 3D or 4D array, got ndim={data.ndim} (shape={data.shape})"
        )

    spatial_shape = data.shape[:3]
    num_samples = data.shape[3] if data.ndim == 4 else 1

    # Initialize analysis mask
    if roi_mask is not None:
        roi_arr = np.asarray(roi_mask)
        if roi_arr.shape != spatial_shape:
            raise DataValidationError(
                f"ROI mask spatial shape {roi_arr.shape} does not match data spatial shape {spatial_shape}"
            )
        valid_mask = (roi_arr > 0.5)
        if not np.any(valid_mask):
            logger.warning("Provided ROI mask contains 0 active voxels. Statistical testing will yield empty maps.")
    else:
        valid_mask = np.ones(spatial_shape, dtype=bool)

    # Check for non-finite values
    non_finite = ~np.isfinite(data)
    non_finite_count = int(np.sum(non_finite))

    cleaned_data = data.copy()

    if non_finite_count > 0:
        msg = f"Detected {non_finite_count:,} non-finite (NaN/Inf) values in input neuroimaging volume."
        if nan_policy == "raise":
            raise DataValidationError(msg)
        elif nan_policy == "mask_out":
            logger.warning(f"{msg} Masking out affected voxels.")
            if data.ndim == 4:
                non_finite_spatial = np.any(non_finite, axis=3)
            else:
                non_finite_spatial = non_finite
            valid_mask &= ~non_finite_spatial
            cleaned_data[non_finite] = 0.0
        elif nan_policy == "zero":
            logger.warning(f"{msg} Imputing non-finite values with 0.0.")
            cleaned_data[non_finite] = 0.0
        else:
            raise ValueError(f"Unknown nan_policy '{nan_policy}'. Choose 'mask_out', 'zero', or 'raise'.")

    # Validate design matrix if provided
    if design_matrix is not None:
        X = np.asarray(design_matrix, dtype=np.float64)
        if X.ndim != 2:
            raise StatisticalInferenceError(f"Design matrix must be 2D, got shape {X.shape}")

        if X.shape[0] != num_samples:
            raise StatisticalInferenceError(
                f"Design matrix row count ({X.shape[0]}) does not match number of volume samples ({num_samples})."
            )

        # Check full column rank
        rank = np.linalg.matrix_rank(X)
        num_regressors = X.shape[1]
        if rank < num_regressors:
            raise StatisticalInferenceError(
                f"Design matrix is rank deficient: rank={rank} < regressors={num_regressors}. "
                "Check for collinear predictors."
            )

        if num_samples <= num_regressors:
            raise StatisticalInferenceError(
                f"Degrees of freedom error: number of samples ({num_samples}) must exceed "
                f"number of regressors ({num_regressors}) in the design matrix."
            )

    return cleaned_data, valid_mask
