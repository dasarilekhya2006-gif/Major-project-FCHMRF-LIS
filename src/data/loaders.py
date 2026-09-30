"""Robust neuroimaging data loading, validation, and metadata extraction module.

Supports 3D and 4D NIfTI files (.nii, .nii.gz) via NiBabel with canonical reorientation,
in-memory data representation normalization, empty volume detection, and comprehensive
structural validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.exceptions import (
    CorruptNiftiError,
    DataValidationError,
    DimensionMismatchError,
    EmptyVolumeError,
    InvalidAffineError,
)
from src.core.logging import get_logger
from src.data.schemas import NeuroimagingMetadata, NeuroimagingVolume, ValidationReport

logger = get_logger(__name__)


def get_image_metadata(file_or_image: Union[str, Path, Any]) -> NeuroimagingMetadata:
    """Extract complete structural, spatial, and header metadata from a NIfTI file or loaded image.

    Args:
        file_or_image: Path to a NIfTI file or a preloaded nibabel image object.

    Returns:
        NeuroimagingMetadata: Dataclass containing dimensions, zooms, affine, orientation, and summary stats.

    Raises:
        DataValidationError: If file does not exist, cannot be read, or metadata cannot be parsed.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise DataValidationError("nibabel is required to extract NIfTI metadata.") from e

    img = _resolve_nibabel_image(file_or_image)

    header = img.header
    affine = np.asarray(img.affine, dtype=np.float64)
    dimensions = tuple(int(dim) for dim in img.shape)
    voxel_dims = tuple(float(z) for z in header.get_zooms()[: len(dimensions)])
    ndim = len(dimensions)

    # Orientation analysis
    try:
        axcodes = nib.orientations.aff2axcodes(affine)
        orientation_str = "".join(axcodes) + "+"
    except Exception:
        orientation_str = "UNKNOWN"

    # Intensity diagnostics (using raw or cached data without mutative re-scaling)
    try:
        data = np.asarray(img.dataobj)
        finite_mask = np.isfinite(data)
        has_nan = bool(np.isnan(data).any())
        has_inf = bool(np.isinf(data).any())

        if np.any(finite_mask):
            finite_vals = data[finite_mask]
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
            mean_val = float(np.mean(finite_vals))
            std_val = float(np.std(finite_vals))
            is_empty = bool(min_val == 0.0 and max_val == 0.0)
        else:
            min_val, max_val, mean_val, std_val = 0.0, 0.0, 0.0, 0.0
            is_empty = True
    except Exception as e:
        logger.warning(f"Could not compute intensity statistics from dataobj: {e}")
        min_val, max_val, mean_val, std_val = 0.0, 0.0, 0.0, 0.0
        has_nan, has_inf, is_empty = False, False, False

    # Extract key raw header fields safely
    raw_header: Dict[str, Any] = {}
    try:
        raw_header["descrip"] = str(header.get("descrip", b"").tobytes().decode("utf-8", "ignore")).strip("\x00")
    except Exception:
        raw_header["descrip"] = ""

    try:
        raw_header["intent_code"] = int(header.get("intent_code", 0))
    except Exception:
        raw_header["intent_code"] = 0

    try:
        raw_header["xyzt_units"] = int(header.get("xyzt_units", 0))
    except Exception:
        raw_header["xyzt_units"] = 0

    qform_code = int(header.get("qform_code", 0)) if "qform_code" in header else 0
    sform_code = int(header.get("sform_code", 0)) if "sform_code" in header else 0

    return NeuroimagingMetadata(
        dimensions=dimensions,
        voxel_dimensions=voxel_dims,
        affine=affine,
        data_dtype=str(header.get_data_dtype()),
        ndim=ndim,
        num_voxels=int(np.prod(dimensions)),
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
        has_nan=has_nan,
        has_inf=has_inf,
        is_empty=is_empty,
        qform_code=qform_code,
        sform_code=sform_code,
        orientation=orientation_str,
        description=raw_header["descrip"],
        raw_header=raw_header,
    )


def validate_image(
    file_or_image: Union[str, Path, Any],
    check_empty: bool = True,
    max_nan_ratio: float = 0.05,
    check_affine: bool = True,
) -> ValidationReport:
    """Validate a neuroimaging volume against structural, spatial, and numerical integrity rules.

    Checks:
      1. File existence and valid readable NIfTI header.
      2. Valid spatial dimensionality (3D or 4D).
      3. Finite, non-zero voxel dimensions.
      4. Invertible, non-singular 4x4 coordinate affine matrix.
      5. Absence of empty / all-zero volume content (if check_empty=True).
      6. NaN / Inf ratios within acceptable research bounds.

    Args:
        file_or_image: Path to NIfTI file or preloaded nibabel image.
        check_empty: If True, reject volumes containing only zero signal.
        max_nan_ratio: Maximum tolerable fraction of non-finite values before raising an error.
        check_affine: If True, verify non-singularity of the coordinate affine matrix.

    Returns:
        ValidationReport: Report containing .is_valid, .errors, and .warnings.
    """
    path_str = str(file_or_image) if isinstance(file_or_image, (str, Path)) else None
    report = ValidationReport(is_valid=True, file_path=path_str)

    # 1. Existence and basic format verification
    if isinstance(file_or_image, (str, Path)):
        p = Path(file_or_image)
        if not p.exists():
            report.add_error(f"File does not exist: {p}")
            return report
        if not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            report.add_warning(
                f"File '{p.name}' does not possess a standard NIfTI extension (.nii or .nii.gz)."
            )

    # 2. NiBabel load & metadata extraction
    try:
        metadata = get_image_metadata(file_or_image)
        report.metadata = metadata
    except Exception as e:
        report.add_error(f"Failed to read or parse NIfTI image: {e}")
        return report

    # 3. Dimensionality check (3D or 4D)
    if metadata.ndim not in (3, 4):
        report.add_error(
            f"Unsupported dimensionality: expected 3D or 4D volume, but found ndim={metadata.ndim} (shape {metadata.dimensions})"
        )

    # 4. Voxel spacing check
    for idx, spacing in enumerate(metadata.voxel_dimensions[:3]):
        if spacing <= 0.0 or not np.isfinite(spacing):
            report.add_error(
                f"Invalid voxel spacing along spatial axis {idx}: {spacing} mm (must be finite and > 0)."
            )

    # 5. Affine matrix check
    if check_affine:
        aff = metadata.affine
        if aff.shape != (4, 4):
            report.add_error(f"Affine matrix must be 4x4, found shape {aff.shape}")
        elif not np.all(np.isfinite(aff)):
            report.add_error("Affine matrix contains non-finite (NaN or Inf) values.")
        else:
            # Check determinant of 3x3 spatial rotation/scaling block
            det = np.abs(np.linalg.det(aff[:3, :3]))
            if det < 1e-6:
                report.add_error(
                    f"Affine matrix is singular or degenerate: det(affine[:3,:3]) = {det:.2e} < 1e-6"
                )

    # 6. Empty volume check
    if check_empty and metadata.is_empty:
        report.add_error("Volume is empty: contains all zeros or uninformative constant zero signal.")

    # 7. Non-finite values ratio check
    if metadata.has_nan or metadata.has_inf:
        try:
            img = _resolve_nibabel_image(file_or_image)
            data = np.asarray(img.dataobj)
            non_finite_count = int(np.sum(~np.isfinite(data)))
            nan_ratio = non_finite_count / float(data.size)
            if nan_ratio > max_nan_ratio:
                report.add_error(
                    f"Excessive non-finite (NaN/Inf) values: {nan_ratio * 100:.2f}% of voxels exceed threshold {max_nan_ratio * 100:.1f}%"
                )
            else:
                report.add_warning(
                    f"Volume contains {non_finite_count} non-finite voxels ({nan_ratio * 100:.2f}% of volume)."
                )
        except Exception:
            pass

    return report


def load_nifti(
    file_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    canonical_reorient: bool = True,
    squeeze_singleton: bool = True,
    time_index: Optional[int] = None,
    nan_policy: str = "zero",
    validate: bool = True,
    allow_empty: bool = False,
) -> NeuroimagingVolume:
    """Load and normalize a 3D or 4D NIfTI file into a NeuroimagingVolume.

    Normalization performed (in-memory without modifying disk files):
      - Canonical reorientation to RAS+ coordinate axes if requested.
      - Casting voxel intensity to standard float32 representation.
      - Squeezing singleton 4th dimension (e.g. (X, Y, Z, 1) -> (X, Y, Z)) if requested.
      - Extracting a specific timepoint slice if time_index is specified.
      - Handling NaN/Inf values according to nan_policy ("zero", "error", or "keep").

    Args:
        file_path: Path to the input .nii or .nii.gz file.
        mask_path: Optional path to a matching binary brain mask.
        canonical_reorient: If True, reorient axes to standard RAS+ coordinates.
        squeeze_singleton: If True, collapse shape (X, Y, Z, 1) to (X, Y, Z).
        time_index: Optional integer to extract a specific 3D volume from a 4D timeseries.
        nan_policy: Treatment for NaNs: "zero" replaces with 0.0; "error" raises DataValidationError; "keep" retains NaNs.
        validate: If True, run comprehensive validation checks before constructing volume.

    Returns:
        NeuroimagingVolume: Structured container with normalized data, affine, zooms, and metadata.

    Raises:
        DataValidationError: If file does not exist, cannot be read, or fails validation.
        EmptyVolumeError: If the loaded volume is completely empty.
        CorruptNiftiError: If file is truncated or header is corrupt.
        InvalidAffineError: If the affine transformation is invalid or singular.
        DimensionMismatchError: If the optional mask does not match the volume shape.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataValidationError(f"NIfTI file does not exist: {path}")

    try:
        import nibabel as nib
    except ImportError as e:
        raise DataValidationError("nibabel is required to load NIfTI volumes.") from e

    # 1. Validation check
    if validate:
        report = validate_image(path, check_empty=not allow_empty)
        if not report.is_valid:
            err_msg = "; ".join(report.errors)
            if any("empty" in err.lower() for err in report.errors):
                raise EmptyVolumeError(f"Validation failed for {path.name}: {err_msg}")
            if any("affine" in err.lower() for err in report.errors):
                raise InvalidAffineError(f"Validation failed for {path.name}: {err_msg}")
            if any("corrupt" in err.lower() or "read or parse" in err.lower() or "decompress" in err.lower() for err in report.errors):
                raise CorruptNiftiError(f"Validation failed for {path.name}: {err_msg}")
            raise DataValidationError(f"Validation failed for {path.name}: {err_msg}")

    # 2. Load with NiBabel
    try:
        img = nib.load(str(path))
    except Exception as e:
        raise CorruptNiftiError(f"Corrupt or unreadable NIfTI file at {path}: {e}") from e

    # 3. Canonical reorientation (RAS+) if requested
    if canonical_reorient:
        try:
            img = nib.as_closest_canonical(img)
        except Exception as e:
            logger.warning(f"Canonical reorientation failed for {path}: {e}")

    # 4. Extract data array (float32 representation)
    try:
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    except Exception as e:
        raise CorruptNiftiError(f"Failed to decode voxel data from {path}: {e}") from e

    affine = np.asarray(img.affine, dtype=np.float64)
    zooms = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])

    # 5. Handle singleton 4th dimension
    if squeeze_singleton and data.ndim == 4 and data.shape[-1] == 1:
        data = data.squeeze(-1)
        zooms = zooms[:3]

    # 6. Extract specific timepoint if requested
    if time_index is not None:
        if data.ndim != 4:
            raise DataValidationError(
                f"time_index={time_index} was specified, but volume is not 4D (shape={data.shape})"
            )
        if time_index < 0 or time_index >= data.shape[3]:
            raise IndexError(
                f"time_index={time_index} out of bounds for 4D volume with {data.shape[3]} timepoints."
            )
        data = data[..., time_index]
        zooms = zooms[:3]

    # 7. NaN/Inf handling
    has_nans = np.isnan(data).any()
    has_infs = np.isinf(data).any()

    if has_nans or has_infs:
        if nan_policy == "error":
            raise DataValidationError(
                f"Volume from {path.name} contains non-finite values under nan_policy='error'."
            )
        elif nan_policy == "zero":
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info(f"Replaced non-finite values with 0.0 in {path.name} (nan_policy='zero')")
        elif nan_policy == "keep":
            pass
        else:
            raise ValueError(f"Unknown nan_policy: '{nan_policy}'. Choose 'zero', 'error', or 'keep'.")

    # 8. Check for empty volume
    if not allow_empty and np.all(data == 0.0):
        raise EmptyVolumeError(f"Loaded volume {path.name} is completely empty (all voxels are zero).")

    # 9. Optional mask loading and validation
    mask_data: Optional[np.ndarray] = None
    if mask_path is not None:
        m_path = Path(mask_path)
        if not m_path.exists():
            raise DataValidationError(f"Brain mask file does not exist: {m_path}")
        try:
            mask_img = nib.load(str(m_path))
            if canonical_reorient:
                mask_img = nib.as_closest_canonical(mask_img)
            m_arr = np.asarray(mask_img.get_fdata(), dtype=bool)

            spatial_shape = data.shape[:3]
            if m_arr.shape[:3] != spatial_shape:
                raise DimensionMismatchError(
                    f"Mask spatial dimensions {m_arr.shape[:3]} do not match volume spatial dimensions {spatial_shape}"
                )
            mask_data = m_arr
        except Exception as e:
            if isinstance(e, (DataValidationError, DimensionMismatchError)):
                raise
            raise CorruptNiftiError(f"Failed to load mask file {m_path}: {e}") from e

    # 10. Extract metadata
    metadata = get_image_metadata(img)

    logger.info(
        f"Loaded normalized NIfTI volume '{path.name}': shape={data.shape}, "
        f"zooms={zooms}, orientation={metadata.orientation}"
    )

    return NeuroimagingVolume(
        data=data,
        affine=affine,
        voxel_dimensions=zooms,
        mask=mask_data,
        subject_id=path.stem.replace(".nii", ""),
        metadata=metadata,
    )


def save_nifti(
    volume: NeuroimagingVolume,
    file_path: Union[str, Path],
    dtype: Optional[np.dtype] = None,
) -> Path:
    """Save a NeuroimagingVolume to a NIfTI-1 file (.nii or .nii.gz).

    Args:
        volume: NeuroimagingVolume instance to save.
        file_path: Destination path.
        dtype: Optional numpy data type for serialization (defaults to float32).

    Returns:
        Path: Output path written.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise DataValidationError("nibabel is required to save NIfTI volumes.") from e

    out_p = Path(file_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    arr = volume.data if dtype is None else volume.data.astype(dtype)
    img = nib.Nifti1Image(arr, affine=volume.affine)
    img.header.set_zooms(volume.voxel_dimensions[: arr.ndim])

    nib.save(img, str(out_p))
    logger.info(f"Saved NIfTI volume to {out_p} with shape {arr.shape}")
    return out_p


# Backward compatibility aliases
def load_nifti_volume(
    file_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
) -> NeuroimagingVolume:
    """Backward-compatible alias for load_nifti."""
    return load_nifti(file_path=file_path, mask_path=mask_path)


def load_numpy_volume(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
) -> NeuroimagingVolume:
    """Wrap a raw NumPy array into a validated NeuroimagingVolume."""
    if not isinstance(data, np.ndarray):
        raise DataValidationError("Data must be a numpy ndarray.")
    if data.ndim not in (3, 4):
        raise DataValidationError(f"Expected 3D or 4D numpy array, got shape {data.shape}")

    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    return NeuroimagingVolume(
        data=data.astype(np.float32),
        affine=affine,
        mask=mask.astype(bool) if mask is not None else None,
    )


def _resolve_nibabel_image(file_or_image: Union[str, Path, Any]) -> Any:
    """Helper to return a nibabel Nifti1Image whether given a path or an existing image."""
    try:
        import nibabel as nib
    except ImportError as e:
        raise DataValidationError("nibabel is required to load NIfTI files.") from e

    if isinstance(file_or_image, (str, Path)):
        p = Path(file_or_image)
        if not p.exists():
            raise DataValidationError(f"File does not exist: {p}")
        try:
            return nib.load(str(p))
        except Exception as e:
            raise CorruptNiftiError(f"Failed to read NIfTI file at {p}: {e}") from e
    elif hasattr(file_or_image, "header") and hasattr(file_or_image, "affine"):
        return file_or_image
    else:
        raise DataValidationError(
            f"Expected file path or nibabel image object, received: {type(file_or_image)}"
        )


# Functional aliases for standardized API
validate_nifti = validate_image
get_nifti_metadata = get_image_metadata
