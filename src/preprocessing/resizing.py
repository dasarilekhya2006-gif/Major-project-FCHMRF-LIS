"""Spatial dimension standardization (symmetric padding and cropping) for 3D deep-learning models."""

from typing import Optional, Tuple
import numpy as np

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger
from src.data.schemas import NeuroimagingVolume

logger = get_logger(__name__)


def standardize_spatial_dimensions(
    volume: NeuroimagingVolume,
    target_shape: Optional[Tuple[int, int, int]] = None,
    divisible_by: int = 8,
    mode: str = "pad_to_divisible",
) -> Tuple[NeuroimagingVolume, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Standardize spatial dimensions of a volume to satisfy 3D U-Net architectural constraints.

    Preserves exact physical world-space coordinates by updating the affine translation:
        origin_new = origin_old + A[:3, :3] @ (crop_before - pad_before)

    Modes:
      - 'pad_to_divisible': Pads dimensions to the nearest ceiling multiple of `divisible_by`.
      - 'center_pad_crop': Pads or crops centered on the volume to match `target_shape`.
      - 'none': Leaves dimensions unchanged.

    Args:
        volume: Input NeuroimagingVolume.
        target_shape: Optional explicit 3D spatial target shape (nx, ny, nz).
        divisible_by: Factor by which spatial dimensions must be divisible (e.g. 8 or 16).
        mode: Standardization mode ('pad_to_divisible', 'center_pad_crop', 'none').

    Returns:
        Tuple[NeuroimagingVolume, Tuple[Tuple[int, int], ...]]:
            Resized NeuroimagingVolume and offsets applied ((x_before, x_after), ...).

    Raises:
        PreprocessingError: If parameters are invalid or mode is unsupported.
    """
    if mode == "none":
        zero_offsets = ((0, 0), (0, 0), (0, 0))
        return volume, zero_offsets

    curr_spatial = volume.spatial_shape

    # Determine desired spatial target shape
    if mode == "pad_to_divisible":
        if divisible_by <= 0:
            raise PreprocessingError(f"divisible_by must be > 0, got {divisible_by}")
        target_dims = []
        for dim in curr_spatial:
            remainder = dim % divisible_by
            pad_needed = (divisible_by - remainder) if remainder != 0 else 0
            target_dims.append(dim + pad_needed)
        desired_shape = tuple(target_dims)

    elif mode == "center_pad_crop":
        if target_shape is None:
            raise PreprocessingError("target_shape must be specified when mode='center_pad_crop'")
        if len(target_shape) != 3:
            raise PreprocessingError(f"target_shape must be a 3-tuple, got {target_shape}")
        desired_shape = tuple(target_shape)

    else:
        raise PreprocessingError(
            f"Unsupported resizing mode: '{mode}'. Choose 'pad_to_divisible', 'center_pad_crop', or 'none'."
        )

    if desired_shape == curr_spatial:
        zero_offsets = ((0, 0), (0, 0), (0, 0))
        return volume, zero_offsets

    logger.info(
        f"Standardizing spatial dimensions from {curr_spatial} to {desired_shape} (mode='{mode}')"
    )

    data = volume.data
    mask = volume.mask
    is_4d = volume.is_4d
    num_timepoints = volume.num_timepoints

    # Calculate padding and cropping offsets for each axis
    pad_widths_data = []
    pad_widths_spatial = []
    crop_slices_data = []
    crop_slices_spatial = []
    voxel_shifts = []  # shift in index: crop_before - pad_before
    offsets_record = []

    for j in range(3):
        curr_len = curr_spatial[j]
        tgt_len = desired_shape[j]

        if tgt_len >= curr_len:
            # Padding needed
            total_pad = tgt_len - curr_len
            before = total_pad // 2
            after = total_pad - before
            pad_widths_spatial.append((before, after))
            crop_slices_spatial.append(slice(0, curr_len))
            voxel_shifts.append(-before)
            offsets_record.append((before, after))
        else:
            # Cropping needed
            total_crop = curr_len - tgt_len
            before = total_crop // 2
            after = total_crop - before
            pad_widths_spatial.append((0, 0))
            crop_slices_spatial.append(slice(before, curr_len - after))
            voxel_shifts.append(before)
            offsets_record.append((-before, -after))

    # Apply cropping first if any axis requires cropping
    cropped_data = data[tuple(crop_slices_spatial + ([slice(None)] if is_4d else []))]
    cropped_mask = mask[tuple(crop_slices_spatial)] if mask is not None else None

    # Apply padding next if any axis requires padding
    pad_data_spec = pad_widths_spatial + ([(0, 0)] if is_4d else [])
    new_data = np.pad(cropped_data, pad_data_spec, mode="constant", constant_values=0.0)

    new_mask: Optional[np.ndarray] = None
    if cropped_mask is not None:
        new_mask = np.pad(cropped_mask, pad_widths_spatial, mode="constant", constant_values=False)

    # Mathematically update the affine translation vector
    # x_world = A @ [i_old, j_old, k_old, 1]^T
    # [i_old, j_old, k_old]^T = [i_new, j_new, k_new]^T + voxel_shifts
    # origin_new = origin_old + A[:3, :3] @ voxel_shifts
    new_affine = volume.affine.copy()
    shift_vector = np.array(voxel_shifts, dtype=np.float64)
    new_origin = new_affine[:3, 3] + new_affine[:3, :3] @ shift_vector
    new_affine[:3, 3] = new_origin

    resized_volume = NeuroimagingVolume(
        data=new_data.astype(np.float32),
        affine=new_affine,
        voxel_dimensions=volume.voxel_dimensions,
        mask=new_mask,
        subject_id=volume.subject_id,
        metadata=volume.metadata,
    )

    offsets_tuple = (offsets_record[0], offsets_record[1], offsets_record[2])
    return resized_volume, offsets_tuple
