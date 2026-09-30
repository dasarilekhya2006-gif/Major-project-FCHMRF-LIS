"""Research adapter for neuroimaging preprocessing."""

from typing import Tuple
import numpy as np

from src.data.schemas import NeuroimagingVolume
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.schemas import PreprocessingConfig
from src.core.logging import get_logger

logger = get_logger(__name__)


def execute_preprocessing(
    raw_data: np.ndarray,
    affine: np.ndarray,
    normalize_intensity: bool = True,
    apply_brain_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute modular preprocessing using existing scientific research pipeline.

    Returns:
        (preprocessed_data, brain_mask)
    """
    logger.info("Executing preprocessing research module...")
    if raw_data.ndim == 4:
        # Take temporal mean volume for 4D fMRI
        raw_data = np.mean(raw_data, axis=-1).astype(np.float32)

    # Extract voxel spacing from the affine diagonal
    zooms = tuple(float(np.abs(affine[i, i])) for i in range(3))

    volume = NeuroimagingVolume(
        data=raw_data.astype(np.float32),
        affine=affine,
        voxel_dimensions=zooms,
    )

    cfg = PreprocessingConfig(
        normalization_method="zscore" if normalize_intensity else "none",
        compute_mask_if_missing=apply_brain_mask,
    )
    pipeline = PreprocessingPipeline(config=cfg)
    result = pipeline.run(volume)

    brain_mask = result.mask
    if brain_mask is None:
        brain_mask = np.ones(result.data.shape, dtype=bool)

    return result.data.astype(np.float32), brain_mask.astype(bool)
