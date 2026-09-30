"""Data subpackage providing neuroimaging loaders, validation, schemas, and synthetic generators."""

from src.data.schemas import (
    NeuroimagingMetadata,
    NeuroimagingVolume,
    SyntheticBenchmarkSample,
    ValidationReport,
)
from src.data.loaders import (
    get_image_metadata,
    load_nifti,
    load_nifti_volume,
    load_numpy_volume,
    save_nifti,
    validate_image,
)
from src.data.sample_data import (
    create_sample_3d_statistical_nifti,
    create_sample_4d_fmri_nifti,
    initialize_sample_test_datasets,
)
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.data.dataset import NeuroimagingVolumeDataset

__all__ = [
    "NeuroimagingMetadata",
    "NeuroimagingVolume",
    "SyntheticBenchmarkSample",
    "ValidationReport",
    "load_nifti",
    "validate_image",
    "get_image_metadata",
    "save_nifti",
    "load_nifti_volume",
    "load_numpy_volume",
    "create_sample_3d_statistical_nifti",
    "create_sample_4d_fmri_nifti",
    "initialize_sample_test_datasets",
    "generate_synthetic_spatial_dataset",
    "NeuroimagingVolumeDataset",
]
