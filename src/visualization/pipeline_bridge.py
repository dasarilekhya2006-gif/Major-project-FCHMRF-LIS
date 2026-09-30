"""Pipeline execution and data orchestration bridge for interactive neuroimaging visualization."""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import nibabel as nib
import numpy as np

from src.core.logging import get_logger
from src.data.loaders import load_nifti, validate_image
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.schemas import PreprocessingConfig
from src.fchmrf_lis.fchmrf_model import FCHMRFModel
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.evaluation.baselines import BenjaminiHochbergCorrection
from src.evaluation.synthetic_benchmarks import (
    SyntheticBenchmarkVolume,
    generate_synthetic_benchmark_volume,
)

logger = get_logger(__name__)


@dataclass
class VisualizationPipelineSession:
    """Encapsulates all volumetric data layers, coordinates, and metadata for interactive visualization.

    Preserves raw scientific values without cosmetic distortion.
    """

    mri_volume: np.ndarray
    affine: np.ndarray
    voxel_spacing: Tuple[float, float, float]
    preprocessed_volume: Optional[np.ndarray] = None
    brain_mask: Optional[np.ndarray] = None
    roi_mask: Optional[np.ndarray] = None
    statistical_map: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    lis_map: Optional[np.ndarray] = None
    proposed_significant_mask: Optional[np.ndarray] = None
    baseline_significant_mask: Optional[np.ndarray] = None
    ground_truth_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.mri_volume.shape


def load_research_benchmark_session(
    shape: Tuple[int, int, int] = (24, 24, 12),
    snr: float = 3.2,
    nominal_alpha: float = 0.05,
    seed: int = 42,
) -> VisualizationPipelineSession:
    """Generate a research benchmark session with known ground truth activations."""
    logger.info("Initializing research benchmark dataset for interactive visualization...")
    synth_vol = generate_synthetic_benchmark_volume(
        shape=shape,
        snr=snr,
        spatial_smooth_sigma=1.2,
        seed=seed,
    )

    # 1. Base MRI anatomical volume (simulated structural T1w scan with brain shell)
    rng = np.random.RandomState(seed)
    base_mri = np.zeros(shape, dtype=np.float32)
    # Anatomical contrast inside brain mask
    base_mri[synth_vol.spatial_mask] = rng.normal(loc=100.0, scale=12.0, size=int(np.sum(synth_vol.spatial_mask)))
    # Add activation contrast
    base_mri[synth_vol.ground_truth] += (snr * 8.0)

    # 2. Run Baseline BH-FDR
    bh = BenjaminiHochbergCorrection(alpha=nominal_alpha)
    bh_res = bh.run(
        z_scores=synth_vol.z_scores,
        p_values=synth_vol.p_values,
        spatial_mask=synth_vol.spatial_mask,
    )

    # 3. Run Proposed Adaptive ROI-Assisted fcHMRF-LIS
    params = FCHMRFParameters(
        method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
        beta=0.8,
        sigma_spatial=2.5,
        sigma_intensity=1.5,
        lambda_dl=0.5,
        alpha=nominal_alpha,
        max_iterations=25,
    )
    model = FCHMRFModel(params=params)
    fchmrf_res = model.fit_predict(
        z_scores=synth_vol.z_scores,
        dl_prior=synth_vol.dl_prior,
        spatial_mask=synth_vol.spatial_mask,
    )

    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    voxel_spacing = (2.0, 2.0, 2.5)

    meta = {
        "dataset_type": "Synthetic Research Benchmark with Known Ground Truth",
        "dimensions": shape,
        "voxel_spacing_mm": voxel_spacing,
        "snr": snr,
        "nominal_alpha": nominal_alpha,
        "proposed_discoveries": int(np.sum(fchmrf_res.significant_mask)),
        "baseline_discoveries": int(np.sum(bh_res.significant_mask)),
        "ground_truth_voxels": int(np.sum(synth_vol.ground_truth)),
        "lis_threshold": fchmrf_res.fdr_results.get("lis_threshold", 0.0),
    }

    return VisualizationPipelineSession(
        mri_volume=base_mri,
        affine=affine,
        voxel_spacing=voxel_spacing,
        preprocessed_volume=synth_vol.z_scores,
        brain_mask=synth_vol.spatial_mask,
        roi_mask=(synth_vol.dl_prior > 0.4),
        statistical_map=synth_vol.z_scores,
        p_values=synth_vol.p_values,
        lis_map=fchmrf_res.lis_map,
        proposed_significant_mask=fchmrf_res.significant_mask,
        baseline_significant_mask=bh_res.significant_mask,
        ground_truth_mask=synth_vol.ground_truth,
        metadata=meta,
    )


def process_uploaded_nifti_bytes(
    file_bytes: bytes,
    filename: str,
    nominal_alpha: float = 0.05,
) -> VisualizationPipelineSession:
    """Process an uploaded .nii or .nii.gz volume through the full research pipeline."""
    logger.info(f"Ingesting uploaded NIfTI file '{filename}' ({len(file_bytes)} bytes)...")
    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        # 1. Load NIfTI and validate
        loaded_img = load_nifti(tmp_path)
        valid_res = validate_image(tmp_path)
        if not valid_res.is_valid:
            logger.warning(f"Validation warnings on uploaded volume: {valid_res.error_message}")

        raw_data = loaded_img.data
        if raw_data.ndim == 4:
            # If 4D fMRI, take temporal mean volume for anatomical representation
            raw_data = np.mean(raw_data, axis=-1)

        affine = loaded_img.affine
        voxel_spacing = tuple(float(x) for x in loaded_img.voxel_spacing[:3])
        shape = raw_data.shape

        # 2. Preprocessing
        prep_cfg = PreprocessingConfig(normalization_method="zscore", compute_mask_if_missing=True)
        prep_pipe = PreprocessingPipeline(config=prep_cfg)
        prep_out = prep_pipe.process_array(raw_data, affine=affine)
        preprocessed_data = prep_out.processed_data
        brain_mask = prep_out.brain_mask

        # 3. Compute Voxel-Level Statistics (z-scores from standardized contrast)
        z_scores = preprocessed_data.astype(np.float32)
        if brain_mask is not None:
            z_scores[~brain_mask] = 0.0

        from scipy import stats
        p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))
        p_vals = np.clip(p_vals, 1e-15, 1.0).astype(np.float32)

        # 4. Generate 3D U-Net ROI Prior (using intensity thresholding as fallback prior)
        upper_q = float(np.percentile(z_scores[brain_mask], 85)) if np.any(brain_mask) else 1.5
        roi_mask = (z_scores > upper_q) & (brain_mask if brain_mask is not None else np.ones(shape, bool))
        from scipy import ndimage
        dl_prior = ndimage.gaussian_filter(roi_mask.astype(np.float32), sigma=1.0)
        dl_prior = np.clip(dl_prior, 0.01, 0.99)

        # 5. Baseline BH-FDR
        bh = BenjaminiHochbergCorrection(alpha=nominal_alpha)
        bh_res = bh.run(z_scores=z_scores, p_values=p_vals, spatial_mask=brain_mask)

        # 6. Proposed Adaptive fcHMRF-LIS
        params = FCHMRFParameters(
            method_mode=MethodMode.ADAPTIVE_DL_FCHMRF,
            beta=0.8,
            sigma_spatial=2.5,
            sigma_intensity=1.5,
            lambda_dl=0.5,
            alpha=nominal_alpha,
            max_iterations=25,
        )
        model = FCHMRFModel(params=params)
        fchmrf_res = model.fit_predict(
            z_scores=z_scores,
            dl_prior=dl_prior,
            spatial_mask=brain_mask,
        )

        meta = {
            "filename": filename,
            "dimensions": shape,
            "voxel_spacing_mm": voxel_spacing,
            "nominal_alpha": nominal_alpha,
            "proposed_discoveries": int(np.sum(fchmrf_res.significant_mask)),
            "baseline_discoveries": int(np.sum(bh_res.significant_mask)),
            "lis_threshold": fchmrf_res.fdr_results.get("lis_threshold", 0.0),
        }

        return VisualizationPipelineSession(
            mri_volume=raw_data.astype(np.float32),
            affine=affine,
            voxel_spacing=voxel_spacing,
            preprocessed_volume=preprocessed_data.astype(np.float32),
            brain_mask=brain_mask,
            roi_mask=roi_mask,
            statistical_map=z_scores,
            p_values=p_vals,
            lis_map=fchmrf_res.lis_map,
            proposed_significant_mask=fchmrf_res.significant_mask,
            baseline_significant_mask=bh_res.significant_mask,
            ground_truth_mask=None,
            metadata=meta,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
