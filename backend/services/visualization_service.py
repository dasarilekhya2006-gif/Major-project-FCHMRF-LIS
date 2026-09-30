"""Visualization service providing 2D slice data and 3D surface meshes to the Phase 10 frontend."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from fastapi import HTTPException, status

from backend.schemas.visualization import (
    ValueRange,
    VisualizationMetadataResponse,
    SliceDataResponse,
    Mesh3DData,
    Mesh3DResponse,
)
from backend.services.storage_service import storage_service
from src.core.logging import get_logger

logger = get_logger(__name__)


def _extract_slice(volume: np.ndarray, orientation: str, slice_idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume — no external dependencies."""
    orientation = orientation.lower()
    if orientation == "axial":
        return volume[:, :, slice_idx]
    elif orientation == "coronal":
        return volume[:, slice_idx, :]
    elif orientation == "sagittal":
        return volume[slice_idx, :, :]
    return volume[:, :, slice_idx]


def _extract_surface_mesh(
    volume: np.ndarray,
    level: float,
    spacing: Tuple[float, float, float],
    step_size: int,
) -> Optional[Tuple]:
    """Extract Marching Cubes surface mesh. Lazy-imports skimage so startup never fails."""
    try:
        from skimage import measure
    except ImportError:
        logger.warning("scikit-image not installed — 3D mesh endpoint unavailable.")
        return None

    try:
        verts, faces, _, _ = measure.marching_cubes(
            volume, level=level, spacing=spacing, step_size=step_size
        )
        if len(verts) == 0 or len(faces) == 0:
            return None
        x = verts[:, 0].astype(np.float32)
        y = verts[:, 1].astype(np.float32)
        z = verts[:, 2].astype(np.float32)
        i = faces[:, 0].astype(np.int32)
        j = faces[:, 1].astype(np.int32)
        k = faces[:, 2].astype(np.int32)
        return x, y, z, i, j, k
    except Exception as e:
        logger.warning(f"Marching cubes failed: {e}")
        return None


class VisualizationService:
    """Provides raw data arrays and geometric meshes required by the Phase 10 interactive frontend."""

    def get_metadata(self, experiment_id: str) -> VisualizationMetadataResponse:
        """Inspect experiment and compute value bounds and spatial dimensions."""
        storage_service.get_experiment_dir(experiment_id)
        raw_mri, affine = storage_service.load_stage_volume(experiment_id, "input", "raw_volume.nii.gz")
        meta = storage_service.get_metadata(experiment_id)

        shape = list(raw_mri.shape)
        voxel_spacing = meta.get("voxel_spacing", [1.0, 1.0, 1.0])

        layers: Dict[str, Dict[str, Any]] = {
            "original": {"available": True},
            "preprocessed": {
                "available": storage_service.has_stage_artifact(experiment_id, "preprocess", "preprocessed.nii.gz")
            },
            "roi_mask": {
                "available": storage_service.has_stage_artifact(experiment_id, "segmentation", "roi_mask.nii.gz")
            },
            "statistical_map": {
                "available": storage_service.has_stage_artifact(experiment_id, "statistics", "z_scores.nii.gz")
            },
            "lis_map": {
                "available": storage_service.has_stage_artifact(experiment_id, "fchmrf", "lis_map.nii.gz")
            },
            "significant_mask": {
                "available": storage_service.has_stage_artifact(experiment_id, "fdr", "significant_mask.nii.gz")
            },
            "ground_truth": {
                "available": storage_service.has_stage_artifact(experiment_id, "input", "ground_truth.nii.gz")
            },
            "baseline_significant_mask": {
                "available": storage_service.has_stage_artifact(experiment_id, "fdr", "baseline_significant.nii.gz")
            },
        }

        value_ranges: Dict[str, ValueRange] = {
            "original": ValueRange(min=float(np.nanmin(raw_mri)), max=float(np.nanmax(raw_mri))),
        }

        if layers["preprocessed"]["available"]:
            prep, _ = storage_service.load_stage_volume(experiment_id, "preprocess", "preprocessed.nii.gz")
            value_ranges["preprocessed"] = ValueRange(min=float(np.nanmin(prep)), max=float(np.nanmax(prep)))

        if layers["statistical_map"]["available"]:
            z_scores, _ = storage_service.load_stage_volume(experiment_id, "statistics", "z_scores.nii.gz")
            value_ranges["statistical_map"] = ValueRange(
                min=float(np.nanmin(z_scores)), max=float(np.nanmax(z_scores))
            )

        if layers["lis_map"]["available"]:
            lis, _ = storage_service.load_stage_volume(experiment_id, "fchmrf", "lis_map.nii.gz")
            value_ranges["lis_map"] = ValueRange(min=float(np.nanmin(lis)), max=float(np.nanmax(lis)))

        return VisualizationMetadataResponse(
            experiment_id=experiment_id,
            layers=layers,
            visualization={"supports_2d": True, "supports_3d": True},
            dimensions=shape,
            voxel_spacing=voxel_spacing,
            affine=affine.tolist(),
            value_ranges=value_ranges,
        )

    def get_slice_data(
        self,
        experiment_id: str,
        orientation: str = "axial",
        slice_idx: Optional[int] = None,
    ) -> SliceDataResponse:
        """Extract multi-layer 2D scientific data slice for interactive rendering."""
        storage_service.get_experiment_dir(experiment_id)
        raw_mri, _ = storage_service.load_stage_volume(experiment_id, "input", "raw_volume.nii.gz")

        dim_idx = {"axial": 2, "coronal": 1, "sagittal": 0}.get(orientation.lower(), 2)
        max_slice = raw_mri.shape[dim_idx] - 1

        if slice_idx is None:
            slice_idx = max_slice // 2
        slice_idx = max(0, min(max_slice, slice_idx))

        mri_2d = _extract_slice(raw_mri, orientation, slice_idx)
        height, width = mri_2d.shape

        stat_2d = None
        if storage_service.has_stage_artifact(experiment_id, "statistics", "z_scores.nii.gz"):
            z_scores, _ = storage_service.load_stage_volume(experiment_id, "statistics", "z_scores.nii.gz")
            stat_2d = _extract_slice(z_scores, orientation, slice_idx).astype(float).tolist()

        lis_2d = None
        if storage_service.has_stage_artifact(experiment_id, "fchmrf", "lis_map.nii.gz"):
            lis, _ = storage_service.load_stage_volume(experiment_id, "fchmrf", "lis_map.nii.gz")
            lis_2d = _extract_slice(lis, orientation, slice_idx).astype(float).tolist()

        roi_2d = None
        if storage_service.has_stage_artifact(experiment_id, "segmentation", "roi_mask.nii.gz"):
            roi, _ = storage_service.load_stage_volume(experiment_id, "segmentation", "roi_mask.nii.gz")
            roi_2d = (_extract_slice(roi, orientation, slice_idx) > 0).astype(int).tolist()

        sig_2d = None
        if storage_service.has_stage_artifact(experiment_id, "fdr", "significant_mask.nii.gz"):
            sig, _ = storage_service.load_stage_volume(experiment_id, "fdr", "significant_mask.nii.gz")
            sig_2d = (_extract_slice(sig, orientation, slice_idx) > 0).astype(int).tolist()

        gt_2d = None
        if storage_service.has_stage_artifact(experiment_id, "input", "ground_truth.nii.gz"):
            gt, _ = storage_service.load_stage_volume(experiment_id, "input", "ground_truth.nii.gz")
            gt_2d = (_extract_slice(gt, orientation, slice_idx) > 0).astype(int).tolist()

        base_sig_2d = None
        if storage_service.has_stage_artifact(experiment_id, "fdr", "baseline_significant.nii.gz"):
            base_sig, _ = storage_service.load_stage_volume(experiment_id, "fdr", "baseline_significant.nii.gz")
            base_sig_2d = (_extract_slice(base_sig, orientation, slice_idx) > 0).astype(int).tolist()

        return SliceDataResponse(
            experiment_id=experiment_id,
            orientation=orientation,
            slice_idx=slice_idx,
            max_slice=max_slice,
            width=width,
            height=height,
            mri_slice=mri_2d.astype(float).tolist(),
            statistical_slice=stat_2d,
            lis_slice=lis_2d,
            roi_slice=roi_2d,
            significant_slice=sig_2d,
            ground_truth_slice=gt_2d,
            baseline_significant_slice=base_sig_2d,
        )

    def get_3d_mesh_data(self, experiment_id: str, step_size: int = 1) -> Mesh3DResponse:
        """Extract Marching Cubes 3D surface meshes for brain, ROI, and abnormalities."""
        storage_service.get_experiment_dir(experiment_id)
        raw_mri, affine = storage_service.load_stage_volume(experiment_id, "input", "raw_volume.nii.gz")

        spacing = (
            float(np.linalg.norm(affine[:3, 0])),
            float(np.linalg.norm(affine[:3, 1])),
            float(np.linalg.norm(affine[:3, 2])),
        )

        def _to_mesh_data(mesh_tuple) -> Optional[Mesh3DData]:
            if mesh_tuple is None:
                return None
            x, y, z, i, j, k = mesh_tuple
            if len(x) == 0 or len(i) == 0:
                return None
            return Mesh3DData(
                num_vertices=len(x),
                num_faces=len(i),
                x=x.tolist(),
                y=y.tolist(),
                z=z.tolist(),
                i=i.tolist(),
                j=j.tolist(),
                k=k.tolist(),
            )

        if storage_service.has_stage_artifact(experiment_id, "preprocess", "brain_mask.nii.gz"):
            b_mask, _ = storage_service.load_stage_volume(experiment_id, "preprocess", "brain_mask.nii.gz")
            brain_vol = (b_mask > 0).astype(float)
            thresh = 0.5
        else:
            brain_vol = raw_mri
            thresh = float(np.percentile(raw_mri, 25))

        mesh_b = _extract_surface_mesh(brain_vol, level=thresh, spacing=spacing, step_size=step_size)

        mesh_roi = None
        if storage_service.has_stage_artifact(experiment_id, "segmentation", "roi_mask.nii.gz"):
            roi, _ = storage_service.load_stage_volume(experiment_id, "segmentation", "roi_mask.nii.gz")
            if np.any(roi):
                mesh_roi = _extract_surface_mesh(roi.astype(float), level=0.5, spacing=spacing, step_size=step_size)

        mesh_sig = None
        if storage_service.has_stage_artifact(experiment_id, "fdr", "significant_mask.nii.gz"):
            sig, _ = storage_service.load_stage_volume(experiment_id, "fdr", "significant_mask.nii.gz")
            if np.any(sig):
                mesh_sig = _extract_surface_mesh(sig.astype(float), level=0.5, spacing=spacing, step_size=step_size)

        mesh_gt = None
        if storage_service.has_stage_artifact(experiment_id, "input", "ground_truth.nii.gz"):
            gt, _ = storage_service.load_stage_volume(experiment_id, "input", "ground_truth.nii.gz")
            if np.any(gt):
                mesh_gt = _extract_surface_mesh(gt.astype(float), level=0.5, spacing=spacing, step_size=step_size)

        return Mesh3DResponse(
            experiment_id=experiment_id,
            brain_shell=_to_mesh_data(mesh_b),
            roi_mesh=_to_mesh_data(mesh_roi),
            significant_mesh=_to_mesh_data(mesh_sig),
            ground_truth_mesh=_to_mesh_data(mesh_gt),
        )


visualization_service = VisualizationService()
