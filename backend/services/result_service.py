"""Results aggregation and experiment metadata service."""

from backend.schemas.results import (
    LayerAvailability,
    ResultsResponse,
    ExperimentMetadataResponse,
)
from backend.services.storage_service import storage_service
from src.core.logging import get_logger

logger = get_logger(__name__)


class ResultService:
    """Provides safe result references and experiment metadata without exposing raw server paths."""

    def get_results(self, experiment_id: str) -> ResultsResponse:
        """Inspect on-disk artifacts and build a comprehensive layer availability report."""
        storage_service.get_experiment_dir(experiment_id)
        meta = storage_service.get_metadata(experiment_id)

        has_orig = storage_service.has_stage_artifact(experiment_id, "input", "raw_volume.nii.gz")
        has_prep = storage_service.has_stage_artifact(experiment_id, "preprocess", "preprocessed.nii.gz")
        has_roi = storage_service.has_stage_artifact(experiment_id, "segmentation", "roi_mask.nii.gz")
        has_stat = storage_service.has_stage_artifact(experiment_id, "statistics", "z_scores.nii.gz")
        has_lis = storage_service.has_stage_artifact(experiment_id, "fchmrf", "lis_map.nii.gz")
        has_fdr = storage_service.has_stage_artifact(experiment_id, "fdr", "significant_mask.nii.gz")
        has_gt = storage_service.has_stage_artifact(experiment_id, "input", "ground_truth.nii.gz")
        has_base = storage_service.has_stage_artifact(experiment_id, "fdr", "baseline_significant.nii.gz")

        return ResultsResponse(
            experiment_id=experiment_id,
            status=meta.get("status", "unknown"),
            original=LayerAvailability(available=has_orig, result_id=f"{experiment_id}_orig" if has_orig else None),
            preprocessed=LayerAvailability(available=has_prep, result_id=f"{experiment_id}_prep" if has_prep else None),
            roi=LayerAvailability(available=has_roi, result_id=f"{experiment_id}_roi" if has_roi else None),
            statistical_map=LayerAvailability(available=has_stat, result_id=f"{experiment_id}_stat" if has_stat else None),
            lis_map=LayerAvailability(available=has_lis, result_id=f"{experiment_id}_lis" if has_lis else None),
            fdr=LayerAvailability(
                available=has_fdr,
                result_id=f"{experiment_id}_fdr" if has_fdr else None,
                summary={"discoveries": meta.get("discoveries_proposed", 0)},
            ),
            ground_truth=LayerAvailability(available=has_gt, result_id=f"{experiment_id}_gt" if has_gt else None),
            baseline=LayerAvailability(
                available=has_base,
                result_id=f"{experiment_id}_base" if has_base else None,
                summary={"discoveries": meta.get("discoveries_baseline", 0)},
            ),
            proposed=LayerAvailability(
                available=has_fdr,
                result_id=f"{experiment_id}_prop" if has_fdr else None,
                summary={"discoveries": meta.get("discoveries_proposed", 0)},
            ),
        )

    def get_experiment_metadata(self, experiment_id: str) -> ExperimentMetadataResponse:
        """Retrieve full experiment provenance record."""
        meta = storage_service.get_metadata(experiment_id)
        has_gt = storage_service.has_stage_artifact(experiment_id, "input", "ground_truth.nii.gz")

        return ExperimentMetadataResponse(
            experiment_id=experiment_id,
            created_at=meta.get("created_at", ""),
            filename=meta.get("filename", "unknown"),
            status=meta.get("status", "unknown"),
            shape=meta.get("shape", [0, 0, 0]),
            voxel_spacing=meta.get("voxel_spacing", [1.0, 1.0, 1.0]),
            orientation=meta.get("orientation", "RAS"),
            stages=meta.get("stages", {}),
            parameters=meta.get("parameters", {}),
            discoveries_proposed=meta.get("discoveries_proposed"),
            discoveries_baseline=meta.get("discoveries_baseline"),
            ground_truth_available=has_gt,
        )


result_service = ResultService()
