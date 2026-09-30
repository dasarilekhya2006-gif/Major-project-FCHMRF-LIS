"""Pipeline service coordinating the research computation modules and asynchronous job tracking."""

import threading
import time
import uuid
from typing import Any, Dict, Optional
from fastapi import HTTPException, status
import numpy as np

from backend.schemas.pipeline import (
    StageResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
)
from backend.services.storage_service import storage_service
from backend.research.preprocessing import execute_preprocessing
from backend.research.segmentation import execute_segmentation
from backend.research.statistics import execute_statistics
from backend.research.fchmrf import execute_fchmrf_lis
from backend.research.fdr import execute_baseline_fdr
from src.core.logging import get_logger

logger = get_logger(__name__)

# In-memory registry for job tracking
JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}


class PipelineService:
    """Coordinates execution across research modules, updates metadata, and manages job statuses."""

    def get_job_status(self, experiment_id: str) -> PipelineStatusResponse:
        """Retrieve live execution state and progress for an experiment."""
        # First check active job registry
        for job_id, info in JOB_REGISTRY.items():
            if info.get("experiment_id") == experiment_id:
                return PipelineStatusResponse(
                    experiment_id=experiment_id,
                    job_id=job_id,
                    status=info.get("status", "unknown"),
                    current_stage=info.get("current_stage"),
                    progress=info.get("progress", 0),
                    error=info.get("error"),
                )

        # Fallback to persistent metadata.json
        meta = storage_service.get_metadata(experiment_id)
        current_status = meta.get("status", "uploaded")
        stages = meta.get("stages", {})

        progress = 0
        if current_status == "completed":
            progress = 100
        elif current_status == "uploaded":
            progress = 10
        elif stages.get("fdr") == "completed":
            progress = 90
        elif stages.get("fchmrf_lis") == "completed":
            progress = 75
        elif stages.get("statistics") == "completed":
            progress = 50
        elif stages.get("segmentation") == "completed":
            progress = 35
        elif stages.get("preprocess") == "completed":
            progress = 20

        return PipelineStatusResponse(
            experiment_id=experiment_id,
            status=current_status,
            current_stage=meta.get("current_stage"),
            progress=progress,
            error=meta.get("error"),
        )

    def run_preprocess_stage(
        self,
        experiment_id: str,
        normalize_intensity: bool = True,
        apply_brain_mask: bool = True,
    ) -> StageResponse:
        """Execute Stage 2: Neuroimaging preprocessing."""
        storage_service.get_experiment_dir(experiment_id)
        raw_data, affine = storage_service.load_stage_volume(experiment_id, "input", "raw_volume.nii.gz")

        prep_data, brain_mask = execute_preprocessing(
            raw_data=raw_data,
            affine=affine,
            normalize_intensity=normalize_intensity,
            apply_brain_mask=apply_brain_mask,
        )

        storage_service.save_stage_volume(experiment_id, "preprocess", "preprocessed.nii.gz", prep_data, affine)
        storage_service.save_stage_volume(
            experiment_id, "preprocess", "brain_mask.nii.gz", brain_mask.astype(np.uint8), affine
        )

        storage_service.update_metadata(
            experiment_id,
            {
                "status": "preprocessed",
                "stages": {**storage_service.get_metadata(experiment_id).get("stages", {}), "preprocess": "completed"},
            },
        )

        res_id = f"result_preprocess_{uuid.uuid4().hex[:8]}"
        return StageResponse(
            experiment_id=experiment_id,
            stage="preprocess",
            status="completed",
            result_id=res_id,
            details={"voxels_in_mask": int(np.sum(brain_mask))},
        )

    def run_segmentation_stage(
        self,
        experiment_id: str,
        probability_threshold: float = 0.5,
    ) -> StageResponse:
        """Execute Stage 3: 3D U-Net ROI segmentation."""
        if not storage_service.has_stage_artifact(experiment_id, "preprocess", "preprocessed.nii.gz"):
            # Preprocess automatically if missing
            self.run_preprocess_stage(experiment_id)

        prep_data, affine = storage_service.load_stage_volume(experiment_id, "preprocess", "preprocessed.nii.gz")
        brain_mask, _ = storage_service.load_stage_volume(experiment_id, "preprocess", "brain_mask.nii.gz")
        brain_mask = brain_mask > 0

        roi_mask, dl_prior = execute_segmentation(
            preprocessed_data=prep_data,
            brain_mask=brain_mask,
            probability_threshold=probability_threshold,
        )

        storage_service.save_stage_volume(
            experiment_id, "segmentation", "roi_mask.nii.gz", roi_mask.astype(np.uint8), affine
        )
        storage_service.save_stage_volume(
            experiment_id, "segmentation", "dl_prior.nii.gz", dl_prior.astype(np.float32), affine
        )

        storage_service.update_metadata(
            experiment_id,
            {
                "stages": {**storage_service.get_metadata(experiment_id).get("stages", {}), "segmentation": "completed"},
            },
        )

        res_id = f"result_segment_{uuid.uuid4().hex[:8]}"
        return StageResponse(
            experiment_id=experiment_id,
            stage="segment",
            status="completed",
            result_id=res_id,
            details={"roi_voxels": int(np.sum(roi_mask))},
        )

    def run_statistics_stage(self, experiment_id: str) -> StageResponse:
        """Execute Stage 4: Voxel-level statistical testing."""
        if not storage_service.has_stage_artifact(experiment_id, "preprocess", "preprocessed.nii.gz"):
            self.run_preprocess_stage(experiment_id)

        prep_data, affine = storage_service.load_stage_volume(experiment_id, "preprocess", "preprocessed.nii.gz")
        brain_mask, _ = storage_service.load_stage_volume(experiment_id, "preprocess", "brain_mask.nii.gz")
        brain_mask = brain_mask > 0

        z_scores, p_values = execute_statistics(prep_data, brain_mask)

        storage_service.save_stage_volume(experiment_id, "statistics", "z_scores.nii.gz", z_scores, affine)
        storage_service.save_stage_volume(experiment_id, "statistics", "p_values.nii.gz", p_values, affine)

        storage_service.update_metadata(
            experiment_id,
            {
                "stages": {**storage_service.get_metadata(experiment_id).get("stages", {}), "statistics": "completed"},
            },
        )

        res_id = f"result_statistics_{uuid.uuid4().hex[:8]}"
        return StageResponse(
            experiment_id=experiment_id,
            stage="statistics",
            status="completed",
            result_id=res_id,
        )

    def run_fchmrf_lis_stage(
        self,
        experiment_id: str,
        nominal_alpha: float = 0.05,
        beta: float = 0.8,
        lambda_dl: float = 0.5,
        max_iterations: int = 25,
    ) -> StageResponse:
        """Execute Stage 5: fcHMRF-LIS spatial modeling and FDR control."""
        if not storage_service.has_stage_artifact(experiment_id, "statistics", "z_scores.nii.gz"):
            self.run_statistics_stage(experiment_id)
        if not storage_service.has_stage_artifact(experiment_id, "segmentation", "dl_prior.nii.gz"):
            self.run_segmentation_stage(experiment_id)

        z_scores, affine = storage_service.load_stage_volume(experiment_id, "statistics", "z_scores.nii.gz")
        p_values, _ = storage_service.load_stage_volume(experiment_id, "statistics", "p_values.nii.gz")
        dl_prior, _ = storage_service.load_stage_volume(experiment_id, "segmentation", "dl_prior.nii.gz")
        roi_mask, _ = storage_service.load_stage_volume(experiment_id, "segmentation", "roi_mask.nii.gz")
        roi_mask = roi_mask > 0

        # Run proposed fcHMRF-LIS
        lis_map, sig_mask, summary = execute_fchmrf_lis(
            z_scores=z_scores,
            dl_prior=dl_prior,
            brain_mask=roi_mask,
            nominal_alpha=nominal_alpha,
            beta=beta,
            lambda_dl=lambda_dl,
            max_iterations=max_iterations,
        )

        # Run baseline BH-FDR under identical conditions
        baseline_sig, baseline_count = execute_baseline_fdr(
            z_scores=z_scores,
            p_values=p_values,
            brain_mask=roi_mask,
            nominal_alpha=nominal_alpha,
        )

        storage_service.save_stage_volume(experiment_id, "fchmrf", "lis_map.nii.gz", lis_map, affine)
        storage_service.save_stage_volume(
            experiment_id, "fdr", "significant_mask.nii.gz", sig_mask.astype(np.uint8), affine
        )
        storage_service.save_stage_volume(
            experiment_id, "fdr", "baseline_significant.nii.gz", baseline_sig.astype(np.uint8), affine
        )

        storage_service.update_metadata(
            experiment_id,
            {
                "status": "completed",
                "stages": {
                    **storage_service.get_metadata(experiment_id).get("stages", {}),
                    "fchmrf_lis": "completed",
                    "fdr": "completed",
                },
                "parameters": {
                    "nominal_alpha": nominal_alpha,
                    "beta": beta,
                    "lambda_dl": lambda_dl,
                    "max_iterations": max_iterations,
                },
                "discoveries_proposed": int(np.sum(sig_mask)),
                "discoveries_baseline": baseline_count,
                "lis_threshold": summary.get("lis_threshold", 0.0),
            },
        )

        res_id = f"result_lis_{uuid.uuid4().hex[:8]}"
        stage_details = {
            **summary,
            "graph_stats": summary.get("graph_stats", {}),
        }
        return StageResponse(
            experiment_id=experiment_id,
            stage="fchmrf-lis",
            status="completed",
            result_id=res_id,
            details=stage_details,
        )

    def trigger_complete_pipeline(self, req: PipelineRunRequest) -> PipelineRunResponse:
        """Trigger complete end-to-end research pipeline."""
        storage_service.get_experiment_dir(req.experiment_id)
        job_id = f"job_{uuid.uuid4().hex[:10]}"

        JOB_REGISTRY[job_id] = {
            "experiment_id": req.experiment_id,
            "status": "queued",
            "current_stage": "initialization",
            "progress": 5,
            "start_time": time.time(),
        }

        # Spawn asynchronous worker thread so caller does not block
        def _worker():
            try:
                # 1. Preprocessing
                JOB_REGISTRY[job_id]["status"] = "running"
                JOB_REGISTRY[job_id]["current_stage"] = "preprocessing"
                JOB_REGISTRY[job_id]["progress"] = 20
                self.run_preprocess_stage(req.experiment_id)

                # 2. 3D U-Net Segmentation
                if req.run_segmentation:
                    JOB_REGISTRY[job_id]["current_stage"] = "segmentation"
                    JOB_REGISTRY[job_id]["progress"] = 40
                    self.run_segmentation_stage(req.experiment_id)

                # 3. Statistics
                if req.run_statistics:
                    JOB_REGISTRY[job_id]["current_stage"] = "statistics"
                    JOB_REGISTRY[job_id]["progress"] = 60
                    self.run_statistics_stage(req.experiment_id)

                # 4. fcHMRF-LIS & FDR
                if req.run_fchmrf_lis:
                    JOB_REGISTRY[job_id]["current_stage"] = "fchmrf-lis"
                    JOB_REGISTRY[job_id]["progress"] = 80
                    self.run_fchmrf_lis_stage(
                        experiment_id=req.experiment_id,
                        nominal_alpha=req.nominal_alpha,
                        beta=req.beta,
                        lambda_dl=req.lambda_dl,
                    )

                JOB_REGISTRY[job_id]["status"] = "completed"
                JOB_REGISTRY[job_id]["current_stage"] = "completed"
                JOB_REGISTRY[job_id]["progress"] = 100
                logger.info(f"Pipeline job '{job_id}' completed successfully for experiment '{req.experiment_id}'")

            except Exception as e:
                logger.error(f"Pipeline job '{job_id}' failed: {e}", exc_info=True)
                JOB_REGISTRY[job_id]["status"] = "failed"
                JOB_REGISTRY[job_id]["error"] = str(e)
                storage_service.update_metadata(req.experiment_id, {"status": "failed", "error": str(e)})

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        return PipelineRunResponse(
            experiment_id=req.experiment_id,
            job_id=job_id,
            status="queued",
            message="Research pipeline execution started in background.",
        )


pipeline_service = PipelineService()
