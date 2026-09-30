"""Controlled experiment storage management service.

Prevents arbitrary client filesystem paths and maintains strict experiment sandbox isolation.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from fastapi import HTTPException, status
import nibabel as nib
import numpy as np

from backend.core.config import settings
from backend.core.security import validate_experiment_id
from src.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    """Manages secure on-disk directory trees and NIfTI artifact storage for each experiment."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or settings.storage_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_dir(self, experiment_id: str, must_exist: bool = True) -> Path:
        """Resolve and validate experiment directory path."""
        valid_id = validate_experiment_id(experiment_id)
        exp_dir = (self.root_dir / valid_id).resolve()

        # Security check: Ensure exp_dir is strictly inside root_dir
        try:
            exp_dir.relative_to(self.root_dir.resolve())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "ACCESS_DENIED", "message": "Directory traversal attempt detected."},
            )

        if must_exist and not exp_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "EXPERIMENT_NOT_FOUND",
                    "message": f"Experiment '{experiment_id}' does not exist.",
                    "experiment_id": experiment_id,
                },
            )

        return exp_dir

    def create_experiment(self, experiment_id: str, filename: str, initial_meta: Optional[Dict[str, Any]] = None) -> Path:
        """Initialize directory layout and metadata for a new experiment."""
        exp_dir = self.get_experiment_dir(experiment_id, must_exist=False)
        exp_dir.mkdir(parents=True, exist_ok=True)

        for sub in ["input", "preprocess", "segmentation", "statistics", "fchmrf", "fdr", "visualization"]:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)

        meta = {
            "experiment_id": experiment_id,
            "filename": filename,
            "created_at": datetime.utcnow().isoformat(),
            "status": "uploaded",
            "stages": {
                "upload": "completed",
                "preprocess": "pending",
                "segmentation": "pending",
                "statistics": "pending",
                "fchmrf_lis": "pending",
                "fdr": "pending",
            },
            "parameters": {},
        }
        if initial_meta:
            meta.update(initial_meta)

        self.save_metadata(experiment_id, meta)
        logger.info(f"Initialized storage tree for experiment '{experiment_id}'")
        return exp_dir

    def get_metadata(self, experiment_id: str) -> Dict[str, Any]:
        """Read metadata.json for the specified experiment."""
        exp_dir = self.get_experiment_dir(experiment_id)
        meta_path = exp_dir / "metadata.json"
        if not meta_path.exists():
            return {"experiment_id": experiment_id, "status": "unknown"}

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for '{experiment_id}': {e}")
            return {"experiment_id": experiment_id, "status": "error", "error": str(e)}

    def save_metadata(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        """Persist updated metadata.json."""
        exp_dir = self.get_experiment_dir(experiment_id)
        meta_path = exp_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def update_metadata(self, experiment_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Atomically merge updates into existing metadata.json."""
        meta = self.get_metadata(experiment_id)
        meta.update(updates)
        self.save_metadata(experiment_id, meta)
        return meta

    def save_stage_volume(
        self,
        experiment_id: str,
        stage: str,
        filename: str,
        data: np.ndarray,
        affine: np.ndarray,
    ) -> Path:
        """Save volumetric numpy array as compressed NIfTI image (.nii.gz)."""
        exp_dir = self.get_experiment_dir(experiment_id)
        stage_dir = exp_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        target_file = stage_dir / (filename if filename.endswith(".nii.gz") else f"{filename}.nii.gz")
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(target_file))
        logger.debug(f"Saved {stage} artifact for '{experiment_id}' at {target_file.name}")
        return target_file

    def load_stage_volume(
        self,
        experiment_id: str,
        stage: str,
        filename: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load volumetric numpy array and affine from NIfTI artifact."""
        exp_dir = self.get_experiment_dir(experiment_id)
        target_file = exp_dir / stage / (filename if filename.endswith(".nii.gz") else f"{filename}.nii.gz")

        if not target_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "ARTIFACT_NOT_FOUND",
                    "message": f"Artifact '{filename}' for stage '{stage}' does not exist in experiment '{experiment_id}'.",
                },
            )

        img = nib.load(str(target_file))
        data = np.asanyarray(img.dataobj)
        affine = img.affine
        return data, affine

    def has_stage_artifact(self, experiment_id: str, stage: str, filename: str) -> bool:
        """Check whether a given artifact file exists on disk for the experiment."""
        try:
            exp_dir = self.get_experiment_dir(experiment_id)
            target_file = exp_dir / stage / (filename if filename.endswith(".nii.gz") else f"{filename}.nii.gz")
            return target_file.exists()
        except Exception:
            return False

    def list_experiments(self) -> list:
        """List all experiments stored on disk with metadata."""
        experiments = []
        if not self.root_dir.exists():
            return []
        for exp_dir in self.root_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "metadata.json").exists():
                try:
                    with open(exp_dir / "metadata.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    experiments.append(meta)
                except Exception:
                    continue
        # Sort descending by creation timestamp
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments

    def get_artifact_path(self, experiment_id: str, stage: str, filename: str) -> Path:
        """Safely resolve an artifact path for download."""
        exp_dir = self.get_experiment_dir(experiment_id)
        # Prevent any path traversal in stage or filename
        safe_stage = Path(stage).name
        safe_file = Path(filename).name
        target = exp_dir / safe_stage / safe_file
        try:
            target.resolve().relative_to(exp_dir.resolve())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "ACCESS_DENIED", "message": "Invalid file path."},
            )
        if not target.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "FILE_NOT_FOUND", "message": f"File '{filename}' does not exist."},
            )
        return target


storage_service = StorageService()

