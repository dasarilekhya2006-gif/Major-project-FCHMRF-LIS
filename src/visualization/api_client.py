"""API client for communicating with the Phase 11 FastAPI backend."""

import time
from typing import Any, Dict, List, Optional
import httpx
from src.core.logging import get_logger

logger = get_logger(__name__)


class NeuroimagingAPIClient:
    """HTTP Client interfacing Phase 10 frontend with Phase 11 FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Verify API service reachability."""
        with httpx.Client(base_url=self.base_url, timeout=5.0) as client:
            resp = client.get("/health")
            resp.raise_for_status()
            return resp.json()

    def upload_volume(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Upload NIfTI volume to POST /upload."""
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            resp = client.post("/upload", files=files)
            resp.raise_for_status()
            return resp.json()

    def run_pipeline(
        self,
        experiment_id: str,
        nominal_alpha: float = 0.05,
        beta: float = 0.8,
        lambda_dl: float = 0.5,
    ) -> Dict[str, Any]:
        """Trigger complete research pipeline via POST /run-pipeline."""
        payload = {
            "experiment_id": experiment_id,
            "run_segmentation": True,
            "run_statistics": True,
            "run_fchmrf_lis": True,
            "run_fdr": True,
            "nominal_alpha": nominal_alpha,
            "beta": beta,
            "lambda_dl": lambda_dl,
        }
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            resp = client.post("/run-pipeline", json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_pipeline_status(self, experiment_id: str) -> Dict[str, Any]:
        """Poll pipeline execution progress via GET /pipeline/status/{id}."""
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            resp = client.get(f"/pipeline/status/{experiment_id}")
            resp.raise_for_status()
            return resp.json()

    def poll_pipeline_until_complete(
        self,
        experiment_id: str,
        poll_interval: float = 0.5,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Block until pipeline status transitions to 'completed' or 'failed'."""
        start = time.time()
        while time.time() - start < timeout:
            status_info = self.get_pipeline_status(experiment_id)
            status_val = status_info.get("status")
            if status_val in ("completed", "failed"):
                return status_info
            time.sleep(poll_interval)
        return {"status": "timeout", "error": "Pipeline polling timed out."}

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve available research results via GET /results/{id}."""
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            resp = client.get(f"/results/{experiment_id}")
            resp.raise_for_status()
            return resp.json()

    def get_visualization_meta(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve visualization metadata and value ranges via GET /visualization/{id}."""
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            resp = client.get(f"/visualization/{experiment_id}")
            resp.raise_for_status()
            return resp.json()

    def get_slice_data(
        self,
        experiment_id: str,
        orientation: str = "axial",
        slice_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve multi-layer raw 2D slice data via GET /visualization/{id}/slice."""
        params: Dict[str, Any] = {"orientation": orientation}
        if slice_idx is not None:
            params["slice_idx"] = slice_idx
        with httpx.Client(base_url=self.base_url, timeout=15.0) as client:
            resp = client.get(f"/visualization/{experiment_id}/slice", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_3d_mesh(self, experiment_id: str, step_size: int = 1) -> Dict[str, Any]:
        """Retrieve 3D Marching Cubes surface meshes via GET /visualization/{id}/3d-mesh."""
        params = {"step_size": step_size}
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            resp = client.get(f"/visualization/{experiment_id}/3d-mesh", params=params)
            resp.raise_for_status()
            return resp.json()

    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve full experiment history and parameters via GET /experiment/{id}."""
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            resp = client.get(f"/experiment/{experiment_id}")
            resp.raise_for_status()
            return resp.json()
