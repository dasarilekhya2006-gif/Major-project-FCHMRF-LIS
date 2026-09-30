"""Comprehensive test suite for Phase 11 FastAPI backend.

Tests upload validation, security, pipeline execution, results, and visualization endpoints.
"""

import gzip
import io
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np

from backend.main import create_app
from backend.core.config import settings
from backend.services.storage_service import StorageService, storage_service


@pytest.fixture(scope="module")
def test_env():
    """Create isolated temporary storage directory for tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_backend_storage_"))
    original_storage = settings.storage_dir
    settings.storage_dir = temp_dir
    storage_service.root_dir = temp_dir

    app = create_app()
    client = TestClient(app)

    yield {"client": client, "storage_dir": temp_dir}

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    settings.storage_dir = original_storage
    storage_service.root_dir = original_storage


def create_mock_nifti_bytes(shape=(12, 12, 8), gzipped=True) -> bytes:
    """Generate in-memory valid NIfTI bytes with synthetic brain activation."""
    data = np.zeros(shape, dtype=np.float32)
    data[3:9, 3:9, 2:6] = 5.0  # Synthetic brain region
    data[5:7, 5:7, 3:5] = 12.0  # Synthetic focal cluster
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    img = nib.Nifti1Image(data, affine)

    bio = io.BytesIO()
    img.to_bytes()  # check
    # Write to temp file then read bytes
    with tempfile.NamedTemporaryFile(suffix=".nii.gz" if gzipped else ".nii", delete=False) as f:
        tmp_name = f.name
    try:
        nib.save(img, tmp_name)
        with open(tmp_name, "rb") as f:
            raw_bytes = f.read()
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    return raw_bytes


# ==============================================================================
# 1. Health and Discovery Tests
# ==============================================================================


def test_health_check(test_env):
    client: TestClient = test_env["client"]
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "fcHMRF-LIS Neuroimaging Backend"
    assert data["storage_accessible"] is True


# ==============================================================================
# 2. Upload and NIfTI Security Tests
# ==============================================================================


def test_upload_valid_nifti_gzipped(test_env):
    client: TestClient = test_env["client"]
    nii_bytes = create_mock_nifti_bytes(shape=(10, 10, 6), gzipped=True)

    files = {"file": ("patient_scan.nii.gz", nii_bytes, "application/gzip")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert "experiment_id" in data
    assert data["filename"] == "patient_scan.nii.gz"
    assert data["status"] == "uploaded"
    assert data["shape"] == [10, 10, 6]
    assert len(data["voxel_spacing"]) == 3


def test_upload_invalid_non_nifti(test_env):
    client: TestClient = test_env["client"]
    fake_bytes = b"This is a plain text file pretending to be nifti."

    files = {"file": ("malicious.nii.gz", fake_bytes, "application/gzip")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "INVALID_NIFTI"


def test_upload_path_traversal_attempt(test_env):
    client: TestClient = test_env["client"]
    nii_bytes = create_mock_nifti_bytes(shape=(8, 8, 6), gzipped=True)

    # Attempt path traversal in filename
    files = {"file": ("../../etc/passwd.nii.gz", nii_bytes, "application/gzip")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 201
    data = resp.json()
    # Sanitized filename must not contain directory traversal slashes
    assert "/" not in data["filename"]
    assert ".." not in data["filename"]


def test_upload_oversized_file(test_env, monkeypatch):
    client: TestClient = test_env["client"]
    # Temporarily set limit to 0MB for testing
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)

    nii_bytes = create_mock_nifti_bytes(shape=(6, 6, 6), gzipped=True)
    files = {"file": ("oversized.nii.gz", nii_bytes, "application/gzip")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 422
    assert "exceeds maximum allowed limit" in resp.json()["message"]


# ==============================================================================
# 3. Stage-by-Stage Research Pipeline Tests
# ==============================================================================


def test_stage_pipeline_lifecycle(test_env):
    client: TestClient = test_env["client"]
    nii_bytes = create_mock_nifti_bytes(shape=(12, 12, 8), gzipped=True)

    # 1. Upload
    files = {"file": ("brain_t1.nii.gz", nii_bytes, "application/gzip")}
    up_resp = client.post("/upload", files=files)
    assert up_resp.status_code == 201
    exp_id = up_resp.json()["experiment_id"]

    # 2. Preprocess
    prep_resp = client.post("/preprocess", json={"experiment_id": exp_id, "normalize_intensity": True})
    assert prep_resp.status_code == 200
    assert prep_resp.json()["stage"] == "preprocess"
    assert prep_resp.json()["status"] == "completed"

    # 3. Segmentation
    seg_resp = client.post("/segment", json={"experiment_id": exp_id, "probability_threshold": 0.5})
    assert seg_resp.status_code == 200
    assert seg_resp.json()["stage"] == "segment"
    assert seg_resp.json()["status"] == "completed"

    # 4. Statistics
    stat_resp = client.post("/statistics", json={"experiment_id": exp_id})
    assert stat_resp.status_code == 200
    assert stat_resp.json()["stage"] == "statistics"
    assert stat_resp.json()["status"] == "completed"

    # 5. fcHMRF-LIS
    fchmrf_resp = client.post(
        "/fchmrf-lis",
        json={"experiment_id": exp_id, "nominal_alpha": 0.05, "beta": 0.5, "lambda_dl": 0.5, "max_iterations": 10},
    )
    assert fchmrf_resp.status_code == 200
    assert fchmrf_resp.json()["stage"] == "fchmrf-lis"
    assert fchmrf_resp.json()["status"] == "completed"


# ==============================================================================
# 4. End-to-End Pipeline & Status Tests
# ==============================================================================


def test_complete_pipeline_run_and_status(test_env):
    client: TestClient = test_env["client"]
    nii_bytes = create_mock_nifti_bytes(shape=(10, 10, 6), gzipped=True)

    files = {"file": ("test_run.nii.gz", nii_bytes, "application/gzip")}
    up_resp = client.post("/upload", files=files)
    exp_id = up_resp.json()["experiment_id"]

    # Trigger complete pipeline
    run_resp = client.post(
        "/run-pipeline",
        json={
            "experiment_id": exp_id,
            "run_segmentation": True,
            "run_statistics": True,
            "run_fchmrf_lis": True,
            "run_fdr": True,
            "nominal_alpha": 0.05,
        },
    )
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert data["experiment_id"] == exp_id
    assert "job_id" in data

    # Poll status (in-memory job thread will progress)
    status_resp = client.get(f"/pipeline/status/{exp_id}")
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["status"] in ("queued", "running", "completed")
    assert 0 <= st_data["progress"] <= 100


# ==============================================================================
# 5. Results and Metadata Endpoints
# ==============================================================================


def test_results_and_experiment_metadata(test_env):
    client: TestClient = test_env["client"]
    nii_bytes = create_mock_nifti_bytes(shape=(10, 10, 6), gzipped=True)

    files = {"file": ("sample_experiment.nii.gz", nii_bytes, "application/gzip")}
    up_resp = client.post("/upload", files=files)
    exp_id = up_resp.json()["experiment_id"]

    # Execute all stages sequentially
    client.post("/preprocess", json={"experiment_id": exp_id})
    client.post("/segment", json={"experiment_id": exp_id})
    client.post("/statistics", json={"experiment_id": exp_id})
    client.post("/fchmrf-lis", json={"experiment_id": exp_id, "max_iterations": 5})

    # Test GET /results/{id}
    res_resp = client.get(f"/results/{exp_id}")
    assert res_resp.status_code == 200
    res_data = res_resp.json()
    assert res_data["original"]["available"] is True
    assert res_data["preprocessed"]["available"] is True
    assert res_data["roi"]["available"] is True
    assert res_data["statistical_map"]["available"] is True
    assert res_data["lis_map"]["available"] is True
    assert res_data["fdr"]["available"] is True
    assert res_data["proposed"]["available"] is True

    # Test GET /experiment/{id}
    meta_resp = client.get(f"/experiment/{exp_id}")
    assert meta_resp.status_code == 200
    meta_data = meta_resp.json()
    assert meta_data["experiment_id"] == exp_id
    assert meta_data["status"] == "completed"
    assert meta_data["shape"] == [10, 10, 6]
    assert "fchmrf_lis" in meta_data["stages"]


# ==============================================================================
# 6. Visualization Endpoints (Phase 10 Support)
# ==============================================================================


def test_visualization_metadata_and_slices(test_env):
    client: TestClient = test_env["client"]
    nii_bytes = create_mock_nifti_bytes(shape=(12, 12, 8), gzipped=True)

    files = {"file": ("vis_test.nii.gz", nii_bytes, "application/gzip")}
    up_resp = client.post("/upload", files=files)
    exp_id = up_resp.json()["experiment_id"]

    # Process
    client.post("/preprocess", json={"experiment_id": exp_id})
    client.post("/statistics", json={"experiment_id": exp_id})
    client.post("/fchmrf-lis", json={"experiment_id": exp_id, "max_iterations": 5})

    # 1. Visualization metadata
    vmeta_resp = client.get(f"/visualization/{exp_id}")
    assert vmeta_resp.status_code == 200
    vmeta = vmeta_resp.json()
    assert vmeta["visualization"]["supports_2d"] is True
    assert vmeta["visualization"]["supports_3d"] is True
    assert "original" in vmeta["value_ranges"]
    assert "statistical_map" in vmeta["value_ranges"]

    # 2. 2D Slice extraction (Axial, Coronal, Sagittal)
    for plane in ["axial", "coronal", "sagittal"]:
        s_resp = client.get(f"/visualization/{exp_id}/slice", params={"orientation": plane, "slice_idx": 3})
        assert s_resp.status_code == 200
        s_data = s_resp.json()
        assert s_data["orientation"] == plane
        assert len(s_data["mri_slice"]) > 0
        assert s_data["statistical_slice"] is not None

    # 3. 3D Mesh extraction
    mesh_resp = client.get(f"/visualization/{exp_id}/3d-mesh", params={"step_size": 2})
    assert mesh_resp.status_code == 200
    mesh_data = mesh_resp.json()
    assert mesh_data["experiment_id"] == exp_id
    assert mesh_data["brain_shell"] is not None
    assert mesh_data["brain_shell"]["num_vertices"] > 0
    assert mesh_data["brain_shell"]["num_faces"] > 0


# ==============================================================================
# 7. Security and Error Handling Tests
# ==============================================================================


def test_nonexistent_experiment_404(test_env):
    client: TestClient = test_env["client"]
    resp = client.get("/results/exp_nonexistent999")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "EXPERIMENT_NOT_FOUND"


def test_invalid_experiment_id_format_400(test_env):
    client: TestClient = test_env["client"]
    # Pass malicious characters into path
    resp = client.get("/results/exp_../../etc")
    assert resp.status_code in (400, 404)


# ==============================================================================
# 8. Phase 12 Research Dashboard Supporting Endpoints Tests
# ==============================================================================


def test_list_experiments_endpoint(test_env):
    """Test GET /experiments lists past experiments."""
    client: TestClient = test_env["client"]
    resp = client.get("/experiments")
    assert resp.status_code == 200
    experiments = resp.json()
    assert isinstance(experiments, list)
    assert len(experiments) > 0
    assert "experiment_id" in experiments[0]


def test_download_artifact_endpoint(test_env):
    """Test GET /download/{experiment_id}/{stage}/{filename} safely serves artifact."""
    client: TestClient = test_env["client"]
    # Upload and preprocess to generate preprocessed.nii.gz
    nii_bytes = create_mock_nifti_bytes()
    up = client.post("/upload", files={"file": ("brain.nii.gz", nii_bytes, "application/gzip")})
    exp_id = up.json()["experiment_id"]
    client.post("/preprocess", json={"experiment_id": exp_id})

    # Download
    dl_resp = client.get(f"/download/{exp_id}/preprocess/preprocessed.nii.gz")
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 0
    assert dl_resp.headers["content-type"] == "application/octet-stream"

    # Test download traversal rejection
    dl_bad = client.get(f"/download/{exp_id}/preprocess/../../secret.txt")
    assert dl_bad.status_code in (400, 404)

