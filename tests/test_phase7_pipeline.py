"""Integration tests for Phase 7: Master Research Pipeline Integration.

Tests the full sequential workflow:
NIfTI input -> preprocessing -> 3D U-Net ROI -> statistical testing ->
fcHMRF-LIS -> adaptive parameter selection -> FDR-controlled results.
"""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from data import load_nifti
from src.pipeline.master_pipeline import MasterPipeline, MasterPipelineConfig
from src.pipeline.stages import PipelineStage, StageStatus


@pytest.fixture
def test_volume(tmp_path: Path) -> Path:
    """Generate a small synthetic 3D NIfTI volume with signal."""
    vol_path = tmp_path / "synthetic_brain.nii.gz"
    rng = np.random.default_rng(42)
    # Background noise
    data = rng.normal(loc=0.0, scale=1.0, size=(24, 24, 12)).astype(np.float32)
    # Ground truth activated cluster
    data[10:15, 10:15, 4:8] += 3.8
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(vol_path))
    return vol_path


def test_full_integrated_pipeline_e2e(tmp_path: Path, test_volume: Path):
    """Verify that the integrated pipeline executes end-to-end and saves all intermediate outputs."""
    out_dir = tmp_path / "pipeline_output"
    config = MasterPipelineConfig(
        experiment_id="phase7_e2e_test",
        output_dir=out_dir,
        raw_input_path=test_volume,
        device="cpu",
        run_adaptive_selection=True,
        bo_iterations=2,
        bo_initial=2,
        max_mf_iterations=10,
        force_recompute=True,
        seed=42,
    )

    pipeline = MasterPipeline(config=config)
    results = pipeline.run()

    # 1. Verify all stages completed successfully
    expected_stages = [
        PipelineStage.DATA_LOADING,
        PipelineStage.PREPROCESSING,
        PipelineStage.ROI_DETECTION,
        PipelineStage.STATISTICAL_TESTING,
        PipelineStage.FCHMRF_LIS,
        PipelineStage.ADAPTIVE_SELECTION,
    ]
    for stg in expected_stages:
        assert stg in results, f"Missing stage in results: {stg.value}"
        assert results[stg].status == StageStatus.COMPLETED, f"Stage {stg.value} did not complete: {results[stg].error_message}"
        assert results[stg].recomputed is True

    # 2. Verify all intermediate artifacts exist on disk
    # Stage 1: Data loading
    assert results[PipelineStage.DATA_LOADING].output_files["raw_volume"].exists()

    # Stage 2: Preprocessing
    preproc_file = results[PipelineStage.PREPROCESSING].output_files["preprocessed_nifti"]
    qc_fig = results[PipelineStage.PREPROCESSING].output_files["qc_figure"]
    assert preproc_file.exists()
    assert qc_fig.exists()
    preproc_vol = load_nifti(preproc_file)
    assert not np.isnan(preproc_vol.data).any()

    # Stage 3: ROI detection
    roi_bin = results[PipelineStage.ROI_DETECTION].output_files["roi_binary"]
    roi_prob = results[PipelineStage.ROI_DETECTION].output_files["roi_prob"]
    assert roi_bin.exists()
    assert roi_prob.exists()

    # Stage 4: Statistical testing
    z_file = results[PipelineStage.STATISTICAL_TESTING].output_files["z_scores"]
    p_file = results[PipelineStage.STATISTICAL_TESTING].output_files["p_values"]
    ortho_fig = results[PipelineStage.STATISTICAL_TESTING].output_files["ortho_figure"]
    assert z_file.exists()
    assert p_file.exists()
    assert ortho_fig.exists()

    # Stage 5: Baseline fcHMRF-LIS
    lis_file = results[PipelineStage.FCHMRF_LIS].output_files["lis_map"]
    sig_file = results[PipelineStage.FCHMRF_LIS].output_files["significant_mask"]
    prob_file = results[PipelineStage.FCHMRF_LIS].output_files["posterior_prob"]
    sum_json = results[PipelineStage.FCHMRF_LIS].output_files["summary_json"]
    sum_fig = results[PipelineStage.FCHMRF_LIS].output_files["summary_figure"]
    assert lis_file.exists()
    assert sig_file.exists()
    assert prob_file.exists()
    assert sum_json.exists()
    assert sum_fig.exists()

    # Stage 6: Adaptive Selection & Final adaptively-tuned FDR results
    sel_yaml = results[PipelineStage.ADAPTIVE_SELECTION].output_files["selected_yaml"]
    hist_json = results[PipelineStage.ADAPTIVE_SELECTION].output_files["history_json"]
    bo_fig = results[PipelineStage.ADAPTIVE_SELECTION].output_files["summary_figure"]
    adapt_sig = results[PipelineStage.ADAPTIVE_SELECTION].output_files["adaptive_significant_mask"]
    adapt_lis = results[PipelineStage.ADAPTIVE_SELECTION].output_files["adaptive_lis_map"]
    assert sel_yaml.exists()
    assert hist_json.exists()
    assert bo_fig.exists()
    assert adapt_sig.exists()
    assert adapt_lis.exists()

    # Stage 6 metrics
    assert "best_score" in results[PipelineStage.ADAPTIVE_SELECTION].metrics
    assert "adaptive_discoveries" in results[PipelineStage.ADAPTIVE_SELECTION].metrics

    # Verify Reports
    report_json = out_dir / "reports" / "phase7_e2e_test_report.json"
    report_md = out_dir / "reports" / "phase7_e2e_test_report.md"
    assert report_json.exists()
    assert report_md.exists()

    with open(report_json, "r") as f:
        rep_data = json.load(f)
        assert rep_data["experiment_id"] == "phase7_e2e_test"
        assert len(rep_data["stages"]) == 6


def test_pipeline_clean_failure_handling(tmp_path: Path):
    """Verify that an invalid or corrupt input halts the pipeline gracefully without fabricating outputs."""
    corrupt_file = tmp_path / "corrupt_image.nii.gz"
    corrupt_file.write_text("This is not a valid NIfTI header or data stream.")

    out_dir = tmp_path / "pipeline_fail_out"
    config = MasterPipelineConfig(
        experiment_id="phase7_fail_test",
        output_dir=out_dir,
        raw_input_path=corrupt_file,
        force_recompute=True,
    )

    pipeline = MasterPipeline(config=config)
    results = pipeline.run()

    # Data loading stage must report FAILED
    assert PipelineStage.DATA_LOADING in results
    res_s1 = results[PipelineStage.DATA_LOADING]
    assert res_s1.status == StageStatus.FAILED
    assert res_s1.error_message is not None
    assert len(res_s1.error_message) > 0

    # Downstream stages MUST NOT have executed or fabricated outputs
    assert PipelineStage.PREPROCESSING not in results
    assert PipelineStage.ROI_DETECTION not in results
    assert PipelineStage.STATISTICAL_TESTING not in results
    assert PipelineStage.FCHMRF_LIS not in results

    # Report must still be generated recording the exact failure
    report_json = out_dir / "reports" / "phase7_fail_test_report.json"
    report_md = out_dir / "reports" / "phase7_fail_test_report.md"
    assert report_json.exists()
    assert report_md.exists()

    with open(report_md, "r") as f:
        md_text = f.read()
        assert "FAILED" in md_text
        assert "Validation failed" in md_text or "Error" in md_text


def test_pipeline_caching_behavior(tmp_path: Path, test_volume: Path):
    """Verify that pipeline properly reuses cached artifacts when force_recompute=False."""
    out_dir = tmp_path / "pipeline_cache_test"
    config = MasterPipelineConfig(
        experiment_id="cache_run",
        output_dir=out_dir,
        raw_input_path=test_volume,
        device="cpu",
        run_adaptive_selection=False,
        force_recompute=True,
        seed=42,
    )

    # First execution: recompute
    p1 = MasterPipeline(config=config)
    res1 = p1.run()
    for s, r in res1.items():
        assert r.status == StageStatus.COMPLETED
        assert r.recomputed is True

    # Second execution: cached
    config.force_recompute = False
    p2 = MasterPipeline(config=config)
    res2 = p2.run()
    for s in [PipelineStage.PREPROCESSING, PipelineStage.ROI_DETECTION, PipelineStage.STATISTICAL_TESTING, PipelineStage.FCHMRF_LIS]:
        assert res2[s].status == StageStatus.SKIPPED
        assert res2[s].recomputed is False


def test_cli_entry_point(tmp_path: Path, test_volume: Path):
    """Verify that the top-level run_pipeline.py CLI executes successfully with exit code 0."""
    out_dir = tmp_path / "cli_test_out"
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--input", str(test_volume),
        "--output-dir", str(out_dir),
        "--experiment-id", "cli_test_exp",
        "--device", "cpu",
        "--force-recompute",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"CLI failed with stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "PIPELINE EXECUTION SUMMARY: cli_test_exp" in result.stdout
    assert "data_loading" in result.stdout
    assert "fchmrf_lis" in result.stdout
    assert (out_dir / "reports" / "cli_test_exp_report.md").exists()
