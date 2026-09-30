"""Unit and integration test suite for the Master Neuroimaging Research Pipeline."""

from pathlib import Path
import pytest
import numpy as np

from src.pipeline.stages import PipelineStage, StageStatus
from src.pipeline.master_pipeline import MasterPipeline, MasterPipelineConfig
from src.pipeline.reporting import collect_system_environment, generate_experiment_report


class TestEnvironmentAndReporting:
    """Test runtime system environment discovery and report generation."""

    def test_collect_system_environment(self):
        env = collect_system_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "numpy_version" in env
        assert "torch_version" in env
        assert "cuda_available" in env

    def test_generate_experiment_report(self, tmp_path):
        env = collect_system_environment()
        reports = generate_experiment_report(
            experiment_id="test_exp",
            stage_results={},
            pipeline_config={"test_param": 123},
            output_dir=tmp_path,
            start_time="2026-09-27T00:00:00",
            end_time="2026-09-27T00:01:00",
            total_duration_seconds=60.0,
        )
        assert reports["json_report"].exists()
        assert reports["markdown_report"].exists()
        content = reports["markdown_report"].read_text(encoding="utf-8")
        assert "test_exp" in content
        assert "Computational Environment" in content


class TestMasterPipelineEndToEnd:
    """Test single stage, end-to-end execution, and caching behaviors."""

    def test_pipeline_end_to_end_and_caching(self, tmp_path):
        """Execute complete pipeline end-to-end on synthetic data and verify caching."""
        config = MasterPipelineConfig(
            experiment_id="test_run",
            output_dir=tmp_path / "pipeline_outputs",
            device="cpu",
            force_recompute=False,
            run_adaptive_selection=False,
            seed=42,
            max_mf_iterations=10,
        )

        pipeline = MasterPipeline(config=config)

        # -------------------------------------------------------------
        # 1. Run Complete Pipeline (Initial Execution: All Recomputed)
        # -------------------------------------------------------------
        results = pipeline.run()

        assert PipelineStage.DATA_LOADING in results
        assert PipelineStage.PREPROCESSING in results
        assert PipelineStage.ROI_DETECTION in results
        assert PipelineStage.STATISTICAL_TESTING in results
        assert PipelineStage.FCHMRF_LIS in results

        for stage, res in results.items():
            assert res.status == StageStatus.COMPLETED
            assert res.recomputed is True
            assert res.elapsed_seconds > 0.0

        # Check key artifact outputs exist on disk
        assert (pipeline.dir_preproc / "test_run_preprocessed.nii.gz").exists()
        assert (pipeline.dir_segment / "test_run_roi_binary.nii.gz").exists()
        assert (pipeline.dir_segment / "test_run_roi_prob.nii.gz").exists()
        assert (pipeline.dir_stats / "test_run_zscores.nii.gz").exists()
        assert (pipeline.dir_fchmrf / "test_run_lis_map.nii.gz").exists()
        assert (pipeline.dir_fchmrf / "test_run_significant_mask.nii.gz").exists()
        assert (pipeline.dir_figures / "test_run_fchmrf_summary.png").exists()
        assert (pipeline.dir_reports / "test_run_report.json").exists()
        assert (pipeline.dir_reports / "test_run_report.md").exists()

        # -------------------------------------------------------------
        # 2. Second Execution with force_recompute=False (All Skipped)
        # -------------------------------------------------------------
        pipeline_cached = MasterPipeline(config=config)
        cached_results = pipeline_cached.run()

        for stage, res in cached_results.items():
            if stage != PipelineStage.DATA_LOADING:  # Data loading is lightweight check
                assert res.status == StageStatus.SKIPPED
                assert res.recomputed is False

        # -------------------------------------------------------------
        # 3. Third Execution with force_recompute=True (Recomputed)
        # -------------------------------------------------------------
        config.force_recompute = True
        pipeline_recompute = MasterPipeline(config=config)
        recompute_results = pipeline_recompute.run()

        for stage, res in recompute_results.items():
            assert res.status == StageStatus.COMPLETED
            assert res.recomputed is True


class TestIndependentStageExecution:
    """Verify that each stage can be executed independently."""

    def test_run_stage_data_loading_independent(self, tmp_path):
        config = MasterPipelineConfig(
            experiment_id="stage1_test",
            output_dir=tmp_path / "stage1_out",
            device="cpu",
        )
        pipeline = MasterPipeline(config=config)
        res = pipeline.run_stage_data_loading()
        assert res.stage == PipelineStage.DATA_LOADING
        assert res.status == StageStatus.COMPLETED
        assert "raw_volume" in res.output_files
        assert res.output_files["raw_volume"].exists()
