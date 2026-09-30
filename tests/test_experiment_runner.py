"""Comprehensive unit and integration tests for automated research experiments."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.experiments.schemas import (
    AdaptiveConfig,
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    FCHMRFConfig,
    PreprocessingConfig,
    StatisticalTestConfig,
    UNetConfig,
    get_current_system_metadata,
)
from src.experiments.runner import ExperimentRunner
from src.experiments.matrix import build_experiment_matrix, load_suite_from_yaml
from src.experiments.reproducibility import reproduce_experiment
from src.experiments.tables import (
    export_suite_summary_csv,
    generate_comparison_latex_table,
    generate_comparison_markdown_table,
)


class TestExperimentSchemas:
    """Test serialization, deserialization, and schema integrity."""

    def test_experiment_config_json_yaml_roundtrip(self, tmp_path: Path):
        cfg = ExperimentConfig(
            experiment_id="test_exp_001",
            name="Unit Test Experiment",
            seed=123,
            dataset=DatasetConfig(grid_size=(24, 24, 12), snr=2.5),
            preprocessing=PreprocessingConfig(norm_method="robust"),
            unet=UNetConfig(enabled=True, roi_threshold=0.6),
            statistical_test=StatisticalTestConfig(test_type="one_sample_t"),
            fchmrf=FCHMRFConfig(beta=0.9, lambda_dl=0.4),
            adaptive=AdaptiveConfig(enabled=False),
        )

        # JSON Roundtrip
        json_file = tmp_path / "config.json"
        cfg.save_json(json_file)
        assert json_file.exists()
        loaded_json = ExperimentConfig.from_json(json_file)
        assert loaded_json.experiment_id == "test_exp_001"
        assert loaded_json.dataset.snr == 2.5
        assert loaded_json.fchmrf.beta == 0.9

        # YAML Roundtrip
        yaml_file = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_file)
        assert yaml_file.exists()
        loaded_yaml = ExperimentConfig.from_yaml(yaml_file)
        assert loaded_yaml.seed == 123
        assert loaded_yaml.preprocessing.norm_method == "robust"

    def test_system_metadata_capture(self):
        meta = get_current_system_metadata()
        assert "timestamp" in meta
        assert "platform" in meta
        assert "python_version" in meta
        assert "numpy_version" in meta


class TestExperimentExecution:
    """Test end-to-end execution of experiments and output artifact generation."""

    def test_single_experiment_execution(self, tmp_path: Path):
        out_dir = tmp_path / "run_test"
        cfg = ExperimentConfig(
            experiment_id="unit_exp_01",
            name="Single Experiment Test",
            seed=42,
            output_dir=str(out_dir),
            dataset=DatasetConfig(
                dataset_type="synthetic",
                grid_size=(20, 20, 10),
                snr=2.5,
                num_clusters=2,
                cluster_radius=3,
            ),
            unet=UNetConfig(enabled=True),
            fchmrf=FCHMRFConfig(
                method_mode="proposed_fchmrf",
                beta=0.6,
                lambda_dl=0.5,
                max_iterations=10,
            ),
            adaptive=AdaptiveConfig(enabled=False),
        )

        runner = ExperimentRunner(base_output_dir=out_dir)
        result = runner.run_experiment(cfg)

        assert result.status == "completed"
        assert result.total_runtime > 0.0

        # Verify all recorded components
        assert "data_loading" in result.runtime_breakdown
        assert "preprocessing" in result.runtime_breakdown
        assert "unet_inference" in result.runtime_breakdown
        assert "statistical_testing" in result.runtime_breakdown
        assert "fchmrf_inference" in result.runtime_breakdown

        # Verify evaluation metrics
        m = result.metrics
        assert "empirical_fdr" in m
        assert "tpr" in m
        assert "dice" in m
        assert "precision" in m
        assert "num_discoveries" in m

        # Verify output artifact locations exist on disk
        exp_dir = out_dir / "unit_exp_01"
        assert (exp_dir / "preprocessed.nii.gz").exists()
        assert (exp_dir / "roi_mask.nii.gz").exists()
        assert (exp_dir / "z_scores.nii.gz").exists()
        assert (exp_dir / "lis_map.nii.gz").exists()
        assert (exp_dir / "significant_mask.nii.gz").exists()
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "result.json").exists()
        assert (exp_dir / "metrics.csv").exists()
        assert (exp_dir / "diagnostic_summary.png").exists()

    def test_baseline_bh_experiment_execution(self, tmp_path: Path):
        out_dir = tmp_path / "run_bh"
        cfg = ExperimentConfig(
            experiment_id="bh_baseline_test",
            name="BH Baseline Test",
            seed=42,
            output_dir=str(out_dir),
            dataset=DatasetConfig(
                dataset_type="synthetic",
                grid_size=(20, 20, 10),
                snr=2.0,
                num_clusters=1,
                cluster_radius=3,
            ),
            fchmrf=FCHMRFConfig(method_mode="baseline_bh", alpha=0.05),
        )
        runner = ExperimentRunner(base_output_dir=out_dir)
        result = runner.run_experiment(cfg)
        assert result.status == "completed"
        assert result.metrics["num_discoveries"] >= 0


class TestSuiteAndTables:
    """Test batch matrix sweeps, comparison tables, and summary CSV."""

    def test_experiment_matrix_and_suite_run(self, tmp_path: Path):
        suite_dir = tmp_path / "suite_test"
        configs = build_experiment_matrix(
            snr_list=[2.0],
            methods=["baseline_bh", "proposed_fchmrf"],
            seeds=[42],
            grid_size=(20, 20, 10),
            base_output_dir=str(suite_dir),
        )
        assert len(configs) == 2

        runner = ExperimentRunner(base_output_dir=suite_dir)
        results = runner.run_suite(configs, suite_output_dir=suite_dir)

        assert len(results) == 2
        assert all(r.status == "completed" for r in results)

        # Verify summary CSV
        csv_file = suite_dir / "suite_summary.csv"
        assert csv_file.exists()
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3  # Header + 2 rows

        # Verify comparison tables
        md_file = suite_dir / "comparison_table.md"
        assert md_file.exists()
        tex_file = suite_dir / "comparison_table.tex"
        assert tex_file.exists()

        # Verify suite comparison plots
        assert (suite_dir / "suite_metrics_comparison_bar.png").exists()
        assert (suite_dir / "suite_runtime_breakdown_bar.png").exists()
        assert (suite_dir / "suite_fdr_power_calibration.png").exists()


class TestReproducibility:
    """Test exact bit-for-bit reproducibility from saved configuration."""

    def test_exact_reproducibility_from_config(self, tmp_path: Path):
        out_dir = tmp_path / "orig_exp"
        cfg = ExperimentConfig(
            experiment_id="repro_target",
            name="Reproducibility Target",
            seed=999,
            output_dir=str(out_dir),
            dataset=DatasetConfig(
                dataset_type="synthetic",
                grid_size=(20, 20, 10),
                snr=2.5,
                num_clusters=2,
                cluster_radius=3,
            ),
            fchmrf=FCHMRFConfig(
                method_mode="proposed_fchmrf",
                beta=0.7,
                lambda_dl=0.5,
                max_iterations=10,
            ),
        )

        runner = ExperimentRunner(base_output_dir=out_dir)
        orig_res = runner.run_experiment(cfg)
        assert orig_res.status == "completed"

        # Now reproduce directly from saved config.json
        saved_cfg_path = out_dir / "repro_target" / "config.json"
        assert saved_cfg_path.exists()

        repro_out = tmp_path / "repro_run"
        new_res, is_match, delta_report = reproduce_experiment(
            config_path=saved_cfg_path,
            output_dir=repro_out,
            tolerance=1e-5,
        )

        assert is_match is True
        assert new_res.status == "completed"
        assert new_res.metrics["num_discoveries"] == orig_res.metrics["num_discoveries"]
        assert np.isclose(new_res.metrics["empirical_fdr"], orig_res.metrics["empirical_fdr"], atol=1e-5)
        assert np.isclose(new_res.metrics["tpr"], orig_res.metrics["tpr"], atol=1e-5)
        assert np.isclose(new_res.metrics["dice"], orig_res.metrics["dice"], atol=1e-5)
