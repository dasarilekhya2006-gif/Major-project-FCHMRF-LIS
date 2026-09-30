"""Unit and integration test suite for Phase 16: Final Research Report Generator."""

from pathlib import Path
import tempfile
import pytest

from src.reporting.schemas import (
    ReportFigure,
    ReportSectionID,
    ReportTable,
    ResearchReportData,
    SystemEnvironmentInfo,
)
from src.reporting.data_collector import (
    ResearchDataCollector,
    get_system_environment_info,
)
from src.reporting.generator import FinalResearchReportGenerator


class TestReportSchemasAndEnvironment:
    """Verify metadata collection and schema validation."""

    def test_system_environment_collection(self) -> None:
        env = get_system_environment_info()
        assert isinstance(env, SystemEnvironmentInfo)
        assert len(env.os_platform) > 0
        assert len(env.python_version) > 0
        assert len(env.numpy_version) > 0
        assert len(env.torch_version) > 0
        assert env.cpu_cores > 0
        assert env.ram_gb > 0.0

    def test_report_section_ids(self) -> None:
        # Verify all 16 required sections are represented
        required_sections = [
            ReportSectionID.TITLE,
            ReportSectionID.RESEARCH_OBJECTIVE,
            ReportSectionID.DATASET,
            ReportSectionID.PREPROCESSING,
            ReportSectionID.UNET_METHODOLOGY,
            ReportSectionID.STATISTICAL_TESTING,
            ReportSectionID.FCHMRF_LIS_METHODOLOGY,
            ReportSectionID.ADAPTIVE_OPTIMIZATION,
            ReportSectionID.EXPERIMENTAL_SETUP,
            ReportSectionID.BASELINES,
            ReportSectionID.RESULTS,
            ReportSectionID.ABLATION_STUDY,
            ReportSectionID.ROBUSTNESS_ANALYSIS,
            ReportSectionID.LIMITATIONS,
            ReportSectionID.CONCLUSION,
            ReportSectionID.FUTURE_WORK,
        ]
        assert len(required_sections) == 16


class TestFinalResearchReportGenerator:
    """Verify document compilation across Markdown, LaTeX, HTML, and CSV."""

    @pytest.fixture
    def sample_report_data(self, tmp_path: Path) -> ResearchReportData:
        env = get_system_environment_info()
        data = ResearchReportData(
            project_title="ROI-Assisted Adaptive fcHMRF-LIS Neuroimaging Benchmark",
            target_alpha=0.05,
            environment=env,
        )

        # Mock table 1
        t1_headers = ["Method", "FDR", "Power", "Dice"]
        t1_rows = [
            ["BH-FDR", "0.0421", "0.7120", "0.7410"],
            ["Proposed", "0.0385", "0.8950", "0.9120"],
        ]
        data.tables["table1_baseline_comparison"] = ReportTable(
            table_id="table1_baseline_comparison",
            title="Baseline Comparison",
            headers=t1_headers,
            rows=t1_rows,
            markdown_str="| Method | FDR | Power | Dice |\n| :--- | :--- | :--- | :--- |\n| BH-FDR | 0.0421 | 0.7120 | 0.7410 |\n| Proposed | 0.0385 | 0.8950 | 0.9120 |",
            latex_str="\\begin{table*}...\n\\end{table*}",
        )

        # Mock figure 1
        dummy_fig = tmp_path / "fig1.png"
        dummy_fig.write_text("fake image content")
        data.figures["fig1_comparative_summary"] = ReportFigure(
            figure_id="fig1_comparative_summary",
            title="Comparative Evaluation",
            file_path=dummy_fig,
            caption="Sample comparative figure caption.",
        )
        return data

    def test_markdown_rendering(self, sample_report_data: ResearchReportData, tmp_path: Path) -> None:
        generator = FinalResearchReportGenerator(data=sample_report_data, output_dir=tmp_path)
        md = generator.render_markdown()

        # Check all 16 required headings
        headings = [
            "1. Project Title",
            "2. Research Objective",
            "3. Dataset Description",
            "4. Preprocessing Pipeline",
            "5. 3D U-Net Deep Learning Prior Methodology",
            "6. Voxel-Level Statistical Testing",
            "7. fcHMRF-LIS Spatial Statistical Inference Methodology",
            "8. Adaptive Parameter Optimization",
            "9. Experimental Setup & Protocol",
            "10. Baseline Methods",
            "11. Experimentally Observed Results",
            "12. Systematic Ablation Study & Component Attribution",
            "13. Robustness & Uncertainty Analysis",
            "14. Limitations & Scientific Scope",
            "15. Conclusion",
            "16. Future Work",
        ]
        for heading in headings:
            assert heading in md, f"Missing heading: {heading}"

        # Verify distinctions
        assert "Experimentally Observed Results" in md
        assert "Theoretical & Methodological Assumptions" in md
        assert "Scientific Limitations" in md
        assert "Future Work" in md

    def test_generate_all_outputs(self, sample_report_data: ResearchReportData, tmp_path: Path) -> None:
        generator = FinalResearchReportGenerator(data=sample_report_data, output_dir=tmp_path)
        paths = generator.generate_all()

        assert paths["markdown"].exists()
        assert paths["latex"].exists()
        assert paths["html"].exists()

        # Check CSV export
        csv_file = tmp_path / "tables" / "table1_baseline_comparison.csv"
        assert csv_file.exists()
        content = csv_file.read_text(encoding="utf-8")
        assert "BH-FDR" in content
        assert "Proposed" in content


class TestEndToEndReportGeneration:
    """Fast integration test running real data collector and report generator."""

    def test_fast_collector_and_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            collector = ResearchDataCollector(
                output_dir=out_dir,
                nominal_alpha=0.05,
                device="cpu",
            )
            # Run quick execution with 1 small grid and 2 seeds
            report_data = collector.collect_or_run_evaluations(
                snr_levels=[2.5],
                seeds=[42, 101],
                grid_size=(16, 16, 8),
                run_benchmarks=True,
            )

            assert "table1_baseline_comparison" in report_data.tables
            assert "table2_ablation_summary" in report_data.tables
            assert "table3_component_attribution" in report_data.tables
            assert "table4_statistical_tests" in report_data.tables
            assert "table5_robustness_summary" in report_data.tables

            assert "fig1_comparative_summary" in report_data.figures
            assert "fig2_ablation_attribution" in report_data.figures
            assert "fig3_robustness_uncertainty" in report_data.figures

            generator = FinalResearchReportGenerator(
                data=report_data,
                output_dir=out_dir,
            )
            paths = generator.generate_all()

            for key, path in paths.items():
                assert path.exists(), f"Path {key} does not exist: {path}"
                assert path.stat().st_size > 500, f"Path {key} is empty: {path}"
