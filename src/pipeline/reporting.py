"""Experiment reporting engine: collects execution metadata and produces comprehensive reports."""

from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.core.logging import get_logger
from src.pipeline.stages import PipelineStage, StageResult, StageStatus

logger = get_logger(__name__)


def collect_system_environment() -> Dict[str, Any]:
    """Capture runtime system environment, hardware accelerators, and library versions."""
    env = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown CPU",
        "architecture": platform.architecture()[0],
        "numpy_version": np.__version__,
    }

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
        else:
            env["gpu_name"] = "CPU (No CUDA device detected)"
    except ImportError:
        env["torch_version"] = "Not installed"
        env["cuda_available"] = False
        env["gpu_name"] = "None"

    try:
        import nibabel as nib
        env["nibabel_version"] = nib.__version__
    except ImportError:
        env["nibabel_version"] = "Not installed"

    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except ImportError:
        env["scipy_version"] = "Not installed"

    return env


def generate_experiment_report(
    experiment_id: str,
    stage_results: Dict[PipelineStage, StageResult],
    pipeline_config: Dict[str, Any],
    output_dir: Union[str, Path],
    start_time: str,
    end_time: str,
    total_duration_seconds: float,
) -> Dict[str, Path]:
    """Compile master experiment report in JSON and Markdown formats.

    Args:
        experiment_id: Unique experiment run identifier.
        stage_results: Mapping of PipelineStage to StageResult.
        pipeline_config: Pipeline configuration dictionary.
        output_dir: Destination directory.
        start_time: ISO timestamp.
        end_time: ISO timestamp.
        total_duration_seconds: Wall-clock duration.

    Returns:
        Dict[str, Path] with paths to generated 'json_report' and 'markdown_report'.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    env_info = collect_system_environment()

    # 1. Assemble structured dictionary
    report_data = {
        "experiment_id": experiment_id,
        "start_time": start_time,
        "end_time": end_time,
        "total_duration_seconds": total_duration_seconds,
        "system_environment": env_info,
        "pipeline_config": pipeline_config,
        "stages": {stage.value: res.to_dict() for stage, res in stage_results.items()},
    }

    # 2. Save JSON Report
    json_path = out_p / f"{experiment_id}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    paths["json_report"] = json_path

    # 3. Generate Human-Readable Markdown Report
    md_path = out_p / f"{experiment_id}_report.md"
    md_lines: List[str] = [
        f"# Research Experiment Report: {experiment_id}",
        "",
        f"**Date**: {start_time}  ",
        f"**Duration**: {total_duration_seconds:.2f} seconds  ",
        f"**Host Hardware**: {env_info.get('processor')} | **Accelerator**: {env_info.get('gpu_name')}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Pipeline Stage | Status | Duration | Cached / Recomputed | Key Metrics |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for stage, res in stage_results.items():
        recomp_str = "Recomputed" if res.recomputed else "Cached"
        metrics_summary = []
        for k, v in res.metrics.items():
            if isinstance(v, float):
                metrics_summary.append(f"{k}: {v:.4f}")
            elif isinstance(v, (int, str)):
                metrics_summary.append(f"{k}: {v}")
        m_str = ", ".join(metrics_summary[:3]) if metrics_summary else "N/A"
        md_lines.append(
            f"| **{stage.value}** | `{res.status.value.upper()}` | {res.elapsed_seconds:.2f}s | {recomp_str} | {m_str} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Stage Execution Details & Artifacts",
        "",
    ])

    for stage, res in stage_results.items():
        md_lines.append(f"### Stage: {stage.value.replace('_', ' ').title()}")
        md_lines.append(f"- **Status**: `{res.status.value}`")
        md_lines.append(f"- **Execution Time**: {res.elapsed_seconds:.3f} s")
        if res.error_message:
            md_lines.append(f"- **Error**: `{res.error_message}`")

        if res.metrics:
            md_lines.append("- **Detailed Metrics**:")
            for k, v in res.metrics.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                md_lines.append(f"  - `{k}`: {val_str}")

        if res.output_files:
            md_lines.append("- **Generated Artifacts**:")
            for name, path in res.output_files.items():
                file_url = f"file:///{str(path).replace('\\', '/')}"
                md_lines.append(f"  - [{name}]({file_url})")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 3. Computational Environment",
        "",
        f"- **Python**: `{env_info.get('python_version')}`",
        f"- **Platform**: `{env_info.get('platform')}`",
        f"- **PyTorch**: `{env_info.get('torch_version')}` (CUDA Available: `{env_info.get('cuda_available')}`)",
        f"- **Accelerator**: `{env_info.get('gpu_name')}`",
        f"- **NiBabel**: `{env_info.get('nibabel_version')}`",
        f"- **NumPy**: `{env_info.get('numpy_version')}`",
        f"- **SciPy**: `{env_info.get('scipy_version')}`",
        "",
        "---",
        "*Report auto-generated by Adaptive Deep Learning-Assisted fcHMRF-LIS Research Framework.*",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    paths["markdown_report"] = md_path

    logger.info(f"Generated comprehensive experiment reports: {json_path} and {md_path}")
    return paths
