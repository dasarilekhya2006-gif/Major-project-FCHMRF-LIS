"""Data models and schemas for the Final Research Report Generator."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ReportSectionID(str, Enum):
    """The 16 canonical report sections specified by the research framework."""

    TITLE = "project_title"
    RESEARCH_OBJECTIVE = "research_objective"
    DATASET = "dataset"
    PREPROCESSING = "preprocessing"
    UNET_METHODOLOGY = "unet_methodology"
    STATISTICAL_TESTING = "statistical_testing"
    FCHMRF_LIS_METHODOLOGY = "fchmrf_lis_methodology"
    ADAPTIVE_OPTIMIZATION = "adaptive_optimization"
    EXPERIMENTAL_SETUP = "experimental_setup"
    BASELINES = "baselines"
    RESULTS = "results"
    ABLATION_STUDY = "ablation_study"
    ROBUSTNESS_ANALYSIS = "robustness_analysis"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"
    FUTURE_WORK = "future_work"


@dataclass
class ReportFigure:
    """Figure asset embedded in the research report.

    Attributes:
        figure_id: Unique identifier (e.g. 'fig1_comparative_summary').
        title: Descriptive title.
        file_path: Relative or absolute path to the image file.
        caption: Detailed publication-grade scientific caption.
        panel_labels: Descriptions of sub-panels (e.g. A, B, C, D).
    """

    figure_id: str
    title: str
    file_path: Path
    caption: str
    panel_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReportTable:
    """Table asset embedded in the research report.

    Attributes:
        table_id: Unique identifier (e.g. 'table1_baseline_comparison').
        title: Table caption/title.
        headers: Column header strings.
        rows: Matrix of row string values.
        markdown_str: Rendered Markdown representation.
        latex_str: Rendered LaTeX representation.
        notes: Explanatory footnotes (e.g. significance markers, abbreviations).
    """

    table_id: str
    title: str
    headers: List[str]
    rows: List[List[str]]
    markdown_str: str = ""
    latex_str: str = ""
    notes: Optional[str] = None


@dataclass
class SystemEnvironmentInfo:
    """Hardware and software environment metadata for reproducibility."""

    os_platform: str
    python_version: str
    numpy_version: str
    scipy_version: str
    torch_version: str
    device_name: str
    cpu_cores: int
    ram_gb: float
    timestamp_iso: str


@dataclass
class ResearchReportData:
    """Container for all empirical data, tables, figures, and metadata."""

    project_title: str
    target_alpha: float
    environment: SystemEnvironmentInfo
    tables: Dict[str, ReportTable] = field(default_factory=dict)
    figures: Dict[str, ReportFigure] = field(default_factory=dict)
    empirical_metrics: Dict[str, Any] = field(default_factory=dict)
    ablation_metrics: Dict[str, Any] = field(default_factory=dict)
    robustness_metrics: Dict[str, Any] = field(default_factory=dict)
