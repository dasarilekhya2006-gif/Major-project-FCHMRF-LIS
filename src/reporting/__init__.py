"""Automated Research Report Generator module (Phase 16)."""

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

__all__ = [
    "ReportSectionID",
    "ReportFigure",
    "ReportTable",
    "SystemEnvironmentInfo",
    "ResearchReportData",
    "ResearchDataCollector",
    "get_system_environment_info",
    "FinalResearchReportGenerator",
]
