"""Pipeline stage definitions, status enums, and stage execution results."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PipelineStage(str, Enum):
    """The 6 core stages of the integrated neuroimaging research pipeline."""

    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    ROI_DETECTION = "roi_detection"
    STATISTICAL_TESTING = "statistical_testing"
    FCHMRF_LIS = "fchmrf_lis"
    ADAPTIVE_SELECTION = "adaptive_selection"


class StageStatus(str, Enum):
    """Execution lifecycle status of an individual pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageResult:
    """Detailed output and execution metadata for a single pipeline stage.

    Attributes:
        stage: PipelineStage identifier.
        status: StageStatus (COMPLETED, SKIPPED, FAILED).
        output_files: Dictionary mapping artifact names to their saved Path.
        metrics: Dictionary of numerical summary metrics produced by the stage.
        elapsed_seconds: Wall-clock execution time.
        error_message: Error string if status == FAILED.
        recomputed: True if stage was executed, False if loaded from cache.
    """

    stage: PipelineStage
    status: StageStatus
    output_files: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    recomputed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "output_files": {k: str(v) for k, v in self.output_files.items()},
            "metrics": self.metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "error_message": self.error_message,
            "recomputed": self.recomputed,
        }
