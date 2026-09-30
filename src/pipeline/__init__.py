"""Master pipeline integration module.

Connects:
Neuroimaging Input
→ Preprocessing
→ 3D U-Net ROI Detection
→ ROI-Restricted Statistical Testing
→ Adaptive fcHMRF
→ LIS
→ Spatial FDR Control
"""

from src.pipeline.stages import PipelineStage, StageResult, StageStatus
from src.pipeline.master_pipeline import MasterPipeline, MasterPipelineConfig
from src.pipeline.reporting import generate_experiment_report

__all__ = [
    "PipelineStage",
    "StageResult",
    "StageStatus",
    "MasterPipeline",
    "MasterPipelineConfig",
    "generate_experiment_report",
]
