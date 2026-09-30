"""Phase 3: 3D U-Net Volumetric Segmentation and Spatial Prior Estimation.

Provides:
  - Configurable 3D U-Net architecture with depth, channels, norm, and activation customization.
  - Neuroimaging ROI Dataset and Synthetic Benchmark Dataset classes.
  - Subject-level dataset splitting to guarantee strict separation and prevent data leakage.
  - Mini-batch DataLoaders.
  - Training, validation, and evaluation loops with checkpoint saving and CPU/CUDA support.
  - Exact volumetric segmentation inference producing aligned NIfTI outputs.
  - Evaluation metrics: Dice score, IoU, precision, and recall.
  - Orthogonal multi-planar visualization of image, predicted prior, and ROI composite overlay.
"""

from src.deep_learning.models.unet3d import ConfigurableUNet3D
from src.deep_learning.models.base_model import BaseROIPriorModel
from src.deep_learning.dataset import (
    DatasetMode,
    NeuroimagingROIDataset,
    SyntheticROIDataset,
    create_dataloaders,
    split_dataset_by_subject,
)
from src.deep_learning.trainer import ROIPriorTrainer
from src.deep_learning.losses import (
    DiceLoss,
    FocalLoss,
    CombinedBceDiceLoss,
    FocalDiceLoss,
    get_segmentation_loss,
)
from src.deep_learning.metrics import (
    compute_segmentation_metrics,
    MetricTracker,
)
from src.deep_learning.inference import segment_volume
from src.deep_learning.visualization import (
    visualize_roi_segmentation,
    plot_training_curves,
)

__all__ = [
    "ConfigurableUNet3D",
    "BaseROIPriorModel",
    "DatasetMode",
    "NeuroimagingROIDataset",
    "SyntheticROIDataset",
    "create_dataloaders",
    "split_dataset_by_subject",
    "ROIPriorTrainer",
    "DiceLoss",
    "FocalLoss",
    "CombinedBceDiceLoss",
    "FocalDiceLoss",
    "get_segmentation_loss",
    "compute_segmentation_metrics",
    "MetricTracker",
    "segment_volume",
    "visualize_roi_segmentation",
    "plot_training_curves",
]
