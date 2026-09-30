"""Deep learning module for 3D U-Net neuroimaging ROI detection and spatial prior estimation."""

from src.deep_learning.models.base_model import BaseROIPriorModel
from src.deep_learning.models.unet_roi import LightweightUNet3D
from src.deep_learning.models.unet3d import ConfigurableUNet3D
from src.deep_learning.losses import (
    DiceLoss,
    FocalLoss,
    CombinedBceDiceLoss,
    FocalDiceLoss,
    get_segmentation_loss,
)
from src.deep_learning.metrics import compute_segmentation_metrics, MetricTracker
from src.deep_learning.dataset import (
    NeuroimagingROIDataset,
    SyntheticROIDataset,
    DatasetMode,
    create_dataloaders,
)
from src.deep_learning.trainer import ROIPriorTrainer
from src.deep_learning.predictor import ROIPriorPredictor
from src.deep_learning.inference import segment_volume
from src.deep_learning.visualization import visualize_roi_segmentation, plot_training_curves

__all__ = [
    "BaseROIPriorModel",
    "LightweightUNet3D",
    "ConfigurableUNet3D",
    "DiceLoss",
    "FocalLoss",
    "CombinedBceDiceLoss",
    "FocalDiceLoss",
    "get_segmentation_loss",
    "compute_segmentation_metrics",
    "MetricTracker",
    "DatasetMode",
    "NeuroimagingROIDataset",
    "SyntheticROIDataset",
    "create_dataloaders",
    "ROIPriorTrainer",
    "ROIPriorPredictor",
    "segment_volume",
    "visualize_roi_segmentation",
    "plot_training_curves",
]
