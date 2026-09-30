"""Comprehensive unit and integration test suite for Phase 3: 3D U-Net ROI Detection."""

from pathlib import Path
import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.exceptions import DataValidationError, DeepLearningModelError
from src.data.schemas import NeuroimagingVolume
from src.data.sample_data import create_sample_3d_statistical_nifti
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
    DatasetMode,
    NeuroimagingROIDataset,
    SyntheticROIDataset,
    create_dataloaders,
)
from src.deep_learning.trainer import ROIPriorTrainer
from src.deep_learning.inference import segment_volume
from src.deep_learning.visualization import visualize_roi_segmentation, plot_training_curves


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for deep learning tests")
class TestConfigurableUNet3D:
    """Tests for the configurable 3D U-Net architecture."""

    def test_default_architecture_forward_pass(self):
        """Test default 3D U-Net with depth=2 on small volume."""
        model = ConfigurableUNet3D(
            in_channels=1,
            out_classes=1,
            base_channels=8,
            depth=2,
            activation="relu",
            norm_type="batch_norm",
            dropout_rate=0.0,
        )
        x = torch.randn(2, 1, 16, 16, 16)
        probs = model(x, return_logits=False)
        assert probs.shape == (2, 1, 16, 16, 16)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        logits = model(x, return_logits=True)
        assert logits.shape == (2, 1, 16, 16, 16)

    def test_non_divisible_spatial_dimensions_handling(self):
        """Ensure U-Net handles odd/non-divisible spatial dimensions via internal replicate padding & cropping."""
        model = ConfigurableUNet3D(
            in_channels=1,
            out_classes=1,
            base_channels=4,
            depth=3,  # Requires divisibility by 8
            activation="leaky_relu",
            norm_type="instance_norm",
        )
        # 15, 17, 19 are not divisible by 8
        x = torch.randn(1, 1, 15, 17, 19)
        out = model(x)
        assert out.shape == (1, 1, 15, 17, 19)

    @pytest.mark.parametrize("activation", ["relu", "leaky_relu", "elu", "gelu"])
    def test_configurable_activations(self, activation):
        """Verify model instantiates and executes with different activation functions."""
        model = ConfigurableUNet3D(
            in_channels=1,
            out_classes=1,
            base_channels=4,
            depth=1,
            activation=activation,
            norm_type="none",
        )
        x = torch.randn(1, 1, 8, 8, 8)
        out = model(x)
        assert out.shape == (1, 1, 8, 8, 8)

    @pytest.mark.parametrize("norm_type", ["batch_norm", "instance_norm", "group_norm", "none"])
    def test_configurable_normalization_layers(self, norm_type):
        """Verify model works with batch, instance, group, and identity normalization."""
        model = ConfigurableUNet3D(
            in_channels=1,
            out_classes=1,
            base_channels=4,
            depth=1,
            activation="relu",
            norm_type=norm_type,
        )
        x = torch.randn(2, 1, 8, 8, 8)
        out = model(x)
        assert out.shape == (2, 1, 8, 8, 8)

    def test_predict_prior_map_numpy(self):
        """Verify predict_prior_map helper converts numpy input to numpy probability map."""
        model = ConfigurableUNet3D(in_channels=1, out_classes=1, base_channels=4, depth=1)
        arr = np.random.randn(16, 16, 16).astype(np.float32)
        prior = model.predict_prior_map(arr)
        assert isinstance(prior, np.ndarray)
        assert prior.shape == (16, 16, 16)
        assert (prior >= 0.0).all() and (prior <= 1.0).all()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for loss tests")
class TestSegmentationLosses:
    """Tests for volumetric segmentation loss functions."""

    def test_dice_loss(self):
        """Test DiceLoss behavior for perfect overlap vs zero overlap."""
        loss_fn = DiceLoss(from_logits=False)
        targets = torch.ones(1, 1, 8, 8, 8)
        perfect_preds = torch.ones(1, 1, 8, 8, 8)
        zero_preds = torch.zeros(1, 1, 8, 8, 8)

        loss_perfect = loss_fn(perfect_preds, targets)
        loss_zero = loss_fn(zero_preds, targets)

        assert float(loss_perfect.item()) < 0.01
        assert float(loss_zero.item()) > 0.95

    def test_focal_loss(self):
        """Test FocalLoss returns positive numerical loss."""
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0, from_logits=True)
        logits = torch.randn(2, 1, 8, 8, 8)
        targets = (torch.rand(2, 1, 8, 8, 8) > 0.8).float()

        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0

    def test_combined_bce_dice_loss(self):
        """Test CombinedBceDiceLoss weighted combination."""
        loss_fn = CombinedBceDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=True)
        logits = torch.randn(1, 1, 8, 8, 8)
        targets = torch.zeros(1, 1, 8, 8, 8)
        targets[0, 0, 2:4, 2:4, 2:4] = 1.0

        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0

    def test_get_segmentation_loss_factory(self):
        """Verify loss factory correctly instantiates classes."""
        assert isinstance(get_segmentation_loss("dice"), DiceLoss)
        assert isinstance(get_segmentation_loss("focal"), FocalLoss)
        assert isinstance(get_segmentation_loss("combined"), CombinedBceDiceLoss)
        assert isinstance(get_segmentation_loss("focal_dice"), FocalDiceLoss)

        with pytest.raises(DeepLearningModelError):
            get_segmentation_loss("non_existent_loss")


class TestSegmentationMetrics:
    """Tests for Dice, IoU, Precision, and Recall evaluation metrics."""

    def test_perfect_agreement(self):
        """All metrics should be 1.0 for perfect overlap."""
        mask = np.zeros((10, 10, 10))
        mask[2:6, 2:6, 2:6] = 1.0

        metrics = compute_segmentation_metrics(mask, mask)
        assert metrics["dice"] == pytest.approx(1.0, rel=1e-4)
        assert metrics["iou"] == pytest.approx(1.0, rel=1e-4)
        assert metrics["precision"] == pytest.approx(1.0, rel=1e-4)
        assert metrics["recall"] == pytest.approx(1.0, rel=1e-4)

    def test_empty_masks_corner_case(self):
        """Both predictions and targets empty should yield 1.0 (perfect null agreement)."""
        zeros = np.zeros((10, 10, 10))
        metrics = compute_segmentation_metrics(zeros, zeros)
        assert metrics["dice"] == 1.0
        assert metrics["iou"] == 1.0

    def test_partial_overlap_math(self):
        """Verify exact theoretical metric values for known partial overlap."""
        pred = np.zeros((10, 10, 10))
        target = np.zeros((10, 10, 10))

        # 4 true positives, 2 false positives, 4 false negatives
        pred[:6, :1, :1] = 1.0    # 6 predicted
        target[2:10, :1, :1] = 1.0 # 8 targets
        # Intersection: indices 2, 3, 4, 5 (4 voxels)
        # TP = 4, FP = 2, FN = 4

        metrics = compute_segmentation_metrics(pred, target)
        expected_dice = (2 * 4) / (2 * 4 + 2 + 4)  # 8 / 14 = 4/7 ~ 0.5714
        expected_iou = 4 / (4 + 2 + 4)              # 4 / 10 = 0.4
        expected_prec = 4 / (4 + 2)                 # 4 / 6 ~ 0.6667
        expected_rec = 4 / (4 + 4)                  # 4 / 8 = 0.5

        assert metrics["dice"] == pytest.approx(expected_dice, rel=1e-3)
        assert metrics["iou"] == pytest.approx(expected_iou, rel=1e-3)
        assert metrics["precision"] == pytest.approx(expected_prec, rel=1e-3)
        assert metrics["recall"] == pytest.approx(expected_rec, rel=1e-3)

    def test_metric_tracker(self):
        """Test accumulation across batches."""
        tracker = MetricTracker()
        p1 = np.ones((5, 5, 5))
        t1 = np.ones((5, 5, 5))
        tracker.update(loss=0.2, predictions=p1, targets=t1)

        avg = tracker.compute_averages()
        assert avg["loss"] == pytest.approx(0.2)
        assert avg["dice"] == pytest.approx(1.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for dataset tests")
class TestDatasetsAndDataLoaders:
    """Tests for NeuroimagingROIDataset and SyntheticROIDataset."""

    def test_supervised_mode_missing_masks_error(self, tmp_path):
        """Supervised mode without masks must raise DataValidationError (prevents fabricated labels)."""
        img_path = tmp_path / "img.nii.gz"
        create_sample_3d_statistical_nifti(img_path, shape=(16, 16, 8))

        with pytest.raises(DataValidationError, match="Supervised mode requires genuine ground-truth"):
            NeuroimagingROIDataset(image_paths=[img_path], mask_paths=None, mode=DatasetMode.SUPERVISED)

    def test_synthetic_benchmark_dataset(self):
        """Verify synthetic benchmark dataset yields paired tensors."""
        dataset = SyntheticROIDataset(num_samples=4, grid_size=(16, 16, 8), snr=2.5, seed=42)
        assert len(dataset) == 4

        sample = dataset[0]
        assert "image" in sample and "mask" in sample
        assert sample["image"].shape == (1, 16, 16, 8)
        assert sample["mask"].shape == (1, 16, 16, 8)

    def test_create_dataloaders(self):
        """Verify DataLoader creation and mini-batch loading."""
        train_ds = SyntheticROIDataset(num_samples=4, grid_size=(16, 16, 8), seed=1)
        val_ds = SyntheticROIDataset(num_samples=2, grid_size=(16, 16, 8), seed=2)

        loaders = create_dataloaders(train_dataset=train_ds, val_dataset=val_ds, batch_size=2)
        assert "train" in loaders and "val" in loaders

        batch = next(iter(loaders["train"]))
        assert batch["image"].shape == (2, 1, 16, 16, 8)
        assert batch["mask"].shape == (2, 1, 16, 16, 8)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for training tests")
class TestTrainingPipeline:
    """Tests for ROIPriorTrainer fitting, evaluation, and checkpointing."""

    def test_train_epoch_and_fit(self, tmp_path):
        """Run mini training loop and verify loss tracking and checkpoint saving."""
        train_ds = SyntheticROIDataset(num_samples=4, grid_size=(16, 16, 8), seed=10)
        val_ds = SyntheticROIDataset(num_samples=2, grid_size=(16, 16, 8), seed=20)
        loaders = create_dataloaders(train_ds, val_ds, batch_size=2)

        model = ConfigurableUNet3D(in_channels=1, out_classes=1, base_channels=4, depth=1)
        trainer = ROIPriorTrainer(
            model=model,
            checkpoint_dir=tmp_path / "checkpoints",
            device="cpu",
            learning_rate=1e-3,
        )

        history = trainer.fit(loaders["train"], loaders["val"], epochs=2, save_best=True)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert len(history["val_dice"]) == 2

        # Verify best checkpoint was created
        best_ckpt = tmp_path / "checkpoints" / "unet3d_best_checkpoint.pt"
        assert best_ckpt.exists()

        # Test reload
        new_model = ConfigurableUNet3D(in_channels=1, out_classes=1, base_channels=4, depth=1)
        new_trainer = ROIPriorTrainer(model=new_model, device="cpu")
        loaded_state = new_trainer.load_checkpoint(best_ckpt)
        assert "model_state_dict" in loaded_state


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for inference tests")
class TestVolumetricInferenceAndCoordinates:
    """Tests for segment_volume and spatial coordinate preservation."""

    def test_segment_volume_preserves_shape_and_affine(self, tmp_path):
        """Verify segmentation output matches exact input spatial shape and affine matrix."""
        vol_path = tmp_path / "test_input.nii.gz"
        custom_affine = np.array([
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -120.0],
            [0.0, 0.0, 2.5, -60.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        create_sample_3d_statistical_nifti(vol_path, shape=(20, 22, 14), affine=custom_affine)

        model = ConfigurableUNet3D(in_channels=1, out_classes=1, base_channels=4, depth=2)
        out_dir = tmp_path / "outputs"

        result = segment_volume(
            vol_path,
            model=model,
            device="cpu",
            output_dir=out_dir,
            prefix="test_sub",
        )

        # Coordinate and shape assertions
        assert result["probability_map"].shape == (20, 22, 14)
        assert result["binary_mask"].shape == (20, 22, 14)
        assert np.allclose(result["affine"], custom_affine)

        # File output assertions
        assert result["saved_paths"]["probability"].exists()
        assert result["saved_paths"]["binary_mask"].exists()


class TestVisualizations:
    """Tests for ROI visual inspection and training curves rendering."""

    def test_visualize_roi_segmentation(self, tmp_path):
        """Verify generation of multi-panel orthogonal slice figure."""
        vol = np.random.randn(20, 20, 16).astype(np.float32)
        mask = np.zeros((20, 20, 16), dtype=np.float32)
        mask[8:12, 8:12, 6:10] = 1.0

        out_png = tmp_path / "roi_viz.png"
        res_path = visualize_roi_segmentation(
            original_volume=vol,
            predicted_mask=mask,
            output_path=out_png,
            dpi=100,
        )
        assert res_path.exists()
        assert res_path.stat().st_size > 1000

    def test_plot_training_curves(self, tmp_path):
        """Verify generation of dual-panel training curves."""
        history = {
            "train_loss": [0.8, 0.6, 0.4, 0.3],
            "val_loss": [0.85, 0.65, 0.42, 0.35],
            "val_dice": [0.2, 0.5, 0.75, 0.82],
            "val_iou": [0.15, 0.4, 0.65, 0.72],
            "val_precision": [0.3, 0.6, 0.8, 0.85],
            "val_recall": [0.25, 0.55, 0.78, 0.80],
        }
        out_png = tmp_path / "curves.png"
        res_path = plot_training_curves(history, output_path=out_png, dpi=100)
        assert res_path.exists()
        assert res_path.stat().st_size > 1000
