"""Phase 3 Verification: 3D U-Net Volumetric ROI Segmentation.

Verifies:
  - Configurable 3D U-Net architecture.
  - Neuroimaging ROI Dataset and Synthetic Benchmark Dataset classes.
  - Strict subject-level dataset splitting with zero data leakage.
  - Mini-batch DataLoaders.
  - Training and validation loop execution on CPU/CUDA with checkpoint saving.
  - Volumetric segmentation inference with NIfTI output saving.
  - Preservation of input affine matrix and spatial dimensions.
  - Calculation of segmentation evaluation metrics: Dice, IoU, Precision, and Recall.
  - Multi-planar orthogonal visualization of image, predicted prior, and ROI composite overlay.
"""

from pathlib import Path
import numpy as np
import pytest
import nibabel as nib
import torch

from segmentation import (
    ConfigurableUNet3D,
    SyntheticROIDataset,
    create_dataloaders,
    split_dataset_by_subject,
    ROIPriorTrainer,
    compute_segmentation_metrics,
    segment_volume,
    visualize_roi_segmentation,
)
from data.loaders import load_nifti


def test_configurable_unet3d_architecture():
    """Verify 3D U-Net instantiates with custom depth, normalization, and activations."""
    model = ConfigurableUNet3D(
        in_channels=1,
        out_classes=1,
        base_channels=4,
        depth=2,
        activation="leaky_relu",
        norm_type="instance_norm",
    )
    assert model.depth == 2
    assert model.in_channels == 1
    assert model.out_classes == 1

    x = torch.randn(1, 1, 16, 16, 16)
    out_probs = model(x, return_logits=False)
    assert out_probs.shape == (1, 1, 16, 16, 16)
    assert (out_probs >= 0.0).all() and (out_probs <= 1.0).all()

    out_logits = model(x, return_logits=True)
    assert out_logits.shape == (1, 1, 16, 16, 16)


def test_split_dataset_by_subject_prevents_leakage():
    """Verify subject-level splitting guarantees disjoint subject sets across splits."""
    # 10 subjects with multiple scans per subject
    subject_ids = [
        "sub-01", "sub-01",
        "sub-02", "sub-02", "sub-02",
        "sub-03",
        "sub-04", "sub-04",
        "sub-05",
        "sub-06", "sub-06",
        "sub-07",
        "sub-08", "sub-08",
        "sub-09",
        "sub-10", "sub-10",
    ]

    splits = split_dataset_by_subject(
        subject_ids=subject_ids,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=123,
    )

    train_subjs = set(subject_ids[idx] for idx in splits["train"])
    val_subjs = set(subject_ids[idx] for idx in splits["val"])
    test_subjs = set(subject_ids[idx] for idx in splits["test"])

    # Strict disjointness - zero subject-level data leakage
    assert train_subjs.isdisjoint(val_subjs), "Data leakage: train and val share subjects!"
    assert train_subjs.isdisjoint(test_subjs), "Data leakage: train and test share subjects!"
    assert val_subjs.isdisjoint(test_subjs), "Data leakage: val and test share subjects!"

    # Completeness
    all_assigned_indices = splits["train"] + splits["val"] + splits["test"]
    assert len(all_assigned_indices) == len(subject_ids)
    assert set(all_assigned_indices) == set(range(len(subject_ids)))


def test_segmentation_metrics_calculation():
    """Verify mathematical correctness of Dice, IoU, Precision, and Recall."""
    # Synthetic target with 20 active voxels
    target = np.zeros((10, 10, 10), dtype=np.float32)
    target[:5, :4, :1] = 1.0  # 20 voxels

    # Prediction with 15 true positives, 5 false positives, 5 false negatives
    pred = np.zeros((10, 10, 10), dtype=np.float32)
    pred[:5, :3, :1] = 1.0   # 15 true positives
    pred[5:10, :1, :1] = 1.0  # 5 false positives

    metrics = compute_segmentation_metrics(pred, target, threshold=0.5)

    # TP = 15, FP = 5, FN = 5
    # Dice = 2*15 / (2*15 + 5 + 5) = 30 / 40 = 0.75
    # IoU = 15 / (15 + 5 + 5) = 15 / 25 = 0.60
    # Precision = 15 / (15 + 5) = 0.75
    # Recall = 15 / (15 + 5) = 0.75
    assert np.isclose(metrics["dice"], 0.75, atol=1e-3)
    assert np.isclose(metrics["iou"], 0.60, atol=1e-3)
    assert np.isclose(metrics["precision"], 0.75, atol=1e-3)
    assert np.isclose(metrics["recall"], 0.75, atol=1e-3)


def test_end_to_end_training_and_inference(tmp_path: Path):
    """Run an end-to-end training, validation, checkpoint saving, and inference test."""
    # 1. Dataset & DataLoaders using synthetic benchmark instances (Option C)
    grid_size = (16, 16, 8)
    train_ds = SyntheticROIDataset(num_samples=4, grid_size=grid_size, snr=3.0, seed=42)
    val_ds = SyntheticROIDataset(num_samples=2, grid_size=grid_size, snr=3.0, seed=84)
    loaders = create_dataloaders(train_ds, val_ds, batch_size=2)

    # 2. Configurable 3D U-Net Model & Trainer (supporting CPU)
    model = ConfigurableUNet3D(in_channels=1, out_classes=1, base_channels=4, depth=1)
    ckpt_dir = tmp_path / "checkpoints"
    trainer = ROIPriorTrainer(
        model=model,
        checkpoint_dir=ckpt_dir,
        device="cpu",
        learning_rate=2e-3,
        seed=42,
    )

    # 3. Training & Validation Loop
    history = trainer.fit(loaders["train"], loaders["val"], epochs=2, save_best=True)
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_dice"]) == 2
    assert ckpt_dir.joinpath("unet3d_best_checkpoint.pt").exists()

    # 4. Volumetric Inference with NIfTI Output
    test_vol_path = tmp_path / "test_input_volume.nii.gz"
    test_data = np.random.RandomState(99).randn(16, 16, 8).astype(np.float32)
    test_affine = np.diag([2.0, 2.0, 2.5, 1.0])
    test_affine[0, 3] = -20.0
    test_affine[1, 3] = 15.0
    test_affine[2, 3] = -10.0
    nib.save(nib.Nifti1Image(test_data, test_affine), str(test_vol_path))

    out_dir = tmp_path / "segmentation_outputs"
    seg_result = segment_volume(
        test_vol_path,
        model=model,
        device="cpu",
        threshold=0.5,
        output_dir=out_dir,
        prefix="subj_test",
        canonical_reorient=False,
    )

    # Validate spatial preservation and NIfTI output
    assert seg_result["probability_map"].shape == (16, 16, 8)
    assert seg_result["binary_mask"].shape == (16, 16, 8)
    assert np.allclose(seg_result["affine"], test_affine)
    assert "saved_paths" in seg_result
    prob_nii = Path(seg_result["saved_paths"]["probability_map"])
    mask_nii = Path(seg_result["saved_paths"]["binary_mask"])
    assert prob_nii.exists() and mask_nii.exists()

    # Verify reloaded NIfTI matches
    loaded_mask = load_nifti(mask_nii, canonical_reorient=False, validate=False)
    assert loaded_mask.spatial_shape == (16, 16, 8)
    assert np.allclose(loaded_mask.affine, test_affine)

    # 5. Visualization of Image + ROI Overlay
    vis_path = tmp_path / "figures" / "roi_overlay.png"
    out_vis = visualize_roi_segmentation(
        original_volume=test_data,
        predicted_mask=seg_result["binary_mask"],
        probability_map=seg_result["probability_map"],
        output_path=vis_path,
        title="Test 3D U-Net ROI Overlay",
    )
    assert out_vis.exists()
    assert out_vis.stat().st_size > 0
