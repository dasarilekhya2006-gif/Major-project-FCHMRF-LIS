"""Small end-to-end training and inference execution script for Phase 3."""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import nibabel as nib
import numpy as np
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


def main():
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    print(f"CUDA Available: {cuda_avail}")
    print(f"Execution Device: {device}")

    # Generate benchmark dataset
    grid_size = (32, 32, 16)
    train_ds = SyntheticROIDataset(num_samples=8, grid_size=grid_size, snr=3.5, seed=42)
    val_ds = SyntheticROIDataset(num_samples=4, grid_size=grid_size, snr=3.5, seed=84)
    loaders = create_dataloaders(train_ds, val_ds, batch_size=2)

    # Subject-level split demonstration
    subject_ids = [f"sub-{i:02d}" for i in range(1, 13)]
    splits = split_dataset_by_subject(subject_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    print(f"Subject-level splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # Instantiate configurable 3D U-Net
    model = ConfigurableUNet3D(
        in_channels=1,
        out_classes=1,
        base_channels=8,
        depth=2,
        activation="leaky_relu",
        norm_type="instance_norm",
    )

    ckpt_dir = Path("outputs/phase3_demo/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = ROIPriorTrainer(
        model=model,
        checkpoint_dir=ckpt_dir,
        device=device,
        learning_rate=2e-3,
        seed=42,
    )

    # Run training for 3 epochs
    print("Training 3D U-Net for 3 epochs...")
    history = trainer.fit(loaders["train"], loaders["val"], epochs=3, save_best=True)

    # Evaluate validation metrics
    val_metrics = trainer.evaluate(loaders["val"], split_name="val")
    print("\n--- ACTUAL VALIDATION METRICS ---")
    for k, v in val_metrics.items():
        print(f"val_{k}: {v:.4f}")

    # Run volumetric inference on validation sample
    sample = val_ds[0]
    sample_img = sample["image"].squeeze(0).numpy()
    gt_mask = sample["mask"].squeeze(0).numpy()

    temp_nii = Path("outputs/phase3_demo/sample_input.nii.gz")
    nib.save(nib.Nifti1Image(sample_img, np.eye(4)), str(temp_nii))

    pred_dir = Path("outputs/phase3_demo/predictions")
    inf_result = segment_volume(
        temp_nii,
        model=model,
        device=device,
        threshold=0.5,
        output_dir=pred_dir,
        prefix="sub_val01",
        canonical_reorient=False,
    )

    sample_metrics = compute_segmentation_metrics(
        inf_result["binary_mask"],
        gt_mask,
        threshold=0.5,
    )

    print("\n--- ACTUAL INFERENCE SAMPLE METRICS ---")
    for k, v in sample_metrics.items():
        print(f"{k}: {v:.4f}")

    # Generate multi-planar visualization
    vis_dir = Path("outputs/figures")
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_path = vis_dir / "phase3_roi_overlay.png"

    visualize_roi_segmentation(
        original_volume=sample_img,
        predicted_mask=inf_result["binary_mask"],
        probability_map=inf_result["probability_map"],
        output_path=vis_path,
        title="Phase 3: 3D U-Net Predicted ROI Spatial Prior Overlay",
    )
    print(f"\nOverlay visualization saved to: {vis_path}")


if __name__ == "__main__":
    main()
