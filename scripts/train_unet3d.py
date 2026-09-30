"""Command-line interface to train and evaluate the 3D U-Net neuroimaging ROI detection model."""

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.deep_learning.models.unet3d import ConfigurableUNet3D
from src.deep_learning.dataset import (
    DatasetMode,
    NeuroimagingROIDataset,
    SyntheticROIDataset,
    create_dataloaders,
)
from src.deep_learning.losses import get_segmentation_loss
from src.deep_learning.trainer import ROIPriorTrainer
from src.deep_learning.inference import segment_volume
from src.deep_learning.visualization import plot_training_curves, visualize_roi_segmentation
from src.data.sample_data import create_sample_3d_statistical_nifti

logger = get_logger("train_unet3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and benchmark 3D U-Net for neuroimaging ROI detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration YAML file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override mini-batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["supervised", "synthetic_benchmark", "pretrained_eval"],
        help="Override operational mode.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run evaluation on checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for evaluation or fine-tuning.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(log_level="INFO")
    args = parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config {cfg_path} not found. Using defaults.")
        cfg = {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device_str = args.device or train_cfg.get("device", "cuda")
    epochs = args.epochs or train_cfg.get("epochs", 15)
    batch_size = args.batch_size or train_cfg.get("batch_size", 2)
    mode_str = args.mode or ds_cfg.get("mode", "synthetic_benchmark")
    mode = DatasetMode(mode_str)

    logger.info(f"Phase 3 Execution Mode: '{mode.value}' | Device: '{device_str}' | Epochs: {epochs}")

    # Build datasets according to operational mode
    if mode == DatasetMode.SUPERVISED:
        logger.info("Initializing SUPERVISED dataset with genuine user-provided labels...")
        train_imgs = ds_cfg.get("train_image_paths", [])
        train_masks = ds_cfg.get("train_mask_paths", [])
        if not train_imgs or not train_masks:
            raise ValueError(
                "Supervised mode requires genuine 'train_image_paths' and 'train_mask_paths' in config. "
                "Do NOT fabricate clinical labels; switch to mode='synthetic_benchmark' for simulated experiments."
            )
        train_ds = NeuroimagingROIDataset(train_imgs, train_masks, mode=DatasetMode.SUPERVISED)
        val_ds = NeuroimagingROIDataset(
            ds_cfg.get("val_image_paths", []), ds_cfg.get("val_mask_paths", []), mode=DatasetMode.SUPERVISED
        ) if ds_cfg.get("val_image_paths") else None
        test_ds = NeuroimagingROIDataset(
            ds_cfg.get("test_image_paths", []), ds_cfg.get("test_mask_paths", []), mode=DatasetMode.SUPERVISED
        ) if ds_cfg.get("test_image_paths") else None

    elif mode == DatasetMode.SYNTHETIC_BENCHMARK:
        logger.info("Initializing SYNTHETIC_BENCHMARK dataset (known mathematical ground truth)...")
        syn_cfg = ds_cfg.get("synthetic", {})
        n_train = syn_cfg.get("num_train_samples", 20)
        n_val = syn_cfg.get("num_val_samples", 6)
        n_test = syn_cfg.get("num_test_samples", 6)
        grid = tuple(syn_cfg.get("grid_size", [32, 32, 16]))
        snr = syn_cfg.get("snr", 2.5)

        train_ds = SyntheticROIDataset(num_samples=n_train, grid_size=grid, snr=snr, seed=seed)
        val_ds = SyntheticROIDataset(num_samples=n_val, grid_size=grid, snr=snr, seed=seed + 100)
        test_ds = SyntheticROIDataset(num_samples=n_test, grid_size=grid, snr=snr, seed=seed + 200)

    else:  # PRETRAINED_EVAL
        logger.info("Initializing PRETRAINED_EVAL mode (unsupervised evaluation)...")
        train_ds = None
        val_ds = None
        test_ds = None

    # Instantiate Configurable 3D U-Net
    model = ConfigurableUNet3D(
        in_channels=model_cfg.get("in_channels", 1),
        out_classes=model_cfg.get("out_classes", 1),
        base_channels=model_cfg.get("base_channels", 16),
        depth=model_cfg.get("depth", 3),
        activation=model_cfg.get("activation", "relu"),
        norm_type=model_cfg.get("norm_type", "batch_norm"),
        dropout_rate=model_cfg.get("dropout_rate", 0.1),
    )

    # Loss function
    loss_fn = get_segmentation_loss(
        loss_name=train_cfg.get("loss_function", "combined"),
        from_logits=True,
    )

    # Trainer
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    trainer = ROIPriorTrainer(
        model=model,
        loss_fn=loss_fn,
        device=device_str,
        checkpoint_dir=checkpoint_dir,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        seed=seed,
    )

    # Load existing checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Execute training if not eval-only
    if not args.eval_only and train_ds is not None:
        loaders = create_dataloaders(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            batch_size=batch_size,
        )

        history = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders.get("val"),
            epochs=epochs,
            save_best=True,
        )

        # Plot and save training dynamics curves
        curves_path = Path("outputs/figures/unet_training_curves.png")
        plot_training_curves(history, output_path=curves_path)
        logger.info(f"Saved training curves to {curves_path}")

        # Final evaluation on held-out test split
        if "test" in loaders:
            test_metrics = trainer.test(loaders["test"])
            logger.info(f"Final Test Evaluation: {test_metrics}")

    # Demonstrate volumetric inference and visualization
    sample_vol_path = Path("data/raw/synthetic_sample_input.nii.gz")
    if not sample_vol_path.exists():
        sample_vol_path.parent.mkdir(parents=True, exist_ok=True)
        create_sample_3d_statistical_nifti(sample_vol_path, shape=(32, 32, 16), seed=seed)

    logger.info(f"Running 3D U-Net volumetric inference on {sample_vol_path}...")
    seg_res = segment_volume(
        sample_vol_path,
        model=trainer.model,
        output_dir="outputs/segmentations",
        prefix="synthetic_demo",
    )

    # Generate ROI visualization (Original, Predicted ROI, Overlay)
    from src.data.loaders import load_nifti
    orig_vol = load_nifti(sample_vol_path)
    viz_path = Path("outputs/figures/roi_segmentation_demo.png")
    visualize_roi_segmentation(
        original_volume=orig_vol.data,
        predicted_mask=seg_res["binary_mask"],
        probability_map=seg_res["probability_map"],
        output_path=viz_path,
        title="3D U-Net Neuroimaging ROI Localization Demo",
    )
    logger.info(f"ROI localization visualization saved to {viz_path}")
    logger.info("Phase 3 Execution completed successfully.")


if __name__ == "__main__":
    main()
