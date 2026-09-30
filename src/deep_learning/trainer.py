"""Training, validation, and testing loop for 3D U-Net neuroimaging ROI detection models."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from src.core.exceptions import DeepLearningModelError
from src.core.logging import get_logger
from src.core.seed import set_seed
from src.deep_learning.metrics import MetricTracker, compute_segmentation_metrics

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ROIPriorTrainer:
    """Orchestrates model training, validation, testing, metric tracking, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch model subclassing BaseROIPriorModel.
            optimizer: Optional PyTorch optimizer (defaults to AdamW).
            loss_fn: Optional loss function (defaults to Combined BCE-Dice).
            device: Optional execution device string ('cuda', 'cpu'). Auto-selects if None.
            checkpoint_dir: Directory where checkpoints and training curves are saved.
            learning_rate: Optimizer learning rate.
            weight_decay: L2 weight regularization factor.
            seed: Reproducibility seed.
        """
        if not TORCH_AVAILABLE:
            raise DeepLearningModelError("PyTorch is required to initialize ROIPriorTrainer.")

        set_seed(seed)

        # Device selection with CPU fallback
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(f"CUDA requested ('{device}') but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

        self.model = model.to(self.device)

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            from src.deep_learning.losses import CombinedBceDiceLoss
            self.loss_fn = CombinedBceDiceLoss(from_logits=True)

        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # History tracking across epochs
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
        }
        self.best_val_dice = -1.0
        self.best_checkpoint_path: Optional[Path] = None

        logger.info(
            f"Initialized ROIPriorTrainer on device='{self.device}', seed={seed}, checkpoint_dir='{self.checkpoint_dir}'"
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            dataloader: DataLoader yielding dictionary or (images, masks).

        Returns:
            float: Average epoch training loss.
        """
        self.model.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                volumes = batch["image"]
                targets = batch["mask"]
            else:
                volumes, targets = batch

            if targets.numel() == 0 or volumes.numel() == 0:
                continue

            volumes = volumes.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            # Request raw logits for numerically stable loss computation
            logits = self.model(volumes, return_logits=True) if hasattr(self.model, "required_divisibility") else self.model(volumes)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        return avg_loss

    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """Evaluate loss, Dice, IoU, Precision, and Recall on validation or test split.

        Args:
            dataloader: Evaluation DataLoader.
            split_name: Name of split ('val' or 'test').

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    volumes = batch["image"]
                    targets = batch["mask"]
                else:
                    volumes, targets = batch

                if targets.numel() == 0 or volumes.numel() == 0:
                    continue

                volumes = volumes.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                try:
                    logits = self.model(volumes, return_logits=True)
                    probs = torch.sigmoid(logits)
                except (TypeError, AttributeError):
                    probs = self.model(volumes)
                    # Convert probabilities to numerical log-odds logits for loss_fn(from_logits=True)
                    logits = torch.logit(torch.clamp(probs, 1e-7, 1.0 - 1e-7))

                loss = self.loss_fn(logits, targets)
                tracker.update(loss=loss.item(), predictions=probs, targets=targets, threshold=0.5)

        metrics = tracker.compute_averages()
        logger.info(
            f"[{split_name.upper()}] Loss: {metrics['loss']:.4f} | "
            f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}"
        )
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """Execute full training loop over specified number of epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Optional Validation DataLoader.
            epochs: Total epochs to train.
            save_best: If True, saves checkpoint whenever validation Dice improves.

        Returns:
            Dict[str, List[float]]: Complete training metric history.
        """
        logger.info(f"Starting model training for {epochs} epochs on device='{self.device}'")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, split_name="val")
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_dice"].append(val_metrics["dice"])
                self.history["val_iou"].append(val_metrics["iou"])
                self.history["val_precision"].append(val_metrics["precision"])
                self.history["val_recall"].append(val_metrics["recall"])

                # Checkpoint best model based on validation Dice score
                if save_best and val_metrics["dice"] > self.best_val_dice:
                    self.best_val_dice = val_metrics["dice"]
                    self.best_checkpoint_path = self.save_checkpoint(
                        filename="unet3d_best_checkpoint.pt",
                        epoch=epoch,
                        metrics=val_metrics,
                    )
                    logger.info(f"New best model saved with Val Dice: {self.best_val_dice:.4f}")
            else:
                logger.info(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f}")

        return self.history

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the held-out test split."""
        logger.info("Evaluating model on test split...")
        return self.evaluate(test_loader, split_name="test")

    def save_checkpoint(
        self,
        filename: str = "roi_unet3d_checkpoint.pt",
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save model weights, optimizer state, and training metadata to disk.

        Args:
            filename: Target checkpoint filename.
            epoch: Epoch index when saved.
            metrics: Optional dictionary of evaluation metrics.

        Returns:
            Path: Destination checkpoint path.
        """
        dest = self.checkpoint_dir / filename
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "history": self.history,
            "seed": self.seed,
        }
        torch.save(checkpoint_data, str(dest))
        logger.info(f"Saved model checkpoint to {dest}")
        return dest

    def load_checkpoint(self, path: Union[str, Path]) -> Dict:
        """Load weights and state from a checkpoint file.

        Args:
            path: Path to .pt checkpoint.

        Returns:
            Dict: Checkpoint payload dictionary.
        """
        p = Path(path)
        if not p.exists():
            raise DeepLearningModelError(f"Checkpoint file not found: {p}")
        checkpoint = torch.load(str(p), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        logger.info(f"Successfully loaded checkpoint from {p}")
        return checkpoint
