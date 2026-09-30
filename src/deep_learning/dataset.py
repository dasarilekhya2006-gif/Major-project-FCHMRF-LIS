"""Neuroimaging dataset and DataLoader implementations for 3D volumetric ROI detection.

Supports strictly separated operational modes:
  1. SUPERVISED: Requires genuine paired image and ground-truth ROI NIfTI volumes.
  2. PRETRAINED_EVAL: Unsupervised/inference dataset for applying existing checkpoints without labels.
  3. SYNTHETIC_BENCHMARK: Controlled mathematical synthetic simulations where cluster activation
     boundaries are defined analytically by construction (no fabricated clinical data).
"""

from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

from src.core.exceptions import DataValidationError, DeepLearningModelError
from src.core.logging import get_logger
from src.data.loaders import load_nifti
from src.data.schemas import NeuroimagingVolume
from src.data.synthetic import generate_synthetic_spatial_dataset

logger = get_logger(__name__)


class DatasetMode(str, Enum):
    """Operational mode separating genuine supervised labels from synthetic benchmarks."""

    SUPERVISED = "supervised"
    PRETRAINED_EVAL = "pretrained_eval"
    SYNTHETIC_BENCHMARK = "synthetic_benchmark"


if TORCH_AVAILABLE:

    class NeuroimagingROIDataset(Dataset):
        """PyTorch Dataset for 3D neuroimaging volumes and corresponding ROI target masks."""

        def __init__(
            self,
            image_paths: List[Union[str, Path]],
            mask_paths: Optional[List[Union[str, Path]]] = None,
            mode: Union[DatasetMode, str] = DatasetMode.SUPERVISED,
            transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            canonical_reorient: bool = True,
        ) -> None:
            """Initialize the dataset.

            Args:
                image_paths: List of file paths to 3D NIfTI images (.nii, .nii.gz).
                mask_paths: List of file paths to corresponding binary ROI masks.
                mode: Operational mode ('supervised', 'pretrained_eval', 'synthetic_benchmark').
                transform: Optional callable preprocessing/augmentation transform.
                canonical_reorient: Whether to canonically reorient volumes to RAS+ space.

            Raises:
                DataValidationError: If supervised mode is selected but mask_paths are missing or unequal in length.
            """
            self.image_paths = [Path(p) for p in image_paths]
            self.mode = DatasetMode(mode)
            self.transform = transform
            self.canonical_reorient = canonical_reorient

            if self.mode == DatasetMode.SUPERVISED:
                if mask_paths is None:
                    raise DataValidationError(
                        "Supervised mode requires genuine ground-truth mask_paths. "
                        "If ground-truth labels are not available, use mode='pretrained_eval' or 'synthetic_benchmark'."
                    )
                if len(self.image_paths) != len(mask_paths):
                    raise DataValidationError(
                        f"Mismatched dataset lengths: {len(self.image_paths)} images vs {len(mask_paths)} masks."
                    )
                self.mask_paths = [Path(p) for p in mask_paths]
            else:
                self.mask_paths = [Path(p) for p in mask_paths] if mask_paths is not None else None

            logger.info(f"Initialized NeuroimagingROIDataset: {len(self.image_paths)} samples in mode='{self.mode.value}'")

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, np.ndarray]]:
            img_path = self.image_paths[idx]
            vol = load_nifti(img_path, canonical_reorient=self.canonical_reorient, validate=True)
            img_data = vol.data[..., 0] if vol.is_4d else vol.data

            if self.transform is not None:
                img_data = self.transform(img_data)

            # Convert to (1, D, H, W) tensor
            img_tensor = torch.from_numpy(img_data.astype(np.float32)).unsqueeze(0)

            sample: Dict[str, Union[torch.Tensor, str, np.ndarray]] = {
                "image": img_tensor,
                "subject_id": vol.subject_id or img_path.stem,
                "affine": vol.affine,
                "voxel_dimensions": np.array(vol.voxel_dimensions[:3], dtype=np.float32),
            }

            if self.mode == DatasetMode.SUPERVISED and self.mask_paths is not None:
                mask_vol = load_nifti(self.mask_paths[idx], canonical_reorient=self.canonical_reorient, validate=False)
                mask_data = mask_vol.data[..., 0] if mask_vol.is_4d else mask_vol.data
                mask_binary = (mask_data > 0.5).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)
                sample["mask"] = mask_tensor
            else:
                # In eval mode without masks, provide a dummy zero tensor
                sample["mask"] = torch.zeros_like(img_tensor)

            return sample


    class SyntheticROIDataset(Dataset):
        """Controlled synthetic neuroimaging dataset with mathematically defined ground-truth clusters.

        Used strictly for scientific simulation and reproducibility benchmarks without fabricating clinical labels.
        """

        def __init__(
            self,
            num_samples: int = 20,
            grid_size: Tuple[int, int, int] = (32, 32, 16),
            snr: float = 2.5,
            num_clusters: int = 3,
            seed: int = 42,
        ) -> None:
            """Initialize synthetic dataset."""
            self.num_samples = num_samples
            self.grid_size = grid_size
            self.snr = snr
            self.num_clusters = num_clusters
            self.seed = seed

            # Pre-generate synthetic samples reproducibly
            self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
            for i in range(num_samples):
                sample_sim = generate_synthetic_spatial_dataset(
                    grid_size=grid_size,
                    snr=snr,
                    num_clusters=num_clusters,
                    seed=seed + i,
                )
                gt_mask = getattr(sample_sim, "ground_truth", getattr(sample_sim, "ground_truth_states", None))
                self.samples.append((sample_sim.z_map, gt_mask.astype(np.float32)))

            logger.info(
                f"Generated SyntheticROIDataset: {num_samples} samples, shape={grid_size}, SNR={snr}, seed={seed}"
            )

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, np.ndarray]]:
            z_map, gt_states = self.samples[idx]

            img_tensor = torch.from_numpy(z_map.astype(np.float32)).unsqueeze(0)
            mask_tensor = torch.from_numpy(gt_states.astype(np.float32)).unsqueeze(0)

            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "subject_id": f"synthetic_{idx:03d}",
                "affine": np.eye(4, dtype=np.float64),
                "voxel_dimensions": np.array([2.0, 2.0, 2.0], dtype=np.float32),
            }


    def create_dataloaders(
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders for train, validation, and test splits.

        Args:
            train_dataset: Training Dataset.
            val_dataset: Optional Validation Dataset.
            test_dataset: Optional Testing Dataset.
            batch_size: Mini-batch size.
            num_workers: DataLoader multiprocessing worker count.
            pin_memory: Whether to pin memory for fast GPU transfer.

        Returns:
            Dict[str, DataLoader]: Mapped data loaders.
        """
        loaders: Dict[str, DataLoader] = {}

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if val_dataset is not None:
            loaders["val"] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        if test_dataset is not None:
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        return loaders


    def split_dataset_by_subject(
        subject_ids: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[int]]:
        """Partition sample indices strictly by unique subject ID to prevent data leakage.

        Guarantees:
          1. All samples from a given subject belong to exactly one split.
          2. Disjoint subject sets: train_subjects ∩ val_subjects = ∅, etc.
          3. Exact reproducibility via deterministic pseudo-random permutation.

        Args:
            subject_ids: List of subject identifiers corresponding to each sample index.
            train_ratio: Proportion of unique subjects for training (default: 0.70).
            val_ratio: Proportion of unique subjects for validation (default: 0.15).
            test_ratio: Proportion of unique subjects for testing (default: 0.15).
            seed: Reproducibility random seed.

        Returns:
            Dict[str, List[int]]: Dictionary containing lists of sample indices:
                - 'train': Sample indices for training set
                - 'val': Sample indices for validation set
                - 'test': Sample indices for test set
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-4):
            raise ValueError(
                f"Split ratios must sum to 1.0; got {train_ratio + val_ratio + test_ratio:.4f}"
            )

        # Map unique subjects to their sample indices
        subj_to_indices: Dict[str, List[int]] = {}
        for idx, subj in enumerate(subject_ids):
            subj_to_indices.setdefault(str(subj), []).append(idx)

        unique_subjects = sorted(list(subj_to_indices.keys()))
        rng = np.random.RandomState(seed)
        shuffled_subjects = rng.permutation(unique_subjects)

        n_total = len(shuffled_subjects)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio))) if val_ratio > 0 and n_total > 2 else 0
        
        # Ensure splits don't exceed total subjects
        if n_train + n_val > n_total:
            n_train = max(1, n_total - n_val)

        train_subjs = set(shuffled_subjects[:n_train])
        val_subjs = set(shuffled_subjects[n_train:n_train + n_val])
        test_subjs = set(shuffled_subjects[n_train + n_val:])

        # Verify no subject leakage
        assert train_subjs.isdisjoint(val_subjs), "Subject leakage detected between train and val splits!"
        assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected between train and test splits!"
        assert val_subjs.isdisjoint(test_subjs), "Subject leakage detected between val and test splits!"

        train_indices: List[int] = []
        for s in train_subjs:
            train_indices.extend(subj_to_indices[s])

        val_indices: List[int] = []
        for s in val_subjs:
            val_indices.extend(subj_to_indices[s])

        test_indices: List[int] = []
        for s in test_subjs:
            test_indices.extend(subj_to_indices[s])

        logger.info(
            f"Subject-level split: {len(unique_subjects)} subjects -> "
            f"Train: {len(train_subjs)} subjects ({len(train_indices)} samples), "
            f"Val: {len(val_subjs)} subjects ({len(val_indices)} samples), "
            f"Test: {len(test_subjs)} subjects ({len(test_indices)} samples)"
        )

        return {
            "train": sorted(train_indices),
            "val": sorted(val_indices),
            "test": sorted(test_indices),
        }

else:

    class NeuroimagingROIDataset:  # type: ignore
        pass

    class SyntheticROIDataset:  # type: ignore
        pass

    def create_dataloaders(*args, **kwargs):  # type: ignore
        raise DeepLearningModelError("PyTorch is required for DataLoader creation.")
