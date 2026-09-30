"""PyTorch Dataset definitions for neuroimaging deep learning ROI detection."""

from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Minimal fallback dummy class for systems where torch is installed later
    class Dataset:  # type: ignore
        pass

from src.core.logging import get_logger

logger = get_logger(__name__)


class NeuroimagingVolumeDataset(Dataset):
    """PyTorch Dataset for 3D neuroimaging volumes or sub-volume patches.

    Extracts (C, D, H, W) tensors for lightweight deep learning ROI prior estimators.
    """

    def __init__(
        self,
        volumes: List[np.ndarray],
        labels: Optional[List[np.ndarray]] = None,
        patch_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            volumes: List of 3D numpy arrays containing z-maps or contrasts.
            labels: Optional list of 3D binary ground truth / silver standard masks.
            patch_size: Optional (D, H, W) dimensions if extracting random/fixed patches.
        """
        self.volumes = volumes
        self.labels = labels
        self.patch_size = patch_size

        if self.labels is not None and len(self.labels) != len(self.volumes):
            raise ValueError("Number of volumes and labels must match.")

        logger.info(f"Initialized NeuroimagingVolumeDataset with {len(self.volumes)} volumes.")

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        vol = self.volumes[idx]
        # Shape: (1, D, H, W)
        vol_tensor = torch.from_numpy(vol).unsqueeze(0).float()

        if self.labels is not None:
            lbl = self.labels[idx]
            lbl_tensor = torch.from_numpy(lbl.astype(np.float32)).unsqueeze(0).float()
            return vol_tensor, lbl_tensor

        # Dummy label if none provided (unsupervised or inference mode)
        return vol_tensor, torch.empty(0)
