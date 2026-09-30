import numpy as np
from backend.services.storage_service import storage_service
brain_mask, _ = storage_service.load_stage_volume('exp_db32ca270e92', 'preprocess', 'brain_mask.nii.gz')
print(f'Total voxels: {brain_mask.size}')
print(f'Active voxels: {np.sum(brain_mask > 0)}')
