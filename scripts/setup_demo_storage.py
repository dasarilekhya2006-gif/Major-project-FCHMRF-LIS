"""Helper script to initialize a demo experiment in backend/storage using verified artifacts."""

import json
from pathlib import Path
import shutil
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "backend" / "storage"
DEMO_ID = "demo_neuroimaging_phase8"
DEMO_DIR = STORAGE_DIR / DEMO_ID


def setup_demo():
    print(f"Setting up demo experiment at {DEMO_DIR}...")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    for sub in ["input", "preprocess", "segmentation", "statistics", "fchmrf", "fdr", "visualization"]:
        (DEMO_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Copy files
    src_prep = PROJECT_ROOT / "outputs" / "preprocessed" / "final_phase7_demo_preprocessed.nii.gz"
    src_roi = PROJECT_ROOT / "outputs" / "segmentations" / "final_phase7_demo_roi_binary.nii.gz"
    src_z = PROJECT_ROOT / "outputs" / "statistical_maps" / "final_phase7_demo_zscores.nii.gz"
    src_lis = PROJECT_ROOT / "outputs" / "fchmrf_results" / "final_phase7_demo_adaptive_lis_map.nii.gz"
    src_sig = PROJECT_ROOT / "outputs" / "fchmrf_results" / "final_phase7_demo_adaptive_significant_mask.nii.gz"
    src_base_sig = PROJECT_ROOT / "outputs" / "fchmrf_results" / "final_phase7_demo_significant_mask.nii.gz"

    if src_prep.exists():
        shutil.copy2(src_prep, DEMO_DIR / "input" / "raw_volume.nii.gz")
        shutil.copy2(src_prep, DEMO_DIR / "preprocess" / "preprocessed.nii.gz")
    if src_roi.exists():
        shutil.copy2(src_roi, DEMO_DIR / "segmentation" / "roi_mask.nii.gz")
    if src_z.exists():
        shutil.copy2(src_z, DEMO_DIR / "statistics" / "z_scores.nii.gz")
    if src_lis.exists():
        shutil.copy2(src_lis, DEMO_DIR / "fchmrf" / "lis_map.nii.gz")
    if src_sig.exists():
        shutil.copy2(src_sig, DEMO_DIR / "fdr" / "significant_mask.nii.gz")
    if src_base_sig.exists():
        shutil.copy2(src_base_sig, DEMO_DIR / "fdr" / "baseline_significant.nii.gz")

    # Load shape and voxel spacing from NIfTI
    img = nib.load(str(DEMO_DIR / "input" / "raw_volume.nii.gz"))
    shape = list(img.shape)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]

    sig_data = nib.load(str(DEMO_DIR / "fdr" / "significant_mask.nii.gz")).get_fdata()
    base_sig_data = nib.load(str(DEMO_DIR / "fdr" / "baseline_significant.nii.gz")).get_fdata()

    metadata = {
        "experiment_id": DEMO_ID,
        "filename": "sub-01_T1w_preprocessed.nii.gz",
        "created_at": "2026-09-27T16:00:00Z",
        "status": "completed",
        "modality": "fMRI / T1w",
        "shape": shape,
        "voxel_spacing": zooms,
        "orientation": "RAS",
        "stages": {
            "upload": "completed",
            "preprocess": "completed",
            "segmentation": "completed",
            "statistics": "completed",
            "fchmrf_lis": "completed",
            "fdr": "completed",
        },
        "parameters": {
            "normalization": "zscore",
            "denoising": "gaussian (FWHM 2.0mm)",
            "resampling": "none",
            "statistical_test": "one-sample t-test",
            "nominal_alpha": 0.05,
            "beta": 1.15,
            "lambda_dl": 0.65,
            "sigma_spatial": 2.2,
            "sigma_intensity": 1.35,
            "max_iterations": 35,
            "execution_times": {
                "preprocess": 0.12,
                "segmentation": 0.45,
                "statistics": 0.08,
                "fchmrf_lis": 1.14,
                "fdr": 0.01,
            },
        },
        "discoveries_proposed": int(np.sum(sig_data > 0)),
        "discoveries_baseline": int(np.sum(base_sig_data > 0)),
    }

    with open(DEMO_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Demo experiment '{DEMO_ID}' initialized successfully!")
    print(f"Shape: {shape}, Discoveries Proposed: {metadata['discoveries_proposed']}, Baseline: {metadata['discoveries_baseline']}")


if __name__ == "__main__":
    setup_demo()
