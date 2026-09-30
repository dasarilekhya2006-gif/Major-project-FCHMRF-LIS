# Dataset Specifications for fcHMRF-LIS and 3D U-Net Prior Training

This document defines the structural, statistical, and spatial requirements for neuroimaging datasets utilized in the **Adaptive Deep Learning-Assisted fcHMRF-LIS** framework.

---

## 1. Overview of Data Roles

The proposed framework operates on two distinct data representations:
1. **Statistical Parametric Maps (for fcHMRF-LIS Inference)**: 3D statistical volumes (e.g., $z$-scores, $t$-statistics, or contrast estimates) evaluated across $10^5 \text{ to } 10^6$ voxels for spatial False Discovery Rate (FDR) control.
2. **Training & Context Data (for 3D U-Net Prior Learning)**: 3D/4D volumetric data used to train the lightweight 3D CNN to estimate voxel-wise activation prior probabilities $\pi_i = P(\theta_i = 1 \mid \text{context})$.

---

## 2. Statistical Map Specifications (for fcHMRF-LIS)

The fcHMRF-LIS statistical engine requires voxel-wise test statistics formatted as follows:

| Property | Specification | Rationale & Requirements |
| :--- | :--- | :--- |
| **File Format** | NIfTI-1 or NIfTI-2 (`.nii`, `.nii.gz`) | Standardized neuroimaging format readable via NiBabel. |
| **Dimensionality** | 3D (`X × Y × Z`) | If 4D with a single timepoint (`X × Y × Z × 1`), it is automatically squeezed to 3D. |
| **Voxel Data Type** | `float32` (or convertible from `int16`/`float64`) | Single-precision floating point preserves precision while optimizing memory during graph construction. |
| **Statistic Type** | Two-tailed or one-tailed $z$-scores | Must have an unthresholded null distribution centered near $\mathcal{N}(0, 1)$ or an empirical null $\mathcal{N}(\mu_0, \sigma_0^2)$. $t$-statistics must be converted to $z$-scores via quantile transformation: $z_i = \Phi^{-1}(F_{t, df}(t_i))$. |
| **Spatial Coordinate System** | MNI152 Standard Space or aligned Native T1w | Invertible $4 \times 4$ affine matrix with non-zero determinant. Voxel dimensions typically $2.0 \times 2.0 \times 2.0 \text{ mm}^3$ or $3.0 \times 3.0 \times 3.0 \text{ mm}^3$. |
| **Orientation** | Canonical RAS+ (Right-Anterior-Superior) | Standardized orientation prevents left-right or anterior-posterior spatial graph inversion. |
| **Brain Mask** | Binary intracranial mask (`0` = background, `1` = brain) | Eliminates out-of-brain air and skull voxels, reducing the graph node count from $\sim 10^6$ to $\sim 1.5 \times 10^5$. |

---

## 3. 3D U-Net Prior Model Dataset Specifications

The lightweight 3D U-Net learns to map statistical fields to voxel-wise prior activation probabilities $\pi_i = P(\theta_i = 1)$.

### 3.1 Input Tensor Structure
- **Shape**: `(Batch_Size, Channels, Depth, Height, Width)` $\to (B, C, D, H, W)$
- **Channels ($C$)**:
  - $C = 1$: Single-channel input containing the unthresholded 3D statistical $z$-map.
  - $C = 2$ (Optional): Dual-channel input combining the statistical $z$-map with a coregistered structural T1-weighted anatomical volume or gray-matter probability mask.
- **Spatial Resolution**: Resampled to isotropic $2 \text{ mm}$ or $3 \text{ mm}$ voxels in MNI152 space (e.g., matrix dimensions $96 \times 112 \times 96$ or cropped $64 \times 64 \times 64$ sub-volumes).

### 3.2 Target Supervision Labels
For supervised model training and validation:
- **Synthetic Benchmark**: Binary ground truth maps $\boldsymbol{\theta} \in \{0, 1\}^{D \times H \times W}$ where active clusters are known deterministically.
- **Empirical Real-World Data**:
  - Consensus meta-analytic activation maps (e.g., derived from **NeuroSynth** or **NeuroQuery** across thousands of published studies).
  - High-sample-size group-level contrast maps thresholded at extremely stringent Family-Wise Error Rate ($\text{FWER} < 0.001$) serving as silver-standard pseudo-ground-truth for training priors.

### 3.3 Dataset Splitting Protocol
To ensure rigorous generalization and avoid data leakage:
- **Subject-Independent Splits**: All volumes from the same subject must reside exclusively in Train, Validation, or Test sets.
- **Split Proportions**:
  - **Training Set (70%)**: Used for optimizing 3D U-Net weights via Focal-Dice loss.
  - **Validation Set (15%)**: Used for early stopping, learning rate scheduling, and hyperparameter tuning.
  - **Held-Out Test Set (15%)**: Evaluated exclusively for final empirical FDR and statistical power benchmarking.

---

## 4. 4D fMRI Timeseries Requirements (Upstream Ingestion)

When starting from raw or minimally preprocessed 4D functional MRI data:

```
4D fMRI Timeseries (.nii.gz)
         │
         ▼
[First-Level GLM Analysis] ──── (Task Design Matrix / Regressors)
         │
         ▼
Unthresholded 3D Contrast / t-map
         │
         ▼
[Z-Score Normalization] ─────── (t -> z quantile transform)
         │
         ▼
fcHMRF-LIS Input Volume (.nii.gz)
```

### Preprocessing Prerequisites Before GLM:
1. **Motion Correction**: Rigid-body realignment to the first or mean functional volume.
2. **Slice-Timing Correction**: Temporal interpolation to correct for interleaved slice acquisition.
3. **Coregistration & Normalization**: Non-linear registration of EPI to T1w anatomical image, followed by normalization to MNI152 standard space.
4. **Temporal Filtering**: High-pass filtering (cutoff $\approx 128 \text{ s}$ or $0.008 \text{ Hz}$) to remove slow scanner drifts.
5. **No Excessive Spatial Smoothing**: Heavy spatial Gaussian smoothing ($> 6 \text{ mm}$ FWHM) should be avoided prior to fcHMRF-LIS, because the fully connected bilateral graph already models spatial and feature correlations adaptively without blurring fine activation boundaries.

---

## 5. Recommended Public Benchmark Repositories

Researchers can evaluate this pipeline using the following open-access data:

1. **Human Connectome Project (HCP) S1200**:
   - Access: [db.humanconnectome.org](https://db.humanconnectome.org/)
   - Tasks: Motor, Working Memory, Emotion, Social, Relational, Gambling, Language.
   - Assets: Pre-computed individual subject and group contrast maps in MNI space ($2 \text{ mm}$ isotropic).
2. **NeuroVault.org**:
   - Access: [neurovault.org](https://neurovault.org/)
   - Assets: $> 50,000$ public unthresholded 3D statistical maps from peer-reviewed fMRI investigations.
3. **OpenNeuro**:
   - Access: [openneuro.org](https://openneuro.org/)
   - Datasets: `ds000030` (UCLA Consortium for Neuropsychiatric Phenomics), `ds000108` (Reward processing).
4. **Synthetic Spatial Suite (`src.data.sample_data`)**:
   - Included directly in this codebase for reproducible testing without external downloads.
