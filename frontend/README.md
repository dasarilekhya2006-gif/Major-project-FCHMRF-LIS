# Phase 12: Neuroimaging Research Dashboard

Interactive React + Vite web application for the **Adaptive fcHMRF-LIS Neuroimaging Analysis Pipeline**.

> **RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS**  
> *This system is developed for research and demonstration purposes only. It has not been validated, approved, or intended for clinical diagnosis, treatment decisions, or medical use.*

---

## 1. Quick Start

### Step 1: Start FastAPI Backend (Phase 11)
In the project root:
```bash
python scripts/run_backend.py --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Launch React Frontend (Phase 12)
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:3000` (or `http://localhost:5173`).

---

## 2. Key Features

- **Mandatory Research Disclaimer**: Fixed banner prominently stating prototype research constraints.
- **Visual Pipeline Stepper**:
  `Upload Data → Preprocess → 3D U-Net ROI → Statistics → fcHMRF-LIS → Apply FDR → View Results → Compare Baseline → Export Results`
  with states: `Pending`, `Running`, `Completed`, `Failed`.
- **Tri-Planar 2D Slice Viewer**:
  - Axial, Coronal, and Sagittal cross-sectional views.
  - Interactive slice slider and zoom controls (1.0x - 5.0x).
  - Scientific temperature colormap (Dark Orange $\to$ Orange $\to$ Bright Yellow) overlaid on anatomical grayscale MRI.
  - Voxel Inspector: Hover crosshairs display coordinates $(x, y, z)$, physical world $(X, Y, Z)$ mm coordinates via affine matrix, MRI intensity, $z$-score, LIS value, and FDR significance status.
- **3D Brain Isosurface Viewer**:
  - Renders Marching Cubes surface meshes (Brain shell, 3D U-Net ROI, Significant abnormality clusters).
  - Interactive 360-degree rotation (mouse drag), zoom (mouse wheel), and mesh visibility toggles.
- **Baseline vs Proposed Method Comparison**:
  - Synchronized side-by-side view (Standard Benjamini-Hochberg FDR vs Proposed Adaptive fcHMRF-LIS).
  - Discrepancy Classification Map: Consensus discoveries (green), Proposed-Only discoveries (orange), Baseline-Only discoveries (blue).
- **Quantitative Research Metrics Table**:
  - Direct quantitative comparison of discoveries, empirical FDR, sensitivity, specificity, precision, recall, and Dice similarity score.
- **Computational Latency Breakdown**:
  - Live execution times across preprocessing, 3D U-Net inference, voxel statistics, fcHMRF-LIS mean-field updates, and FDR control.
- **Experiment Provenance & History**:
  - Review past experiments (`exp_xxx`), dataset parameters, and reload any run into the workspace.
- **Export & Download**:
  - Download preprocessed NIfTI, ROI masks, statistical maps, LIS maps, and FDR masks.
  - Export CSV metrics and JSON metadata.
  - Export publication-ready canvas figure snapshots.
