# Phase 8 Rigorous Experimental Comparison

**Monte Carlo Trials**: 5  
**Nominal Alpha**: 0.05  
**Signal-to-Noise Ratio (SNR)**: 3.0  

| Method | FDR (Empirical FDP) | TPR (Recall) | FPR | Precision | Specificity | Localization Error (vox) | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Benjamini-Hochberg FDR** | 0.0042 ± 0.0026 | 0.3798 ± 0.0347 | 0.0002 ± 0.0001 | 0.9958 ± 0.0026 | 0.9998 ± 0.0001 | 1.39 ± 0.51 | 0.001s |
| **Original fcHMRF-LIS** | 0.0076 ± 0.0020 | 0.4465 ± 0.0312 | 0.0004 ± 0.0001 | 0.9924 ± 0.0020 | 0.9996 ± 0.0001 | 1.23 ± 0.46 | 1.059s |
| **ROI + fcHMRF-LIS** | 0.0124 ± 0.0053 | 0.7228 ± 0.0254 | 0.0010 ± 0.0005 | 0.9876 ± 0.0053 | 0.9990 ± 0.0005 | 0.51 ± 0.23 | 1.021s |
| **Adaptive fcHMRF-LIS** | 0.0039 ± 0.0024 | 0.4175 ± 0.0316 | 0.0002 ± 0.0001 | 0.9961 ± 0.0024 | 0.9998 ± 0.0001 | 1.39 ± 0.49 | 1.099s |
| **ROI + Adaptive fcHMRF-LIS** | 0.0086 ± 0.0058 | 0.7456 ± 0.0236 | 0.0007 ± 0.0005 | 0.9914 ± 0.0058 | 0.9993 ± 0.0005 | 0.49 ± 0.21 | 1.141s |

---
*Note: Evaluation conducted under identical volume noise realizations and coordinates.*
