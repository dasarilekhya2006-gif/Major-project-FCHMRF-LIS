# Research Notes: Adaptive Deep Learning-Assisted fcHMRF-LIS for Spatial FDR Control in Neuroimaging

**Project Title**: Adaptive Deep Learning-Assisted fcHMRF-LIS for Spatial False Discovery Rate Control in Neuroimaging  
**Date**: September 27, 2026  

---

## 1. Scientific Background & Problem Statement

In neuroimaging modalities such as functional MRI (fMRI), structural MRI, and Positron Emission Tomography (PET), statistical inference involves simultaneous hypothesis testing across $10^5$ to $10^6$ voxels. Controlling the False Discovery Rate (FDR)—the expected proportion of false positive detections among all rejected null hypotheses:

$$\text{FDR} = \mathbb{E}\left[ \frac{\text{FP}}{\max(\text{FP} + \text{TP}, 1)} \right] \le \alpha$$

### Limitations of Classical Approaches:
1. **Mass-Univariate Voxel-Wise FDR (Benjamini-Hochberg)**:
   - Assumes independent or PRDS test statistics.
   - Ignores local and long-range spatial correlations, leading to conservative detection thresholds and reduced statistical power (sensitivity).
2. **Ad-Hoc Spatial Smoothing + Cluster-Level Random Field Theory (RFT)**:
   - Fixed Gaussian smoothing kernel alters signal boundaries and introduces artificial activation bleed.
   - Cluster-extent RFT depends heavily on arbitrary cluster-defining thresholds and parametric assumptions that frequently violate nominal error rates (Eklund et al., 2016).

---

## 2. Core Methodological Framework

### 2.1 Fully Connected Hidden Markov Random Field (fcHMRF)
Let $\mathbf{Z} = (z_1, \dots, z_N)^T$ denote observed voxel test statistics, and $\boldsymbol{\theta} = (\theta_1, \dots, \theta_N)^T \in \{0, 1\}^N$ denote latent activation states ($0 = \text{inactive/null}$, $1 = \text{active}$).

The joint distribution is defined via Gibbs energy:
$$P(\boldsymbol{\theta} \mid \mathbf{Z}) = \frac{1}{\mathcal{Z}} \exp\left( -\sum_{i=1}^N U_i(\theta_i) - \sum_{i < j} V_{ij}(\theta_i, \theta_j) \right)$$

- **Unary Potentials**: Incorporate the two-group emission likelihoods and deep learning prior:
  $$U_i(0) = -\ln f_0(z_i) - \ln(1 - P(\theta_i = 1))$$
  $$U_i(1) = -\ln f_1(z_i) - \ln(P(\theta_i = 1))$$
  where $P(\theta_i = 1) = \lambda_{\text{DL}} \pi_i^{(\text{DL})} + (1 - \lambda_{\text{DL}}) (1 - p_0)$, and $f_0(z_i)$ is estimated via Efron's empirical null central-matching.
- **Pairwise Potentials**: Bilateral Gaussian affinity capturing spatial distance and radiometric difference:
  $$V_{ij}(\theta_i, \theta_j) = \beta \cdot w_{ij} \cdot \mathbf{1}(\theta_i \ne \theta_j)$$
  $$w_{ij} = \exp\left( -\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2 \sigma_{\text{spatial}}^2} \right) \exp\left( -\frac{(z_i - z_j)^2}{2 \sigma_{\text{feature}}^2} \right)$$

### 2.2 Mean-Field Variational Inference
The intractable posterior is approximated by a factorized distribution $Q(\boldsymbol{\theta}) = \prod_{i=1}^N q_i(\theta_i)$.
Coordinate ascent updates iteratively minimize $\text{KL}(Q \parallel P)$ with linear-time permutohedral lattice filtering for non-local bilateral message passing, or explicitly bounded memory-safe KDTree structures (limiting max spatial degree to a configured threshold) to prevent Out-Of-Memory ($O(N^2)$) allocation failures on large volumetric inputs.
$$q_i^{(t+1)}(1) = \sigma\left( \psi_i(0) - \psi_i(1) + 2\beta \sum_{j \ne i} W_{ij} (q_j^{(t)}(1) - 0.5) \right)$$

### 2.3 Local Index of Significance (LIS) and FDR Control
Under spatial dependence, the optimal test statistic is the Local Index of Significance (Sun & Cai, 2009):
$$\text{LIS}_i = P(\theta_i = 0 \mid \mathbf{Z}) = 1 - q_i(1)$$

To control FDR at nominal level $\alpha$:
1. Sort voxels in ascending order: $\text{LIS}_{(1)} \le \text{LIS}_{(2)} \le \dots \le \text{LIS}_{(N)}$.
2. Determine threshold index:
   $$k = \max \left\{ j : \frac{1}{j}\sum_{i=1}^j \text{LIS}_{(i)} \le \alpha \right\}$$
3. Reject null hypotheses for voxels $\{(1), \dots, (k)\}$.

---

## 3. Scientific Hypotheses for Experimental Validation

1. **Hypothesis 1 (Statistical Power)**: Dense spatial modeling via fcHMRF-LIS will achieve significantly higher true positive rates (statistical power) than classical Benjamini-Hochberg and Local FDR while bounding empirical False Discovery Proportion (FDP) $\le \alpha = 0.05$.
2. **Hypothesis 2 (Prior Contribution)**: Incorporating a 3D U-Net structural prior ($\pi_{\text{DL}}$) will accelerate convergence and improve spatial boundary localization (higher Dice and MCC) in anatomically constrained activations.
3. **Hypothesis 3 (Adaptive Calibration)**: Bayesian Optimization of $(\beta, \lambda_{\text{DL}})$ on an isolated validation partition will provide robust FDR control and higher sensitivity across diverse SNR conditions compared to fixed heuristic parameters.

---

## 4. Scientific Integrity & Experimental Safeguards

- **No Premature Novelty Claims**: Algorithmic components are treated as hypotheses to be demonstrated through controlled experiments.
- **Zero Data Leakage**: Subject partitions (Train / Validation / Test) must remain strictly isolated. Hyperparameter optimization must run strictly on validation folds.
- **NIfTI Affine Fidelity**: All coordinate transformations must preserve orientation, voxel spacing, and affine matrices.
- **Multi-Seed Determinism**: All Monte Carlo evaluations must execute across fixed seeds (e.g., 42, 101, 202) for exact multi-framework reproducibility.
