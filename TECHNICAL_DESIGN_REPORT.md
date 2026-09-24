# Technical Design Report: Optimization Dissipation, Manifold Curvature, and Neural-Network Robustness

**Status**: AWAITING APPROVAL BEFORE IMPLEMENTATION  
**Version**: 1.0  
**Date**: September 2026

---

## Table of Contents

1. [Paper Citations and DOIs](#1-paper-citations-and-dois)
2. [Exact Curvature Equations](#2-exact-curvature-equations)
3. [Plain-Language Explanation of Curvature Calculation](#3-plain-language-explanation-of-curvature-calculation)
4. [Official Author Code/Repositories](#4-official-author-coderepositories)
5. [Mapping Between Published Algorithm and Proposed Experiment](#5-mapping-between-published-algorithm-and-proposed-experiment)
6. [Mathematical Derivation of the Dissipation Proxy](#6-mathematical-derivation-of-the-dissipation-proxy)
7. [Interpretation of Gradient Norm Squared as Dissipation](#7-interpretation-of-gradient-norm-squared-as-dissipation)
8. [Limitations for Discrete SGD, Momentum, and Adam](#8-limitations-for-discrete-sgd-momentum-and-adam)
9. [Exact Variables to Record](#9-exact-variables-to-record)
10. [Proposed Checkpoint Frequency](#10-proposed-checkpoint-frequency)
11. [Computational Cost Estimate](#11-computational-cost-estimate)
12. [Statistical Analysis Plan](#12-statistical-analysis-plan)
13. [Potential Confounding Variables](#13-potential-confounding-variables)
14. [Unit Test Plan](#14-unit-test-plan)
15. [Proposed Figures](#15-proposed-figures)
16. [Minimum Reproducible Experiment](#16-minimum-reproducible-experiment)
17. [Prior Literature on Dissipation-Curvature Relationship](#17-prior-literature-on-dissipation-curvature-relationship)
18. [Critical Challenges to the Framework](#18-critical-challenges-to-the-framework)

---

## 1. Paper Citations and DOIs

### Primary Paper: Sekmen-Bilgin (Nature Communications Engineering)

**Full Citation:**
> Sekmen, A. & Bilgin, B. Manifold-based approach for neural network robustness analysis. *Communications Engineering* **3**, 118 (2024). https://doi.org/10.1038/s44172-024-00263-8

**DOI:** `10.1038/s44172-024-00263-8`

**PMC ID:** PMC11344765

**Key Prior Work:**
> Sekmen, A. & Bilgin, B. Manifold curvature estimation for neural networks. In *2022 IEEE International Conference on Big Data (Big Data)* 3903–3908 (Osaka, Japan, 2022).

### Secondary Paper: Kaufman-Azencot (ICML 2023)

**Full Citation:**
> Kaufman, I. & Azencot, O. Data Representations' Study of Latent Image Manifolds. In *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 15928–15945 (2023). https://proceedings.mlr.press/v202/kaufman23a.html

**arXiv:** `2305.19730`

### Foundational Curvature Algorithm

Both papers build on:
> Li, Y. Curvature-aware manifold learning. *Pattern Recognition* **83**, 273–286 (2018). https://doi.org/10.1016/j.patcog.2018.05.027

---

## 2. Exact Curvature Equations

### 2.1 Sekmen-Bilgin Weighted Angle Method

The Sekmen-Bilgin paper introduces a **weighted angle** approach that improves upon standard principal angles between subspaces.

#### Local Subspace Construction

For each data point x_i on manifold M:

1. **Find neighbors**: Let G_i = {x_{i_k}}_{k=1}^{N_i} be the set of N_i neighboring points selected by shortest Euclidean distances.

2. **Create local data matrix**:

   M_i = [x_{i_1} - μ_i  ···  x_{i_{N_i}} - μ_i]

   where μ_i is the mean of neighboring points.

3. **SVD decomposition**:

   M_i = U_i Σ_i V_i^T

   where U_i = [u_1 ··· u_{N_i}].

4. **Define local subspace**:

   S_i = span{u_1, ..., u_{d_i}}

   where d_i is the local subspace dimension (estimated via rank estimation or set as hyperparameter).

#### Weighted Angle Definition

**Definition 1 (Weighted Angle):** Let M_i and M_j be matrices whose columns are data points for subspaces S_i and S_j. With SVDs M_i = U_iΣ_iV_i^T and M_j = U_jΣ_jV_j^T, define:

   Q = (U_iΣ_i)^T(U_jΣ_j)

Compute SVD of Q: Q = U_QΣ_QV_Q^T

The **weighted angle** between S_i and S_j is:

```
θ_ij = arccos(tr(Σ_Q) / tr(Σ_i^T Σ_j))
```

Note: If Σ_i^T Σ_j or Σ_Q is not square, zero-pad to make square.

#### Point-Wise and Manifold Curvature

**Definition 2 (Curvature):** For manifold M with data points C = {x_i}_{i=1}^N:

1. **Local curvature at x_i**: Average weighted angle between S_i and its neighboring subspaces:

   θ_i = (Σ_{j=1}^{N_i} θ_ij) / N_i

2. **Manifold curvature**:

   ρ_M = (Σ_{i=1}^N θ_i) / N

#### Robustness Measure

**Definition 3 (Robustness):** For network Net_k with m output class manifolds {M_i^k}_{i=1}^m:

   R_{Net_k} = (Σ_{i=1}^m ρ_{M_i^k}) / m

Lower curvature → higher robustness (empirical correlation demonstrated).

---

### 2.2 Kaufman-Azencot CAML Method

The CAML (Curvature-Aware Manifold Learning) algorithm estimates **principal curvatures** via second-order Taylor expansion of the embedding map.

#### Embedding Map Approximation

Assume data Y = {y_i}_{i=1}^N ⊂ R^D lies on a d-dimensional manifold M. The embedding map f: R^d → R^D satisfies:

   y_i = f(x_i) + ε_i

where X = {x_i}_{i=1}^N ⊂ R^d are low-dimensional representations.

#### Local Coordinate Frame

For point y_i ∈ Y with neighborhood {y_{i_1}, ..., y_{i_K}}:

1. Use SVD to construct **local natural orthonormal coordinate frame**:

   {∂/∂x^1, ..., ∂/∂x^d, ∂/∂y^1, ..., ∂/∂y^{D-d}}

   - First d vectors: tangent space basis
   - Remaining D-d vectors: normal space basis

2. In this frame, embedding becomes:

   f(x^1, ..., x^d) = [x^1, ..., x^d, f^1, ..., f^{D-d}]

#### Hessian Estimation via Taylor Expansion

For each component f^α (α = 1, ..., D-d), second-order Taylor expansion at u_{i_j} around x_i:

   f^α(u_{i_j}) ≈ f^α(x_i) + Δ_{x_i}^T ∇f^α + (1/2)Δ_{x_i}^T H^α Δ_{x_i}

where:
- Δ_{x_i} = u_{i_j} - x_i (projected neighbor displacement)
- H^α = ∂²f^α/∂x^i∂x^j is the Hessian matrix

This forms a **linear regression problem** to estimate H^α from the neighborhood.

#### Principal Curvatures

**Principal curvatures** are the eigenvalues of H^α:

   eigenvalues(H^α) = {κ_1^α, ..., κ_d^α}

Each sample has (D-d) × d principal curvatures total.

#### Mean Absolute Principal Curvature (MAPC)

   MAPC = (1 / (N(D-d)d)) Σ_{i=1}^N Σ_{α=1}^{D-d} Σ_{k=1}^d |κ_k^α(y_i)|

This is the primary curvature metric used by Kaufman-Azencot.

---

### 2.3 Comparison of Methods

| Aspect | Sekmen-Bilgin | Kaufman-Azencot |
|--------|--------------|-----------------|
| **Curvature type** | Weighted angles between subspaces | Principal curvatures (Hessian eigenvalues) |
| **Theoretical basis** | Discrete differential geometry | Riemannian geometry |
| **Neighborhood** | K-nearest neighbors in input space | SVD-based artificial augmentation |
| **ID estimation** | Rank estimation | TwoNN algorithm |
| **Primary metric** | ρ_M (average weighted angle) | MAPC (mean absolute principal curvature) |
| **Robustness link** | Direct: lower curvature = higher robustness | Indirect: curvature gap correlates with generalization |

**Recommendation for this project:** Use the Sekmen-Bilgin weighted angle method as primary (explicitly designed for robustness), with Kaufman-Azencot CAML as secondary validation.

---

## 3. Plain-Language Explanation of Curvature Calculation

### Sekmen-Bilgin Method (Weighted Angles)

**Intuition:** Imagine standing at a point on a curved surface. Look at your immediate neighbors. If the surface is flat (like a table), all your neighbors lie in the same plane as you. If the surface is curved (like a ball), your neighbors' local tangent planes are tilted relative to yours.

**Algorithm in plain language:**

1. **For each data point** (e.g., each image representation in a neural network layer):
   - Find its K nearest neighbors
   - Fit a small "local plane" (subspace) to those neighbors using SVD
   
2. **Measure the angle between neighboring planes:**
   - The key innovation is using "weighted angles" that account for both:
     - The **direction** of the planes (traditional principal angles)
     - The **spread/magnitude** of data within each plane (via singular values)
   
3. **Average the angles:**
   - Average over all neighbors → curvature at that point
   - Average over all points → curvature of the manifold
   
4. **Robustness interpretation:**
   - A "flat" output manifold (low curvature) means the network's outputs vary smoothly with inputs
   - A "curved" manifold means small input changes can cause large output changes → less robust

### Kaufman-Azencot Method (Principal Curvatures)

**Intuition:** A circle has constant curvature 1/r. A saddle has positive curvature in one direction and negative in another. Principal curvatures capture these directional curvatures.

**Algorithm in plain language:**

1. **Generate a dense neighborhood:** Use SVD to create artificial neighbors (zeroing small singular values of the image)

2. **Estimate the embedding function locally:** Fit a second-order polynomial (Taylor expansion) that maps the low-dimensional "intrinsic" coordinates to the high-dimensional representation

3. **Extract the Hessian matrix:** The second-order coefficients form the Hessian, whose eigenvalues are principal curvatures

4. **Summarize:** Take the mean absolute value of all principal curvatures across all points

---

## 4. Official Author Code/Repositories

### Kaufman-Azencot Implementation

**Repository:** https://github.com/azencot-group/CRLM

**Key files:**
- `est_curv.py` - Main entry point for curvature estimation
- `utils/curv_utils.py` - Orchestration of curvature calculation
- `CAML/caml.py` - Core CAML algorithm implementation
- `utils/analysis_utils.py` - Neighborhood generation, ID estimation

**Implementation details observed:**
- Uses TwoNN for intrinsic dimension estimation
- SVD-based neighborhood augmentation (zeros 10 smallest singular values per channel)
- GPU-accelerated Hessian estimation via batched linear algebra
- Supports ResNet and VGG families on CIFAR-10/100

### Sekmen-Bilgin Implementation

**Status:** "The codes for algorithms and carrying out the experiments in this study are available from the corresponding author upon request."

**Contact:** Bahadir Bilgin (bbilgin@tnstate.edu), Tennessee State University

**Recommendation:** 
1. Contact authors to request official implementation
2. If unavailable, implement Algorithm 1 from the paper (clearly documented pseudocode provided)
3. Validate against their reported results on MNIST and Extended YaleB

---

## 5. Mapping Between Published Algorithm and Proposed Experiment

### What the Published Methods Provide

| Published Method | What It Measures | Our Use |
|-----------------|------------------|---------|
| Sekmen-Bilgin weighted angles | Curvature of output manifold at a single time | K_l(t) per layer, per checkpoint |
| Kaufman-Azencot MAPC | Curvature profile across layers | Validation; layer-wise curvature evolution |
| Black-box robustness measure | Average curvature → robustness proxy | R_curv(t) for comparison |

### New Measurements We Add

| Our Measurement | Definition | Purpose |
|----------------|------------|---------|
| Layer-wise dissipation D_l(t) | ‖∇_{W_l}L(t)‖_F² | Optimization activity indicator |
| Parameter motion ΔW_l(t) | W_l(t+1) - W_l(t) | Actual displacement vs. gradient |
| Temporal evolution K_l(t) | Curvature at each checkpoint | Track geometric reorganization |
| Lagged correlation | D_l(t) → K_l(t+τ) | Test predictive relationship |

### Mapping Table

| Paper Quantity | Our Notation | Computation |
|----------------|--------------|-------------|
| Data matrix M_i | Same | Per-point, per-layer activations |
| Weighted angle θ_ij | Same | Equation (2) from Sekmen-Bilgin |
| Manifold curvature ρ_M | K_l(t) | Per-layer, per-checkpoint |
| Robustness R_{Net_k} | R_curv(t) | Average over output manifolds |
| — | D_l(t) | **New:** Frobenius norm of layer gradients squared |
| — | E_l^diss(T) | **New:** Cumulative dissipation |

---

## 6. Mathematical Derivation of the Dissipation Proxy

### Gradient Flow Dynamics

Consider continuous-time gradient flow on the loss L(θ):

   θ̇ = -∇_θ L

The time derivative of the loss is:

   dL/dt = ∇_θL · θ̇ = ∇_θL · (-∇_θL) = -‖∇_θL‖²

This is the **dissipation identity**: loss decreases at a rate equal to the squared gradient norm.

### Layer-Wise Decomposition

For a neural network with parameters partitioned by layer θ = (W_1, ..., W_L):

   ‖∇_θL‖² = Σ_{l=1}^L ‖∇_{W_l}L‖_F²

Define the **layer-wise optimization dissipation proxy**:

   **D_l(t) = ‖∇_{W_l}L(t)‖_F²**

And **total dissipation**:

   D(t) = Σ_{l=1}^L D_l(t)

### Cumulative Dissipation

For discrete time with step size Δt:

   **E_l^diss(T) = Σ_{t=1}^T D_l(t) · Δt**

For constant learning rate η, this approximates the integral:

   E_l^diss(T) ≈ ∫_0^T D_l(s) ds

---

## 7. Interpretation of Gradient Norm Squared as Dissipation

### Physical Analogy

In physics, **dissipation** refers to the irreversible transformation of energy, typically into heat. For gradient flow:

- L(θ) acts as a **potential energy**
- -dL/dt = ‖∇L‖² is the **dissipation rate**
- The system evolves toward lower energy states

This is analogous to overdamped motion in a viscous medium, where kinetic energy is negligible and all potential energy decrease goes to dissipation (friction/heat).

### Why "Dissipation" is Appropriate Terminology

1. **Lyapunov function:** L(θ) is a Lyapunov function for gradient flow; it monotonically decreases along trajectories.

2. **Energy interpretation:** The quantity ‖∇L‖² measures how rapidly the system dissipates potential energy.

3. **Connection to thermodynamics:** In stochastic gradient Langevin dynamics (SGLD), the loss landscape has formal thermodynamic interpretations where gradient magnitude relates to entropy production.

### What Dissipation Does NOT Mean Here

- It does **not** mean loss itself is "heat"
- It does **not** imply a full thermodynamic theory without further justification
- The term is borrowed from physics as a **useful analogy**, not a proven equivalence

---

## 8. Limitations for Discrete SGD, Momentum, and Adam

### 8.1 Discrete SGD

For discrete gradient descent with learning rate η:

   θ_{t+1} = θ_t - η∇L(θ_t)

The loss change is:

   ΔL = L(θ_{t+1}) - L(θ_t)

Taylor expansion:

   ΔL ≈ -η‖∇L‖² + (η²/2)∇L^T H ∇L + O(η³)

**Key insight:** The dissipation identity -ΔL ≈ η‖∇L‖² only holds when:
- η is small (first-order approximation valid)
- Curvature term (η²/2)∇L^T H ∇L is negligible

**Edge of Stability (Cohen et al., 2021):** When the maximum Hessian eigenvalue λ_max(H) ≈ 2/η, the quadratic term is no longer negligible, and loss can increase temporarily. This is commonly observed in neural network training.

**Implication:** Our dissipation proxy D_l(t) may not equal -ΔL/η at edge-of-stability regimes. We should record both D_l(t) and -ΔL(t) to detect discrepancies.

### 8.2 SGD with Momentum

With momentum coefficient β:

   v_{t+1} = βv_t + ∇L(θ_t)
   θ_{t+1} = θ_t - ηv_{t+1}

Now the parameter update is:

   Δθ = -η(βv_t + ∇L)

**Problem:** Parameter displacement Δθ includes momentum history, not just current gradient.

**Solution:** Record both:
- D_l(t) = ‖∇_{W_l}L(t)‖_F² (gradient-based)
- ‖ΔW_l(t)‖_F (actual displacement)

The ratio or discrepancy between these reveals momentum effects.

### 8.3 Adam and Adaptive Optimizers

Adam uses adaptive learning rates:

   m_t = β_1 m_{t-1} + (1-β_1)∇L
   v_t = β_2 v_{t-1} + (1-β_2)(∇L)²
   Δθ ∝ m_t / (√v_t + ε)

**Problems:**
1. Gradient magnitudes are normalized by √v_t, breaking direct connection to dissipation
2. Effective learning rate varies across parameters
3. The "dissipation" interpretation becomes highly non-trivial

**Recommendation:**
1. Start with vanilla SGD for cleanest interpretation
2. For Adam experiments, treat D_l(t) = ‖∇_{W_l}L‖_F² as an **activity measure** rather than true dissipation
3. The geometric interpretation (curvature changes) may still hold even if the thermodynamic interpretation is weakened

### 8.4 Mini-batch Stochasticity

With mini-batch SGD:

   ∇L_batch = ∇L_true + ξ

where ξ is gradient noise with variance σ²/B (batch size B).

**Implication:** Measured D_l(t) includes noise contribution. For fair comparison:
- Use the same batch size across experiments
- Consider averaging D_l(t) over multiple mini-batches per checkpoint
- Full-batch gradient computation for curvature checkpoints (computationally expensive but cleaner)

---

## 9. Exact Variables to Record

### 9.1 Training Variables (Every Epoch/Checkpoint)

| Variable | Symbol | Definition |
|----------|--------|------------|
| Training loss | L_train(t) | Loss on training set |
| Validation loss | L_val(t) | Loss on held-out validation set |
| Training accuracy | Acc_train(t) | Proportion correct on training set |
| Validation accuracy | Acc_val(t) | Proportion correct on validation set |
| Epoch number | t | Training epoch |
| Wall-clock time | T_wall(t) | Elapsed time |

### 9.2 Optimization Variables (Every Layer, Every Checkpoint)

| Variable | Symbol | Definition |
|----------|--------|------------|
| Instantaneous dissipation | D_l(t) | ‖∇_{W_l}L(t)‖_F² |
| Parameter displacement | ‖ΔW_l(t)‖_F | ‖W_l(t+1) - W_l(t)‖_F |
| Relative displacement | R_{W,l}(t) | ‖ΔW_l(t)‖_F / (‖W_l(t)‖_F + ε) |
| Cumulative dissipation | E_l^diss(t) | Σ_{s=1}^t D_l(s) Δt |
| Parameter norm | ‖W_l(t)‖_F | Frobenius norm of layer weights |
| Gradient direction | ĝ_l(t) | ∇_{W_l}L / ‖∇_{W_l}L‖_F (optional, for alignment) |

### 9.3 Geometric Variables (Every Layer, Every Checkpoint)

Using Sekmen-Bilgin weighted angle method:

| Variable | Symbol | Definition |
|----------|--------|------------|
| Layer curvature (per class) | K_l^c(t) | Weighted-angle curvature for class c manifold at layer l |
| Layer curvature (average) | K_l(t) | (1/C)Σ_c K_l^c(t) |
| Curvature variance | Var[K_l](t) | Variance across classes or samples |
| Curvature change | ΔK_l(t) | K_l(t+1) - K_l(t) |
| Smoothing quantity | S_l(t) | -Δ|K_l(t)| (if curvature magnitude meaningful) |

Using Kaufman-Azencot CAML (for validation):

| Variable | Symbol | Definition |
|----------|--------|------------|
| MAPC per layer | MAPC_l(t) | Mean absolute principal curvature at layer l |
| MAPC gap | ΔMAPC(t) | Gap between last two layers |
| Intrinsic dimension | ID_l(t) | TwoNN estimate of intrinsic dimension |

### 9.4 Robustness Variables (Selected Checkpoints)

| Variable | Symbol | Definition |
|----------|--------|------------|
| Clean accuracy | R_clean(t) | Test accuracy on clean data |
| Noise robustness | R_σ(t) | Accuracy under Gaussian noise (std=σ) |
| Corruption robustness | R_corr(t) | Accuracy under controlled corruptions |
| FGSM robustness | R_FGSM(t) | Accuracy under FGSM attack (ε-bounded) |
| PGD robustness | R_PGD(t) | Accuracy under PGD attack |

### 9.5 Jacobian/Metric Variables (Small Experiments Only)

| Variable | Symbol | Definition |
|----------|--------|------------|
| Layer Jacobian | J_l(x,t) | ∂h_l/∂h_{l-1} |
| Induced metric | g_l(x,t) | J_l^T J_l |
| Metric change | Δg_l(t) | Change in induced metric |
| Jacobian singular values | σ_k(J_l) | Singular values of Jacobian |

---

## 10. Proposed Checkpoint Frequency

### Phase I: MNIST MLP Experiments

| Phase | Epochs | Checkpoint Frequency | Rationale |
|-------|--------|---------------------|-----------|
| Early training | 0–10 | Every epoch | Rapid changes expected |
| Mid training | 10–30 | Every 2 epochs | Moderate changes |
| Late training | 30–50 | Every 5 epochs | Slower dynamics |

**Total checkpoints per run:** ~20–25

### Alternative: Adaptive Checkpointing

If computational resources permit, checkpoint based on **loss change magnitude**:
- Checkpoint when |ΔL| > τ or every k epochs (whichever first)
- This captures rapid early dynamics and sparse late dynamics naturally

### Dissipation Recording (Separate from Full Checkpoints)

Since D_l(t) is cheap to compute:
- Record dissipation metrics **every mini-batch** during training
- Aggregate to per-epoch statistics for analysis
- Store full per-batch traces for select seeds (for fine-grained temporal analysis)

---

## 11. Computational Cost Estimate

### 11.1 Per-Checkpoint Costs

**Network:** MLP 784→256→128→64→10 on MNIST (60,000 training samples)

| Operation | Time Estimate | Memory |
|-----------|---------------|--------|
| Forward pass (full dataset) | ~1 second | ~100 MB |
| Gradient computation (full batch) | ~2 seconds | ~200 MB |
| Curvature estimation (1000 samples, all layers) | ~10–30 seconds | ~500 MB |
| Robustness evaluation (5 noise levels, 1000 samples) | ~30 seconds | ~100 MB |

**Total per checkpoint:** ~1–2 minutes

### 11.2 Per-Run Costs

| Item | Value |
|------|-------|
| Training epochs | 50 |
| Checkpoints | 25 |
| Time per checkpoint | 1.5 minutes |
| Training time | ~10 minutes |
| **Total per run** | **~45–60 minutes** |

### 11.3 Full Experiment Costs

| Experiment | Runs | Time per Run | Total Time |
|------------|------|--------------|------------|
| SGD baseline (5 seeds) | 5 | 1 hour | 5 hours |
| Learning rate sweep (5 LR × 3 seeds) | 15 | 1 hour | 15 hours |
| Batch size sweep (3 BS × 3 seeds) | 9 | 1 hour | 9 hours |
| Optimizer comparison (3 opt × 3 seeds) | 9 | 1 hour | 9 hours |
| Activation comparison (3 act × 3 seeds) | 9 | 1 hour | 9 hours |
| Confirmatory (SGD, 10 seeds) | 10 | 1 hour | 10 hours |
| **Total Phase I** | | | **~60 hours** |

### 11.4 GPU Requirements

- Curvature estimation is GPU-accelerated (CAML uses batched SVD on GPU)
- Minimum: Single NVIDIA GPU with 8GB VRAM
- Recommended: 16GB GPU or multiple GPUs for parallel seed runs

---

## 12. Statistical Analysis Plan

### 12.1 Primary Hypothesis Tests

**H1: Dissipation predicts curvature change**

   D_l(t) → ΔK_l(t+τ)

Analysis:
1. **Lagged cross-correlation:** Compute Corr(D_l(t), ΔK_l(t+τ)) for τ ∈ {0, 1, 2, 5, 10} epochs
2. **Lagged regression:** ΔK_l(t+τ) = β_0 + β_1 D_l(t) + ε
3. **Granger causality test:** Does including past D_l improve prediction of K_l beyond past K_l alone?

### 12.2 Handling Repeated Measurements

**Problem:** Multiple epochs within same training run are not independent.

**Solution:** Mixed-effects models

   ΔK_{l,r,t} = β_0 + β_1 D_{l,r,t} + u_r + v_{l,r} + ε_{l,r,t}

where:
- r = run (random effect)
- l = layer (crossed random effect)
- t = time (within-run)

### 12.3 Correlations to Report

| Analysis | Variables | Method |
|----------|-----------|--------|
| Contemporaneous | (D_l(t), K_l(t)) | Pearson, Spearman |
| Lagged | (D_l(t), ΔK_l(t+τ)) | Cross-correlation |
| Cumulative | (E_l^diss(T), K_l(T)) | Regression |
| Robustness | (K_l(T), R(T)) | Correlation, regression |
| Incremental | (D_l(t), K_l(t)) → R(T) | Multiple regression, R² decomposition |

### 12.4 Effect Sizes and Confidence Intervals

Report for all correlations:
- Point estimate (e.g., r = 0.65)
- 95% confidence interval (bootstrap or analytic)
- Effect size interpretation (Cohen's conventions: 0.1 small, 0.3 medium, 0.5 large)

### 12.5 Multiple Comparisons

When testing across:
- Multiple layers (L = 4)
- Multiple lags (|τ| = 5)
- Multiple training phases (early/mid/late)

Apply Bonferroni or FDR correction:
- Bonferroni: α_adj = 0.05 / (4 × 5 × 3) = 0.0008
- FDR (Benjamini-Hochberg): Control FDR at 0.05

### 12.6 Change-Point Detection

For identifying training regimes:
- PELT (Pruned Exact Linear Time) algorithm
- Bayesian online change-point detection
- Visual inspection + domain knowledge

Report detected transitions with uncertainty estimates.

---

## 13. Potential Confounding Variables

### 13.1 Training Progress Confound

**Problem:** Both dissipation and curvature may decrease during training simply because the network is converging. Any correlation could be spurious.

**Control:**
- Regress out training epoch: ΔK_l(t) ~ D_l(t) + t
- Analyze residuals after detrending
- Compare correlation structure across different training configurations

### 13.2 Gradient Magnitude Confound

**Problem:** D_l(t) = ‖∇_{W_l}L‖_F² may correlate with curvature simply because larger gradients → larger representation changes.

**Control:**
- Separate gradient magnitude from gradient direction
- Analyze ΔK_l / D_l (curvature change per unit dissipation)
- Compare runs with similar total D_l but different temporal profiles

### 13.3 Layer Size Confound

**Problem:** Larger layers have more parameters → potentially larger gradient norms and different curvature properties.

**Control:**
- Normalize D_l by number of parameters: D̃_l = D_l / |W_l|
- Use relative metrics throughout
- Compare layers of similar size

### 13.4 Batch Effects

**Problem:** Different mini-batches produce different gradient estimates.

**Control:**
- Full-batch gradients for curvature checkpoints
- Multiple mini-batch samples, report mean and variance
- Same random seed controls across conditions

### 13.5 Initialization Effects

**Problem:** Different initializations may lead to different trajectories.

**Control:**
- Multiple seeds (≥10 for confirmatory experiments)
- Report variance across seeds
- Test whether effects are consistent across initializations

### 13.6 Piecewise Linearity (ReLU Networks)

**Problem:** ReLU networks are piecewise linear; differential curvature is technically undefined except at boundaries.

**Control:**
- Compare ReLU vs. smooth activations (tanh, GELU)
- Note that Sekmen-Bilgin method uses discrete curvature (angles between subspaces), which is well-defined for piecewise linear manifolds
- Document any discrepancies between activation types

---

## 14. Unit Test Plan

### 14.1 Dissipation Computation Tests

```
test_dissipation_gradient_flow():
    """Verify D_l = ||grad L||^2 for vanilla GD with small step."""
    - Create simple quadratic loss
    - Compute gradient, compute D
    - Check D ≈ -ΔL/η within tolerance
    
test_dissipation_layer_decomposition():
    """Verify sum of layer dissipations equals total."""
    - D = sum(D_l for all layers)
    - Check within floating point tolerance
    
test_dissipation_nonnegativity():
    """D_l ≥ 0 always."""
```

### 14.2 Curvature Computation Tests

```
test_curvature_flat_manifold():
    """Curvature of points on a hyperplane should be ~0."""
    
test_curvature_sphere():
    """Curvature of points on sphere of radius r should be ~1/r."""
    
test_curvature_known_geometry():
    """Compare against analytic curvature of ellipsoid."""
    - Use equation from Kaufman-Azencot paper
    
test_weighted_angle_vs_principal_angle():
    """Weighted angle should differ from principal angle."""
    - Construct example from Sekmen-Bilgin Fig. 4
    - Principal angles: 0° and 90° regardless of data spread
    - Weighted angles: should capture spread
```

### 14.3 Jacobian/Metric Tests

```
test_jacobian_finite_difference():
    """Compare autograd Jacobian to finite differences."""
    
test_metric_symmetric():
    """g = J^T J should be symmetric."""
    
test_metric_positive_semidefinite():
    """g should have non-negative eigenvalues."""
```

### 14.4 Integration Tests

```
test_checkpoint_save_load():
    """Verify checkpoint contains all required variables."""
    
test_determinism():
    """Same seed produces identical results."""
    
test_gradient_not_modified():
    """Metric computation does not affect model gradients."""
```

### 14.5 Numerical Validation

```
test_gradient_flow_dissipation():
    """For small η, verify -ΔL ≈ η·D."""
    - Train for 1 epoch with η=1e-6
    - Check |ΔL + η·D| / |ΔL| < 0.01
```

---

## 15. Proposed Figures

### 15.1 Core Result Figures

**Figure 1: Layer-Time Heatmaps**
- Panel A: D_l(t) heatmap (layer × epoch, color = dissipation)
- Panel B: K_l(t) heatmap (layer × epoch, color = curvature)
- Panel C: ΔK_l(t) heatmap (layer × epoch, color = curvature change)

**Figure 2: Temporal Dynamics**
- Panel A: D_l(t) trajectories for each layer (one line per layer)
- Panel B: K_l(t) trajectories for each layer
- Panel C: Phase plot K_l vs D_l colored by time

**Figure 3: Lagged Correlation Analysis**
- Panel A: Cross-correlation Corr(D_l(t), ΔK_l(t+τ)) vs τ for each layer
- Panel B: Optimal lag τ* by layer
- Panel C: Correlation strength at optimal lag

**Figure 4: Robustness Relationship**
- Panel A: Final curvature K_l(T) vs robustness R(T)
- Panel B: Cumulative dissipation E_l^diss(T) vs robustness
- Panel C: R² decomposition: curvature-only, dissipation-only, combined

### 15.2 Control Figures

**Figure 5: Activation Function Comparison**
- MAPC profiles for ReLU, tanh, GELU networks

**Figure 6: Optimizer Comparison**
- Dissipation-curvature relationship for SGD, SGD+momentum, Adam

**Figure 7: Learning Rate Effects**
- How LR affects D_l-K_l relationship

### 15.3 Validation Figures

**Figure 8: Reproduce Sekmen-Bilgin**
- Curvature progression through layers (compare to their Fig. 7)

**Figure 9: Reproduce Kaufman-Azencot**
- MAPC profile (compare to their Fig. 3)

---

## 16. Minimum Reproducible Experiment

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Network | MLP: 784→256→128→64→10 |
| Activation | ReLU (primary), tanh (comparison) |
| Optimizer | SGD (no momentum) |
| Learning rate | 0.01 |
| Batch size | 128 |
| Epochs | 50 |
| Seeds | 5 (debugging), 10 (confirmatory) |

### Checkpoint Schedule

- Epochs 1–10: every epoch
- Epochs 10–30: every 2 epochs
- Epochs 30–50: every 5 epochs
- Total: ~25 checkpoints

### Measurements per Checkpoint

1. Full-batch gradient computation → D_l(t) for all layers
2. Curvature estimation (1000 training samples, N_i=20 neighbors, d_i=2)
3. Validation accuracy

### Measurements at Final Checkpoint

1. Clean test accuracy
2. Noise robustness (Gaussian σ ∈ {0.1, 0.2, 0.3, 0.5})
3. FGSM attack (ε = 0.1, 0.2, 0.3)

### Expected Outputs

1. Heatmaps of D_l(t) and K_l(t)
2. Lagged correlation analysis
3. Robustness correlation
4. Statistical summary with confidence intervals

### Success Criteria

- Curvature measurement reproduces qualitative trends from published work
- Statistically significant correlation between dissipation and curvature change
- Dissipation provides predictive information about robustness

---

## 17. Prior Literature on Dissipation-Curvature Relationship

### 17.1 Direct Prior Work

**Finding:** No published work directly investigates the relationship between layer-wise gradient norm squared (as dissipation proxy) and manifold curvature evolution during training.

This appears to be a novel research direction.

### 17.2 Related Work

**Gradient dynamics and geometry:**
- Cohen et al. (2021) "Edge of Stability": Gradient descent dynamics are constrained by loss Hessian curvature, but this is loss-landscape curvature, not representation-manifold curvature.
- The edge-of-stability phenomenon shows that ‖∇L‖² dynamics are coupled to second-order loss properties.

**Manifold curvature evolution:**
- Kaufman & Azencot (2023): Show curvature profile is "step-like" across layers, but do not study temporal evolution during training.
- Poole et al. (2016): Find that data manifold curvature increases with depth in random networks.

**Gradient magnitude and training:**
- Gradient magnitude is commonly used for learning rate scheduling, but not connected to representation geometry.
- Neural Tangent Kernel (NTK) literature connects gradient dynamics to representation but in the infinite-width limit.

**Thermodynamic interpretations:**
- Stochastic thermodynamics of SGD has been studied, but not connected to representation geometry.
- "Entropy production" in SGLD is related to gradient magnitude but focused on sampling, not geometry.

### 17.3 Gap This Project Addresses

The proposed research bridges:
1. **Optimization dynamics** (gradient magnitudes, dissipation)
2. **Representation geometry** (manifold curvature)
3. **Model properties** (robustness)

This three-way connection has not been systematically studied.

---

## 18. Critical Challenges to the Framework

### 18.1 Challenges to the Dissipation-Curvature Hypothesis

**Challenge 1: Correlation ≠ Causation**

The hypothesis proposes: D_l(t) → ΔK_l(t+τ)

Even if correlation exists, alternatives include:
- Both caused by a third variable (e.g., distance from optimum)
- Reverse causation: curvature constrains gradient magnitude
- Coincidental correlation due to training dynamics

**Mitigation:** 
- Temporal ordering (lag analysis)
- Interventional experiments (modify D_l directly)
- Multiple controls

**Challenge 2: Curvature May Not Be Primary Driver of Robustness**

Robustness could depend on:
- Margin (not curvature)
- Local Lipschitz constant
- Decision boundary properties (not data manifold properties)
- Spectral properties of weight matrices

**Mitigation:**
- Record multiple robustness-related metrics
- Compare curvature-based prediction to alternatives
- Be cautious about causal claims

**Challenge 3: Discrete vs. Continuous Time**

The gradient-flow interpretation is for continuous time. Real training is discrete.

**Mitigation:**
- Use small learning rates where continuous approximation is better
- Record both D_l and -ΔL/η to detect discrepancies
- Document where discrete effects dominate

### 18.2 Challenges to Curvature Measurement

**Challenge 4: Curvature in High Dimensions Is Ill-Defined**

True Riemannian curvature requires smooth manifolds. Neural network representations are:
- High-dimensional
- Sparse
- Potentially not smooth (ReLU)

**Mitigation:**
- Use empirical curvature measures (weighted angles) that don't assume smoothness
- Document limitations explicitly
- Compare multiple curvature definitions

**Challenge 5: Neighborhood Definition Is Arbitrary**

Curvature depends on:
- Number of neighbors N_i
- Local dimension d_i
- Distance metric

Different choices may yield different results.

**Mitigation:**
- Sensitivity analysis over hyperparameters
- Report results for multiple settings
- Follow published conventions for comparability

### 18.3 Challenges to Experimental Design

**Challenge 6: Small-Scale Results May Not Generalize**

MNIST + MLP is a toy problem. Results may not apply to:
- Larger datasets (ImageNet)
- Modern architectures (transformers)
- Different domains (NLP, RL)

**Mitigation:**
- Document scope explicitly
- Phase II expansion to CIFAR-10, CNNs
- Propose but do not claim generalization

**Challenge 7: Multiple Comparisons Problem**

Testing many hypotheses (layers × lags × phases × conditions) inflates false positive risk.

**Mitigation:**
- Pre-register primary hypotheses
- Apply multiple comparison corrections
- Separate exploratory from confirmatory analyses

### 18.4 Mathematical Corrections Needed

**Issue 1: Terminology**

The request uses "dissipation" but cautions against "heat." This is appropriate. The current terminology is:
- "Layer-wise optimization dissipation proxy" ✓
- NOT "heat" ✓
- NOT asserting thermodynamic theory ✓

**Issue 2: Smoothing Quantity**

The proposed smoothing quantity S_l(t) = -Δ|K_l(t)| assumes curvature magnitude is meaningful. This is valid for the weighted-angle definition (angles are non-negative and interpretable).

**Issue 3: Layer-Wise vs. Class-Wise Curvature**

Sekmen-Bilgin compute curvature **per class** and average. This should be preserved:

   K_l(t) = (1/C) Σ_{c=1}^C K_l^c(t)

rather than computing curvature over the entire layer activations ignoring class structure.

---

## Approval Checklist

Before proceeding to implementation, please confirm:

- [ ] The cited papers are correctly identified
- [ ] The curvature equations are correctly transcribed
- [ ] The dissipation interpretation is mathematically sound
- [ ] The limitations for discrete optimization are adequately documented
- [ ] The experimental design is appropriate for the research questions
- [ ] The statistical analysis plan is rigorous
- [ ] The scope (Phase I: MNIST/MLP) is appropriate
- [ ] Potential confounds are adequately addressed
- [ ] The unit test plan covers critical functionality
- [ ] Resource estimates are realistic

---

**Document prepared by:** Claude (AI Research Assistant)  
**Awaiting approval before Phase I implementation**
