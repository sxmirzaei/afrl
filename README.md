# Optimization Dissipation, Manifold Curvature, and Neural-Network Robustness

Research project investigating whether layer-wise optimization dissipation during neural-network training is systematically related to the evolution of representation-manifold curvature and model robustness.

## Research Question

**How is layer-wise optimization dissipation related to the temporal evolution of representation-manifold curvature during neural-network training?**

### Conceptual Relationship (Hypothesis)

```
Optimization Dissipation → Geometric Reorganization → Manifold Curvature → Robustness
```

This relationship is a **hypothesis to test**, not an assumption.

## Key Concepts

### Optimization Dissipation Proxy

For gradient flow dynamics, the layer-wise dissipation is defined as:

```
D_l(t) = ||∇_{W_l} L(t)||_F²
```

This measures the optimization "activity" at each layer during training.

### Manifold Curvature

Using the weighted-angle method from Sekmen & Bilgin (2024), we estimate the curvature of representation manifolds at each layer. Lower curvature is associated with higher robustness.

## Project Structure

```
├── TECHNICAL_DESIGN_REPORT.md   # Full technical specification
├── experiments/
│   └── test_run/                # Initial validation experiment
│       ├── simple_test.py       # Main test script
│       ├── visualize_results.py # Analysis and plotting
│       └── results/             # Output data and figures
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install torch torchvision scipy numpy matplotlib

# Run the test experiment
python experiments/test_run/simple_test.py

# Generate visualization
python experiments/test_run/visualize_results.py
```

## Initial Results

Test run on MNIST (5000 samples, 20 epochs) shows:

| Finding | Result |
|---------|--------|
| Corr(D_l, K_l) | **Negative** (-0.85 to -0.64) |
| Corr(D_l, ΔK_l) | **Positive** (+0.73 to +0.88) |
| Curvature change | Decreases 14-29% during training |

**Interpretation:** Higher dissipation is associated with lower curvature. The positive correlation with curvature *change* suggests high dissipation stabilizes (slows) the curvature decrease.

## References

- Sekmen, A. & Bilgin, B. (2024). Manifold-based approach for neural network robustness analysis. *Communications Engineering*, 3, 118. https://doi.org/10.1038/s44172-024-00263-8

- Kaufman, I. & Azencot, O. (2023). Data Representations' Study of Latent Image Manifolds. *ICML 2023*. https://proceedings.mlr.press/v202/kaufman23a.html

## Status

**Phase I: Initial Validation** - Complete  
**Phase II: Full MNIST Experiments** - Pending  
**Phase III: CIFAR-10 Extension** - Pending
