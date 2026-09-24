"""
Visualize the dissipation-curvature relationship from test run.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_path = Path(__file__).parent / "results" / "test_run_results.json"
with open(results_path) as f:
    results = json.load(f)

analysis = results['analysis']
metrics = results['metrics']

# Create figure with multiple panels
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Panel 1: Training curves
ax = axes[0, 0]
epochs = metrics['epochs']
ax.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax2 = ax.twinx()
ax2.plot(epochs, metrics['train_acc'], 'g--', label='Train Acc', linewidth=2)
ax2.plot(epochs, metrics['test_acc'], 'r--', label='Test Acc', linewidth=2)
ax2.set_ylabel('Accuracy', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax.set_title('Training Progress')
ax.legend(loc='upper left')
ax2.legend(loc='right')

# Panel 2: Dissipation Evolution by Layer
ax = axes[0, 1]
curv_epochs = analysis['summary']['epochs']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, layer in enumerate(['layer1', 'layer2', 'layer3', 'output']):
    D = analysis[layer]['D_l']
    ax.plot(curv_epochs, D, 'o-', color=colors[i], label=layer, linewidth=2, markersize=8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Dissipation $D_l = ||\\nabla W_l||_F^2$')
ax.set_title('Layer-wise Dissipation Evolution')
ax.legend()
ax.set_yscale('log')

# Panel 3: Curvature Evolution by Layer
ax = axes[0, 2]
for i, layer in enumerate(['layer1', 'layer2', 'layer3', 'output']):
    K = analysis[layer]['K_l']
    ax.plot(curv_epochs, K, 's-', color=colors[i], label=layer, linewidth=2, markersize=8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Curvature $K_l$ (weighted angle)')
ax.set_title('Layer-wise Curvature Evolution')
ax.legend()

# Panel 4: D vs K scatter (all layers combined)
ax = axes[1, 0]
for i, layer in enumerate(['layer1', 'layer2', 'layer3']):
    D = analysis[layer]['D_l']
    K = analysis[layer]['K_l']
    ax.scatter(D, K, c=colors[i], label=layer, s=100, alpha=0.7)
    # Add time arrows
    for j in range(len(D)-1):
        ax.annotate('', xy=(D[j+1], K[j+1]), xytext=(D[j], K[j]),
                   arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.5))
ax.set_xlabel('Dissipation $D_l$')
ax.set_ylabel('Curvature $K_l$')
ax.set_title('Dissipation vs Curvature Phase Space')
ax.legend()

# Panel 5: Correlation bar chart
ax = axes[1, 1]
layers = ['layer1', 'layer2', 'layer3', 'output']
corr_D_K = [analysis[l]['corr_D_K'] for l in layers]
corr_D_dK = [analysis[l]['corr_D_deltaK'] for l in layers]

x = np.arange(len(layers))
width = 0.35
bars1 = ax.bar(x - width/2, corr_D_K, width, label='Corr($D_l$, $K_l$)', color='steelblue')
bars2 = ax.bar(x + width/2, corr_D_dK, width, label='Corr($D_l$, $\\Delta K_l$)', color='coral')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Correlation')
ax.set_title('Dissipation-Curvature Correlations')
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend()
ax.set_ylim(-1, 1)

# Panel 6: Delta K vs D (lagged relationship)
ax = axes[1, 2]
for i, layer in enumerate(['layer1', 'layer2', 'layer3']):
    D = analysis[layer]['D_l'][:-1]  # D at time t
    dK = analysis[layer]['delta_K']   # Delta K at time t+1
    ax.scatter(D, dK, c=colors[i], label=layer, s=100, alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Dissipation $D_l(t)$')
ax.set_ylabel('Curvature Change $\\Delta K_l(t+\\tau)$')
ax.set_title('Dissipation vs Subsequent Curvature Change')
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'results' / 'dissipation_curvature_analysis.png', 
            dpi=150, bbox_inches='tight')
plt.savefig(Path(__file__).parent / 'results' / 'dissipation_curvature_analysis.pdf', 
            bbox_inches='tight')
print(f"Figures saved to {Path(__file__).parent / 'results'}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY: Dissipation-Curvature Relationship")
print("="*60)

print("\n1. Key Finding: NEGATIVE correlation between D and K")
print("   - Higher dissipation is associated with LOWER curvature")
print("   - This suggests dissipation 'flattens' the manifold")

print("\n2. Lagged Relationship: POSITIVE Corr(D, ΔK)")
print("   - But wait - positive correlation with ΔK means:")
print("   - Higher D at time t → LESS NEGATIVE ΔK at time t+τ")
print("   - i.e., high dissipation SLOWS DOWN curvature decrease")

print("\n3. Interpretation:")
print("   - Early training: Low D, high K, K decreasing rapidly")
print("   - Mid training: High D, K stabilizing")  
print("   - Late training: D decreasing, K stable")

print("\n4. Layer-wise pattern:")
for layer in ['layer1', 'layer2', 'layer3']:
    K = analysis[layer]['K_l']
    D = analysis[layer]['D_l']
    print(f"   {layer}: K dropped {K[0]-K[-1]:.3f} ({100*(K[0]-K[-1])/K[0]:.1f}%), "
          f"D increased {D[-1]/D[0]:.1f}x")

print("\n" + "="*60)
