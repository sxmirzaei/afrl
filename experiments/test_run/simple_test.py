"""
Simple Test Run: Optimization Dissipation and Manifold Curvature

This script validates the core concepts on a minimal example:
- Small MLP on MNIST (subset)
- Track layer-wise dissipation D_l(t) = ||grad W_l||^2
- Estimate manifold curvature using weighted angles (Sekmen-Bilgin method)
- Check for temporal relationship between dissipation and curvature

This is a VALIDATION run, not the full experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# Model Definition
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP: 784 -> 256 -> 128 -> 64 -> 10"""
    
    def __init__(self, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'gelu':
            self.act = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Store activations for curvature computation
        self.activations = {}
    
    def forward(self, x, store_activations=False):
        x = x.view(-1, 784)
        
        if store_activations:
            self.activations['input'] = x.detach()
        
        x = self.act(self.fc1(x))
        if store_activations:
            self.activations['layer1'] = x.detach()
        
        x = self.act(self.fc2(x))
        if store_activations:
            self.activations['layer2'] = x.detach()
        
        x = self.act(self.fc3(x))
        if store_activations:
            self.activations['layer3'] = x.detach()
        
        x = self.fc4(x)
        if store_activations:
            self.activations['output'] = x.detach()
        
        return x
    
    def get_layer_params(self) -> Dict[str, nn.Parameter]:
        """Return dict of layer name -> weight parameter"""
        return {
            'layer1': self.fc1.weight,
            'layer2': self.fc2.weight,
            'layer3': self.fc3.weight,
            'output': self.fc4.weight,
        }


# =============================================================================
# Dissipation Computation
# =============================================================================

def compute_layer_dissipation(model: SimpleMLP) -> Dict[str, float]:
    """
    Compute layer-wise dissipation: D_l = ||grad W_l||_F^2
    
    Must be called AFTER loss.backward()
    """
    dissipation = {}
    for name, param in model.get_layer_params().items():
        if param.grad is not None:
            dissipation[name] = (param.grad ** 2).sum().item()
        else:
            dissipation[name] = 0.0
    
    dissipation['total'] = sum(dissipation.values())
    return dissipation


def compute_parameter_displacement(
    old_params: Dict[str, torch.Tensor],
    new_params: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute ||W_l(t+1) - W_l(t)||_F"""
    displacement = {}
    for name in old_params:
        diff = new_params[name] - old_params[name]
        displacement[name] = torch.norm(diff, p='fro').item()
    return displacement


# =============================================================================
# Curvature Computation (Sekmen-Bilgin Weighted Angle Method)
# =============================================================================

def compute_weighted_angle(M_i: np.ndarray, M_j: np.ndarray, d: int) -> float:
    """
    Compute weighted angle between two local subspaces.
    
    Sekmen-Bilgin Equation (2):
    θ_ij = arccos(tr(Σ_Q) / tr(Σ_i^T Σ_j))
    
    Args:
        M_i: Data matrix for subspace i (D x K)
        M_j: Data matrix for subspace j (D x K)
        d: Local subspace dimension
    
    Returns:
        Weighted angle in radians
    """
    # SVD of data matrices
    U_i, S_i, _ = np.linalg.svd(M_i, full_matrices=False)
    U_j, S_j, _ = np.linalg.svd(M_j, full_matrices=False)
    
    # Truncate to d dimensions
    U_i = U_i[:, :d]
    S_i = S_i[:d]
    U_j = U_j[:, :d]
    S_j = S_j[:d]
    
    # Q = (U_i Σ_i)^T (U_j Σ_j)
    US_i = U_i * S_i  # Broadcasting: each column scaled by singular value
    US_j = U_j * S_j
    Q = US_i.T @ US_j
    
    # SVD of Q
    _, S_Q, _ = np.linalg.svd(Q)
    
    # Compute weighted angle
    tr_S_Q = np.sum(S_Q)
    tr_SiSj = np.sum(S_i * S_j)  # tr(Σ_i^T Σ_j) = sum of products
    
    if tr_SiSj < 1e-10:
        return 0.0
    
    cos_theta = np.clip(tr_S_Q / tr_SiSj, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    return theta


def estimate_manifold_curvature(
    activations: np.ndarray,
    n_neighbors: int = 20,
    local_dim: int = 2,
    n_samples: int = 200
) -> Tuple[float, np.ndarray]:
    """
    Estimate manifold curvature using Sekmen-Bilgin weighted angle method.
    
    Args:
        activations: (N, D) array of activations
        n_neighbors: Number of neighbors for local subspace
        local_dim: Dimension of local subspaces
        n_samples: Number of points to sample for estimation
    
    Returns:
        mean_curvature: Average weighted angle (curvature estimate)
        point_curvatures: Per-point curvature estimates
    """
    N, D = activations.shape
    
    # Subsample if needed
    if N > n_samples:
        indices = np.random.choice(N, n_samples, replace=False)
        activations = activations[indices]
        N = n_samples
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(activations, activations, metric='euclidean')
    
    point_curvatures = []
    
    for i in range(N):
        # Find k nearest neighbors (excluding self)
        neighbor_idx = np.argsort(distances[i])[1:n_neighbors+1]
        
        # Create local data matrix centered at mean of neighbors
        neighbors = activations[neighbor_idx]
        mu_i = neighbors.mean(axis=0)
        M_i = (neighbors - mu_i).T  # D x K
        
        # Compute weighted angles to each neighbor's local subspace
        angles = []
        for j in neighbor_idx:
            # Get j's neighbors
            j_neighbor_idx = np.argsort(distances[j])[1:n_neighbors+1]
            j_neighbors = activations[j_neighbor_idx]
            mu_j = j_neighbors.mean(axis=0)
            M_j = (j_neighbors - mu_j).T
            
            # Compute weighted angle
            try:
                angle = compute_weighted_angle(M_i, M_j, local_dim)
                angles.append(angle)
            except:
                continue
        
        if angles:
            point_curvatures.append(np.mean(angles))
    
    point_curvatures = np.array(point_curvatures)
    mean_curvature = np.mean(point_curvatures) if len(point_curvatures) > 0 else 0.0
    
    return mean_curvature, point_curvatures


def compute_layer_curvatures(
    model: SimpleMLP,
    data_loader: DataLoader,
    n_samples: int = 500,
    n_neighbors: int = 15,
    local_dim: int = 2
) -> Dict[str, float]:
    """
    Compute curvature for each layer's activation manifold.
    """
    model.eval()
    
    # Collect activations
    all_activations = {name: [] for name in ['input', 'layer1', 'layer2', 'layer3', 'output']}
    
    n_collected = 0
    with torch.no_grad():
        for data, _ in data_loader:
            if n_collected >= n_samples:
                break
            
            data = data.to(DEVICE)
            _ = model(data, store_activations=True)
            
            for name, act in model.activations.items():
                all_activations[name].append(act.cpu().numpy())
            
            n_collected += data.size(0)
    
    # Concatenate and compute curvature
    curvatures = {}
    for name in all_activations:
        acts = np.concatenate(all_activations[name], axis=0)[:n_samples]
        
        # Adjust local_dim based on activation dimension
        act_dim = acts.shape[1]
        d = min(local_dim, act_dim - 1, 5)
        
        if d < 1:
            curvatures[name] = 0.0
            continue
        
        mean_curv, _ = estimate_manifold_curvature(
            acts, 
            n_neighbors=min(n_neighbors, len(acts) - 1),
            local_dim=d,
            n_samples=min(n_samples, len(acts))
        )
        curvatures[name] = mean_curv
    
    model.train()
    return curvatures


# =============================================================================
# Training Loop with Metric Collection
# =============================================================================

def train_and_collect_metrics(
    model: SimpleMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    curvature_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 0.01,
    checkpoint_freq: int = 2,
    curvature_freq: int = 5
) -> Dict:
    """
    Train model and collect dissipation/curvature metrics.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'dissipation': [],  # Per-layer dissipation
        'displacement': [],  # Per-layer parameter displacement
        'curvature': [],  # Per-layer curvature (computed less frequently)
        'curvature_epochs': [],  # Which epochs have curvature
    }
    
    # Store initial parameters
    prev_params = {name: param.data.clone() for name, param in model.get_layer_params().items()}
    
    print(f"\n{'='*60}")
    print(f"Starting training: {n_epochs} epochs, lr={lr}")
    print(f"Curvature computed every {curvature_freq} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_dissipation = {name: 0.0 for name in ['layer1', 'layer2', 'layer3', 'output', 'total']}
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate dissipation before optimizer step
            batch_dissipation = compute_layer_dissipation(model)
            for name in epoch_dissipation:
                epoch_dissipation[name] += batch_dissipation[name]
            
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            epoch_correct += pred.eq(target).sum().item()
            epoch_total += data.size(0)
        
        # Compute epoch averages
        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total
        n_batches = len(train_loader)
        for name in epoch_dissipation:
            epoch_dissipation[name] /= n_batches
        
        # Compute parameter displacement
        current_params = {name: param.data.clone() for name, param in model.get_layer_params().items()}
        displacement = compute_parameter_displacement(prev_params, current_params)
        prev_params = current_params
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += data.size(0)
        test_acc = test_correct / test_total
        
        # Store metrics
        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)
        metrics['test_acc'].append(test_acc)
        metrics['dissipation'].append(epoch_dissipation)
        metrics['displacement'].append(displacement)
        
        # Compute curvature periodically
        if epoch % curvature_freq == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}: Computing curvature...")
            curvatures = compute_layer_curvatures(
                model, curvature_loader, 
                n_samples=300, n_neighbors=15, local_dim=2
            )
            metrics['curvature'].append(curvatures)
            metrics['curvature_epochs'].append(epoch)
            
            print(f"  Epoch {epoch}: Loss={epoch_loss:.4f}, TrainAcc={epoch_acc:.4f}, "
                  f"TestAcc={test_acc:.4f}")
            print(f"  Dissipation: {epoch_dissipation['total']:.6f}")
            print(f"  Curvature: layer1={curvatures['layer1']:.4f}, "
                  f"layer2={curvatures['layer2']:.4f}, "
                  f"layer3={curvatures['layer3']:.4f}")
        else:
            print(f"  Epoch {epoch}: Loss={epoch_loss:.4f}, TrainAcc={epoch_acc:.4f}, "
                  f"TestAcc={test_acc:.4f}, D={epoch_dissipation['total']:.6f}")
    
    return metrics


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_dissipation_curvature_relationship(metrics: Dict) -> Dict:
    """
    Analyze the relationship between dissipation and curvature.
    """
    results = {}
    
    # Extract time series
    epochs = np.array(metrics['epochs'])
    curvature_epochs = np.array(metrics['curvature_epochs'])
    
    # Get dissipation at curvature epochs
    dissipation_at_curv = []
    for e in curvature_epochs:
        idx = np.where(epochs == e)[0][0]
        dissipation_at_curv.append(metrics['dissipation'][idx])
    
    layers = ['layer1', 'layer2', 'layer3', 'output']
    
    for layer in layers:
        # Extract layer-specific time series
        D_l = np.array([d[layer] for d in dissipation_at_curv])
        K_l = np.array([c[layer] for c in metrics['curvature']])
        
        # Compute correlation
        if len(D_l) > 2 and np.std(D_l) > 0 and np.std(K_l) > 0:
            corr = np.corrcoef(D_l, K_l)[0, 1]
        else:
            corr = np.nan
        
        # Compute curvature change
        if len(K_l) > 1:
            delta_K = np.diff(K_l)
            D_l_lagged = D_l[:-1]  # D at time t
            
            if len(delta_K) > 1 and np.std(D_l_lagged) > 0 and np.std(delta_K) > 0:
                corr_lagged = np.corrcoef(D_l_lagged, delta_K)[0, 1]
            else:
                corr_lagged = np.nan
        else:
            delta_K = []
            corr_lagged = np.nan
        
        results[layer] = {
            'D_l': D_l.tolist(),
            'K_l': K_l.tolist(),
            'delta_K': delta_K.tolist() if len(delta_K) > 0 else [],
            'corr_D_K': corr,
            'corr_D_deltaK': corr_lagged,
        }
    
    # Total dissipation
    D_total = np.array([d['total'] for d in dissipation_at_curv])
    K_mean = np.array([np.mean([c[l] for l in layers[:3]]) for c in metrics['curvature']])
    
    results['summary'] = {
        'D_total': D_total.tolist(),
        'K_mean': K_mean.tolist(),
        'epochs': curvature_epochs.tolist(),
    }
    
    return results


def print_analysis_results(analysis: Dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*60)
    print("ANALYSIS: Dissipation-Curvature Relationship")
    print("="*60)
    
    print("\nPer-Layer Correlations:")
    print("-"*40)
    for layer in ['layer1', 'layer2', 'layer3', 'output']:
        r = analysis[layer]
        print(f"  {layer}:")
        print(f"    Corr(D_l, K_l):      {r['corr_D_K']:.4f}" if not np.isnan(r['corr_D_K']) else f"    Corr(D_l, K_l):      N/A")
        print(f"    Corr(D_l, ΔK_l):     {r['corr_D_deltaK']:.4f}" if not np.isnan(r['corr_D_deltaK']) else f"    Corr(D_l, ΔK_l):     N/A")
    
    print("\nDissipation Evolution:")
    print("-"*40)
    for layer in ['layer1', 'layer2', 'layer3']:
        D = analysis[layer]['D_l']
        if len(D) > 1:
            print(f"  {layer}: D_start={D[0]:.6f} -> D_end={D[-1]:.6f} (ratio: {D[-1]/D[0]:.2f}x)")
    
    print("\nCurvature Evolution:")
    print("-"*40)
    for layer in ['layer1', 'layer2', 'layer3']:
        K = analysis[layer]['K_l']
        if len(K) > 1:
            print(f"  {layer}: K_start={K[0]:.4f} -> K_end={K[-1]:.4f} (change: {K[-1]-K[0]:+.4f})")
    
    print("\n" + "="*60)


# =============================================================================
# Validation Tests
# =============================================================================

def validate_dissipation_identity(model: SimpleMLP, data_loader: DataLoader, lr: float = 0.001):
    """
    Validate: For small lr, -ΔL ≈ η * ||∇L||²
    
    This tests the gradient flow dissipation identity.
    """
    print("\n" + "="*60)
    print("VALIDATION: Gradient Flow Dissipation Identity")
    print("="*60)
    
    model_copy = SimpleMLP().to(DEVICE)
    model_copy.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Get a batch
    data, target = next(iter(data_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)
    
    # Compute initial loss
    output = model_copy(data)
    loss_before = criterion(output, target).item()
    
    # Compute gradient and dissipation
    optimizer.zero_grad()
    output = model_copy(data)
    loss = criterion(output, target)
    loss.backward()
    
    dissipation = compute_layer_dissipation(model_copy)
    D_total = dissipation['total']
    
    # Take gradient step
    optimizer.step()
    
    # Compute new loss
    output = model_copy(data)
    loss_after = criterion(output, target).item()
    
    delta_L = loss_after - loss_before
    expected_delta_L = -lr * D_total
    
    print(f"\n  Learning rate η = {lr}")
    print(f"  Loss before:     {loss_before:.6f}")
    print(f"  Loss after:      {loss_after:.6f}")
    print(f"  Actual ΔL:       {delta_L:.6f}")
    print(f"  Expected -η*D:   {expected_delta_L:.6f}")
    print(f"  Relative error:  {abs(delta_L - expected_delta_L) / abs(delta_L + 1e-10):.4f}")
    
    if abs(delta_L - expected_delta_L) / (abs(delta_L) + 1e-10) < 0.1:
        print("\n  ✓ Dissipation identity validated (error < 10%)")
    else:
        print("\n  ⚠ Larger deviation - expected for non-infinitesimal lr")
    
    print("="*60)


def validate_curvature_on_sphere():
    """
    Validate curvature estimation on a known geometry: sphere.
    Sphere of radius r should have curvature ~1/r.
    """
    print("\n" + "="*60)
    print("VALIDATION: Curvature Estimation on Sphere")
    print("="*60)
    
    for r in [1.0, 2.0, 5.0]:
        # Generate points on sphere of radius r
        n_points = 500
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.random.uniform(0, np.pi, n_points)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points = np.stack([x, y, z], axis=1)
        
        # Estimate curvature
        curv, _ = estimate_manifold_curvature(
            points, n_neighbors=20, local_dim=2, n_samples=300
        )
        
        expected = 1.0 / r  # Gaussian curvature is 1/r^2, but weighted angle scales differently
        print(f"\n  Sphere radius={r}: estimated curvature={curv:.4f}")
    
    print("\n  Note: Weighted angle curvature is not directly 1/r,")
    print("  but should show inverse relationship with radius.")
    print("="*60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "#"*60)
    print("# SIMPLE TEST RUN: Dissipation-Curvature Relationship")
    print("#"*60)
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Load MNIST (subset for speed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Use subset for faster testing
    train_subset = Subset(train_dataset, range(5000))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=256)
    curvature_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Test samples: {len(test_subset)}")
    
    # Initialize model
    model = SimpleMLP(activation='relu').to(DEVICE)
    print(f"\nModel: SimpleMLP (784->256->128->64->10) with ReLU")
    
    # Run validations first
    validate_dissipation_identity(model, train_loader, lr=0.0001)
    validate_curvature_on_sphere()
    
    # Train and collect metrics
    metrics = train_and_collect_metrics(
        model, train_loader, test_loader, curvature_loader,
        n_epochs=20,
        lr=0.01,
        checkpoint_freq=1,
        curvature_freq=4  # Compute curvature every 4 epochs
    )
    
    # Analyze results
    analysis = analyze_dissipation_curvature_relationship(metrics)
    print_analysis_results(analysis)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_epochs': 20,
            'lr': 0.01,
            'batch_size': 64,
            'train_samples': len(train_subset),
            'seed': SEED,
        },
        'metrics': {
            'epochs': metrics['epochs'],
            'train_loss': metrics['train_loss'],
            'train_acc': metrics['train_acc'],
            'test_acc': metrics['test_acc'],
            'curvature_epochs': metrics['curvature_epochs'],
        },
        'analysis': analysis,
    }
    
    results_path = output_dir / 'test_run_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "#"*60)
    print("# TEST RUN COMPLETE")
    print("#"*60)
    print(f"\nFinal test accuracy: {metrics['test_acc'][-1]:.4f}")
    print(f"Training converged: {'Yes' if metrics['train_acc'][-1] > 0.9 else 'Needs more epochs'}")
    
    # Key finding
    print("\nKey observations:")
    for layer in ['layer1', 'layer2', 'layer3']:
        r = analysis[layer]
        if not np.isnan(r['corr_D_deltaK']):
            sign = "positive" if r['corr_D_deltaK'] > 0 else "negative"
            print(f"  - {layer}: Corr(D_l, ΔK_l) = {r['corr_D_deltaK']:.3f} ({sign})")


if __name__ == '__main__':
    main()
