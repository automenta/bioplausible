"""
Diagnostic test: Compare NumPy kernel vs PyTorch LoopedMLP

Purpose: Understand why kernel shows 14.5% accuracy vs PyTorch 31.9%
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

root_path = Path(__file__).parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from models import LoopedMLP
from models.kernel import EqPropKernelBPTT

def create_simple_dataset(n_samples=200, input_dim=64, output_dim=10, seed=42):
    """Create simple separable dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, output_dim, n_samples)
    
    X_torch = torch.from_numpy(X)
    y_torch = torch.from_numpy(y)
    
    return X, y, X_torch, y_torch


def test_architecture_match():
    """Test if kernel and PyTorch use same computation."""
    print("="*70)
    print("TEST 1: Architecture Equivalence")
    print("="*70)
    
    input_dim, hidden_dim, output_dim = 64, 128, 10
    
    # Create models
    torch_model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=False, max_steps=30)
    kernel = EqPropKernelBPTT(input_dim, hidden_dim, output_dim, max_steps=30, lr=0.01)
    
    # Copy weights from PyTorch to kernel
    # PyTorch Linear: weight is [out_features, in_features], stored as W.T for x @ W.T
    # Kernel expects: W is [out_features, in_features] for x @ W.T
    with torch.no_grad():
        # W_in: maps input_dim -> hidden_dim
        # PyTorch: [hidden_dim, input_dim]
        # Kernel: needs [hidden_dim, input_dim]
        kernel.W_in = torch_model.W_in.weight.detach().cpu().numpy().copy()
        
        # W_rec: maps hidden_dim -> hidden_dim  
        # PyTorch: [hidden_dim, hidden_dim]
        # Kernel: needs [hidden_dim, hidden_dim]
        kernel.W_rec = torch_model.W_rec.weight.detach().cpu().numpy().copy()
        
        # W_out: maps hidden_dim -> output_dim
        # PyTorch: [output_dim, hidden_dim]
        # Kernel: needs [output_dim, hidden_dim]
        kernel.W_out = torch_model.W_out.weight.detach().cpu().numpy().copy()
        
        # Biases
        if torch_model.W_in.bias is not None:
            kernel.b_in = torch_model.W_in.bias.detach().cpu().numpy().copy()
        else:
            kernel.b_in = np.zeros(hidden_dim, dtype=np.float32)
            
        if torch_model.W_out.bias is not None:
            kernel.b_out = torch_model.W_out.bias.detach().cpu().numpy().copy()
        else:
            kernel.b_out = np.zeros(output_dim, dtype=np.float32)

    
    # Test on same input
    x = np.random.randn(32, input_dim).astype(np.float32)
    x_torch = torch.from_numpy(x)
    
    # Forward pass
    with torch.no_grad():
        out_torch = torch_model(x_torch).cpu().numpy()
    out_kernel, _ = kernel.forward(x)
    
    diff = np.abs(out_torch - out_kernel).max()
    print(f"Max output difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("✅ Architectures match!")
        return True
    else:
        print(f"❌ Architectures differ by {diff}")
        print(f"   PyTorch output sample: {out_torch[0, :5]}")
        print(f"   Kernel output sample:  {out_kernel[0, :5]}")
        return False


def test_training_convergence():
    """Test if both implementations learn equally well."""
    print("\n" + "="*70)
    print("TEST 2: Training Convergence")
    print("="*70)
    
    input_dim, hidden_dim, output_dim = 64, 128, 10
    
    # Create datasets
    X, y, X_torch, y_torch = create_simple_dataset(200, input_dim, output_dim)
    
    # PyTorch model
    torch_model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=False, max_steps=30)
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
    
    # Kernel model with SAME hyperparameters
    kernel = EqPropKernelBPTT(input_dim, hidden_dim, output_dim, max_steps=30, lr=0.01)
    
    epochs = 50
    
    print("\nTraining PyTorch...")
    torch_losses = []
    torch_accs = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = torch_model(X_torch)
        loss = F.cross_entropy(out, y_torch)
        loss.backward()
        optimizer.step()
        
        acc = (out.argmax(dim=1) == y_torch).float().mean().item()
        torch_losses.append(loss.item())
        torch_accs.append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc*100:.1f}%")
    
    print("\nTraining Kernel...")
    kernel_losses = []
    kernel_accs = []
    for epoch in range(epochs):
        result = kernel.train_step(X, y)
        kernel_losses.append(result['loss'])
        kernel_accs.append(result['accuracy'])
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={result['loss']:.4f}, acc={result['accuracy']*100:.1f}%")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"PyTorch final: loss={torch_losses[-1]:.4f}, acc={torch_accs[-1]*100:.1f}%")
    print(f"Kernel final:  loss={kernel_losses[-1]:.4f}, acc={kernel_accs[-1]*100:.1f}%")
    
    acc_gap = abs(torch_accs[-1] - kernel_accs[-1]) * 100
    if acc_gap < 5:
        print(f"✅ Accuracy gap ({acc_gap:.1f}%) is acceptable")
        return True
    else:
        print(f"❌ Large accuracy gap: {acc_gap:.1f}%")
        return False


def test_gradient_correctness():
    """Test if BPTT gradients are computed correctly."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Correctness")
    print("="*70)
    
    input_dim, hidden_dim, output_dim = 64, 32, 10
    
    # Create models with same weights
    torch_model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=False, max_steps=5)
    kernel = EqPropKernelBPTT(input_dim, hidden_dim, output_dim, max_steps=5, lr=0.01)
    
    # Copy weights
    with torch.no_grad():
        kernel.W_in = torch_model.W_in.weight.detach().cpu().numpy().copy()
        kernel.W_rec = torch_model.W_rec.weight.detach().cpu().numpy().copy()
        kernel.W_out = torch_model.W_out.weight.detach().cpu().numpy().copy()
        kernel.b_in = torch_model.W_in.bias.detach().cpu().numpy().copy() if torch_model.W_in.bias is not None else np.zeros(hidden_dim, dtype=np.float32)
        kernel.b_out = torch_model.W_out.bias.detach().cpu().numpy().copy() if torch_model.W_out.bias is not None else np.zeros(output_dim, dtype=np.float32)
    
    # Test gradient computation
    x = np.random.randn(16, input_dim).astype(np.float32)
    y = np.random.randint(0, output_dim, 16)
    
    x_torch = torch.from_numpy(x)
    y_torch = torch.from_numpy(y)
    
    # PyTorch gradients
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
    optimizer.zero_grad()
    out_torch = torch_model(x_torch)
    loss_torch = F.cross_entropy(out_torch, y_torch)
    loss_torch.backward()
    
    grad_W_out_torch = torch_model.W_out.weight.grad.cpu().numpy()
    
    # Kernel gradients
    logits, trajectory = kernel.forward(x)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(16), y] = 1.0
    d_logits = probs - one_hot
    
    grads = kernel.backward(x, trajectory, d_logits)
    grad_W_out_kernel = grads['dW_out']
    
    # Compare
    diff = np.abs(grad_W_out_torch - grad_W_out_kernel).max()
    print(f"Max W_out gradient difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("✅ Gradients match!")
        return True
    else:
        print(f"❌ Gradients differ by {diff}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("KERNEL DIAGNOSTIC TEST SUITE")
    print("="*70 + "\n")
    
    results = {}
    
    # Test 1: Architecture match
    results['architecture'] = test_architecture_match()
    
    # Test 2: Training convergence
    results['training'] = test_training_convergence()
    
    # Test 3: Gradient correctness
    results['gradients'] = test_gradient_correctness()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize():20s}: {status}")
    
    all_pass = all(results.values())
    print("\n" + ("="*70))
    if all_pass:
        print("✅ ALL TESTS PASSED - Kernel matches PyTorch")
    else:
        print("❌ SOME TESTS FAILED - Kernel implementation needs fixing")
    print("="*70)
