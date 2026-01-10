"""
Kernel Benchmark: Comprehensive accuracy and speed comparison

Purpose: Determine if kernel should replace PyTorch for faster evaluation
"""

import time
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


def benchmark_training(input_dim, hidden_dim, output_dim, n_samples, epochs, max_steps, use_sn=False):
    """Benchmark training for both PyTorch and kernel."""
    
    # Create data
    np.random.seed(42)
    torch.manual_seed(42)
    
    X_np = np.random.randn(n_samples, input_dim).astype(np.float32)
    y_np = np.random.randint(0, output_dim, n_samples)
    X_torch = torch.from_numpy(X_np)
    y_torch = torch.from_numpy(y_np)
    
    n_test = n_samples // 5
    X_train_np, y_train_np = X_np[:-n_test], y_np[:-n_test]
    X_test_np, y_test_np = X_np[-n_test:], y_np[-n_test:]
    X_train_torch, y_train_torch = X_torch[:-n_test], y_torch[:-n_test]
    X_test_torch, y_test_torch = X_torch[-n_test:], y_torch[-n_test:]
    
    results = {}
    
    # PyTorch benchmark
    print("  PyTorch training...", end="", flush=True)
    torch_model = LoopedMLP(input_dim, hidden_dim, output_dim, 
                            use_spectral_norm=use_sn, max_steps=max_steps)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    
    torch_start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = torch_model(X_train_torch)
        loss = F.cross_entropy(out, y_train_torch)
        loss.backward()
        optimizer.step()
    torch_time = time.time() - torch_start
    
    with torch.no_grad():
        pt_train_acc = (torch_model(X_train_torch).argmax(dim=1) == y_train_torch).float().mean().item()
        pt_test_acc = (torch_model(X_test_torch).argmax(dim=1) == y_test_torch).float().mean().item()
    
    results['pytorch'] = {
        'train_acc': pt_train_acc,
        'test_acc': pt_test_acc,
        'time': torch_time,
        'epochs_per_sec': epochs / torch_time,
    }
    print(f" {torch_time:.2f}s")
    
    # Kernel benchmark
    print("  Kernel training...", end="", flush=True)
    kernel = EqPropKernelBPTT(input_dim, hidden_dim, output_dim, 
                              max_steps=max_steps, lr=0.01)
    
    kernel_start = time.time()
    for epoch in range(epochs):
        kernel.train_step(X_train_np, y_train_np)
    kernel_time = time.time() - kernel_start
    
    kernel_train_result = kernel.evaluate(X_train_np, y_train_np)
    kernel_test_result = kernel.evaluate(X_test_np, y_test_np)
    
    results['kernel'] = {
        'train_acc': kernel_train_result['accuracy'],
        'test_acc': kernel_test_result['accuracy'],
        'time': kernel_time,
        'epochs_per_sec': epochs / kernel_time,
    }
    print(f" {kernel_time:.2f}s")
    
    # Compare
    results['comparison'] = {
        'acc_gap': abs(pt_test_acc - kernel_test_result['accuracy']) * 100,
        'speedup': torch_time / kernel_time,
        'kernel_faster': kernel_time < torch_time,
    }
    
    return results


def main():
    print("="*70)
    print("KERNEL BENCHMARK: Accuracy and Speed Comparison")
    print("="*70)
    
    scenarios = [
        {"name": "Small (64→128→10)", "input_dim": 64, "hidden_dim": 128, "output_dim": 10, 
         "n_samples": 200, "epochs": 50, "max_steps": 20},
        {"name": "Medium (784→256→10)", "input_dim": 784, "hidden_dim": 256, "output_dim": 10, 
         "n_samples": 500, "epochs": 30, "max_steps": 30},
        {"name": "Large (1024→512→10)", "input_dim": 1024, "hidden_dim": 512, "output_dim": 10, 
         "n_samples": 300, "epochs": 20, "max_steps": 30},
        {"name": "Deep (64→128→10, 100 steps)", "input_dim": 64, "hidden_dim": 128, "output_dim": 10, 
         "n_samples": 200, "epochs": 30, "max_steps": 100},
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print("="*70)
        
        results = benchmark_training(
            input_dim=scenario['input_dim'],
            hidden_dim=scenario['hidden_dim'],
            output_dim=scenario['output_dim'],
            n_samples=scenario['n_samples'],
            epochs=scenario['epochs'],
            max_steps=scenario['max_steps'],
        )
        
        all_results[scenario['name']] = results
        
        pt = results['pytorch']
        k = results['kernel']
        c = results['comparison']
        
        print(f"\n  {'PyTorch':12} | Acc: {pt['train_acc']*100:.1f}%/{pt['test_acc']*100:.1f}% | "
              f"Time: {pt['time']:.2f}s | Rate: {pt['epochs_per_sec']:.1f} ep/s")
        print(f"  {'Kernel':12} | Acc: {k['train_acc']*100:.1f}%/{k['test_acc']*100:.1f}% | "
              f"Time: {k['time']:.2f}s | Rate: {k['epochs_per_sec']:.1f} ep/s")
        print(f"\n  Accuracy gap: {c['acc_gap']:.1f}%")
        print(f"  Speedup: {c['speedup']:.2f}× ({'KERNEL FASTER' if c['kernel_faster'] else 'PyTorch faster'})")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n| Scenario | PyTorch Acc | Kernel Acc | Gap | Speedup | Winner |")
    print("|----------|-------------|------------|-----|---------|--------|")
    
    kernel_wins = 0
    pytorch_wins = 0
    
    for name, results in all_results.items():
        pt = results['pytorch']
        k = results['kernel']
        c = results['comparison']
        
        winner = "Kernel" if c['kernel_faster'] else "PyTorch"
        if c['kernel_faster']:
            kernel_wins += 1
        else:
            pytorch_wins += 1
            
        print(f"| {name[:30]:30s} | {pt['test_acc']*100:.1f}% | {k['test_acc']*100:.1f}% | "
              f"{c['acc_gap']:.1f}% | {c['speedup']:.2f}× | {winner} |")
    
    print("\n" + "="*70)
    print(f"VERDICT: Kernel wins {kernel_wins}/{len(all_results)} scenarios")
    print("="*70)
    
    # Recommendation
    avg_speedup = np.mean([r['comparison']['speedup'] for r in all_results.values()])
    avg_acc_gap = np.mean([r['comparison']['acc_gap'] for r in all_results.values()])
    
    print(f"\nAverage speedup: {avg_speedup:.2f}×")
    print(f"Average accuracy gap: {avg_acc_gap:.1f}%")
    
    if avg_acc_gap < 5 and avg_speedup > 0.8:
        if avg_speedup > 1.0:
            print("\n✅ RECOMMENDATION: Use kernel - faster AND equivalent accuracy")
        else:
            print("\n⚠️  RECOMMENDATION: Use kernel for memory benefits, accept speed trade-off")
    else:
        print("\n❌ RECOMMENDATION: Keep PyTorch - kernel has accuracy issues")
    
    return all_results


if __name__ == "__main__":
    results = main()
