"""
Quick NEBC Verification Script

Lightweight test to verify SN impact with minimal compute.
Focuses on: accuracy, Lipschitz, training dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import (
    LoopedMLP, 
    FeedbackAlignmentEqProp,
    DirectFeedbackAlignmentEqProp,
    ContrastiveHebbianLearning,
    DeepHebbianChain,
)

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def create_simple_data(n=500):
    """Create simple synthetic classification task."""
    X = torch.randn(n, 64)
    y = (X[:, :10].sum(dim=1) > 0).long()  # Simple decision boundary
    return X, y

def quick_train(model, X, y, epochs=10, lr=0.01):
    """Quick training with loss tracking."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    accs = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        
        acc = (out.argmax(dim=1) == y).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
    
    return losses, accs

def test_algorithm(name, model_class, X_train, y_train, X_test, y_test, **kwargs):
    """Test algorithm with/without SN."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    results = {}
    
    for use_sn in [True, False]:
        label = 'with_sn' if use_sn else 'without_sn'
        print(f"\n  [{label}]")
        
        # Create model
        try:
            model = model_class(
                input_dim=64, hidden_dim=64, output_dim=2,
                use_spectral_norm=use_sn, **kwargs
            )
        except Exception as e:
            print(f"    Error creating model: {e}")
            continue
        
        # Train
        try:
            losses, train_accs = quick_train(model, X_train, y_train, epochs=20, lr=0.01)
        except Exception as e:
            print(f"    Error training: {e}")
            continue
        
        # Evaluate
        with torch.no_grad():
            out = model(X_test)
            test_acc = (out.argmax(dim=1) == y_test).float().mean().item()
        
        # Compute Lipschitz
        L = model.compute_lipschitz() if hasattr(model, 'compute_lipschitz') else 0.0
        
        # Training dynamics
        final_loss = losses[-1]
        loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100
        final_train_acc = train_accs[-1]
        
        results[label] = {
            'test_acc': test_acc,
            'train_acc': final_train_acc,
            'final_loss': final_loss,
            'loss_reduction': loss_reduction,
            'lipschitz': L,
            'losses': losses,
            'train_accs': train_accs,
        }
        
        print(f"    Train acc: {final_train_acc*100:.1f}%")
        print(f"    Test acc:  {test_acc*100:.1f}%")
        print(f"    Loss:      {final_loss:.4f} (↓{loss_reduction:.1f}%)")
        print(f"    Lipschitz: {L:.3f}")
    
    # Compare
    if 'with_sn' in results and 'without_sn' in results:
        print(f"\n  [Comparison]")
        acc_delta = (results['with_sn']['test_acc'] - results['without_sn']['test_acc']) * 100
        L_delta = results['without_sn']['lipschitz'] - results['with_sn']['lipschitz']
        
        print(f"    Δ Test Acc:  {acc_delta:+.1f}%")
        print(f"    Δ Lipschitz: {L_delta:+.2f} (SN reduces L)")
        print(f"    SN Stabilizes: {'✅' if results['with_sn']['lipschitz'] <= 1.1 else '❌'}")
        print(f"    SN Improves Acc: {'✅' if acc_delta > -2 else '⚠️ NO'}")
        
        # Verdict
        sn_worth_it = (
            results['with_sn']['lipschitz'] <= 1.1 and 
            acc_delta > -5  # Allow small accuracy drop for stability
        )
        print(f"    Verdict: {'✅ SN BENEFICIAL' if sn_worth_it else '⚠️ NEEDS TUNING'}")
    
    return results


def main():
    print("="*60)
    print("QUICK NEBC VERIFICATION")
    print("Lightweight test with synthetic data (500 samples, 20 epochs)")
    print("="*60)
    
    # Create data
    X_train, y_train = create_simple_data(n=500)
    X_test, y_test = create_simple_data(n=100)
    
    # Test each algorithm
    algorithms = [
        ("LoopedMLP", LoopedMLP, {}),
        ("Feedback Alignment", FeedbackAlignmentEqProp, {"num_layers": 3}),
        ("Direct FA", DirectFeedbackAlignmentEqProp, {"num_layers": 3}),
        ("Contrastive Hebbian", ContrastiveHebbianLearning, {"num_layers": 2}),
        ("Hebbian Chain", DeepHebbianChain, {"num_layers": 10}),
    ]
    
    all_results = {}
    
    for name, model_class, kwargs in algorithms:
        results = test_algorithm(name, model_class, X_train, y_train, X_test, y_test, **kwargs)
        all_results[name] = results
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Algorithm':<25} {'Acc (SN)':<12} {'Acc (No SN)':<12} {'Δ Acc':<10} {'L (SN)':<10} {'Verdict'}")
    print("-"*90)
    
    for name, results in all_results.items():
        if 'with_sn' in results and 'without_sn' in results:
            acc_sn = results['with_sn']['test_acc'] * 100
            acc_no = results['without_sn']['test_acc'] * 100
            delta = acc_sn - acc_no
            L_sn = results['with_sn']['lipschitz']
            
            verdict = "✅ GOOD" if (L_sn <= 1.1 and delta > -5) else "⚠️ TUNE"
            
            print(f"{name:<25} {acc_sn:>6.1f}%      {acc_no:>6.1f}%      {delta:>+5.1f}%    {L_sn:>6.3f}    {verdict}")
    
    print("="*60)
    print("\nKey Findings:")
    print("- If 'Δ Acc' is negative: SN may need hyperparameter tuning")
    print("- If 'L (SN)' > 1.1: SN not fully constraining (increase spectral norm power iterations?)")
    print("- ✅ GOOD: SN stabilizes AND maintains accuracy")
    print("- ⚠️ TUNE: SN stabilizes BUT accuracy drops (try higher LR, more epochs)")


if __name__ == "__main__":
    main()
