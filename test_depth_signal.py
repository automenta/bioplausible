#!/usr/bin/env python3
"""
Test Spectral Normalization Effect on Deep Linear Chains

This test determines if spectral normalization actually stabilizes
signal propagation through deep networks, or if the claims are false.

CRITICAL TEST: If both With SN and Without SN vanish equally,
the core claims of the project are invalidated.
"""

import torch
import torch.nn as nn
import sys


class LinearChain(nn.Module):
    """Deep linear chain for testing signal propagation."""
    
    def __init__(self, dim=64, depth=100, use_spectral_norm=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_spectral_norm = use_spectral_norm
        
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(dim, dim, bias=False)
            if use_spectral_norm:
                layer = nn.utils.parametrizations.spectral_norm(layer, n_power_iterations=5)
            self.layers.append(layer)
    
    def forward(self, x, return_norms=False):
        """Forward pass, optionally returning norm at each layer."""
        norms = []
        h = x
        
        if return_norms:
            norms.append(h.norm(dim=1).mean().item())
        
        for layer in self.layers:
            h = layer(h)
            if return_norms:
                norms.append(h.norm(dim=1).mean().item())
        
        if return_norms:
            return h, norms
        return h


def test_depth_signal_propagation(depth=100, dim=64):
    """
    Test signal propagation through deep linear chain.
    
    Returns:
        dict with results and verdict
    """
    print(f"\n{'='*70}")
    print(f"TESTING DEPTH={depth}, DIM={dim}")
    print(f"{'='*70}\n")
    
    # Create input
    x = torch.randn(8, dim) * 0.1
    initial_norm = x.norm(dim=1).mean().item()
    
    # Test WITH spectral normalization
    print("Building model WITH spectral normalization...")
    model_sn = LinearChain(dim=dim, depth=depth, use_spectral_norm=True)
    
    with torch.no_grad():
        _, norms_sn = model_sn(x, return_norms=True)
    
    final_sn = norms_sn[-1]
    ratio_sn = final_sn / norms_sn[0]
    
    print(f"  Initial norm: {norms_sn[0]:.6f}")
    print(f"  Layer {depth//4}: {norms_sn[depth//4]:.6f}")
    print(f"  Layer {depth//2}: {norms_sn[depth//2]:.6f}")
    print(f"  Layer {3*depth//4}: {norms_sn[3*depth//4]:.6f}")
    print(f"  Final norm:   {final_sn:.6f}")
    print(f"  Ratio (final/initial): {ratio_sn:.6f}")
    
    # Test WITHOUT spectral normalization
    print("\nBuilding model WITHOUT spectral normalization...")
    model_no_sn = LinearChain(dim=dim, depth=depth, use_spectral_norm=False)
    
    with torch.no_grad():
        _, norms_no_sn = model_no_sn(x, return_norms=True)
    
    final_no_sn = norms_no_sn[-1]
    ratio_no_sn = final_no_sn / norms_no_sn[0]
    
    print(f"  Initial norm: {norms_no_sn[0]:.6f}")
    print(f"  Layer {depth//4}: {norms_no_sn[depth//4]:.6f}")
    print(f"  Layer {depth//2}: {norms_no_sn[depth//2]:.6f}")
    print(f"  Layer {3*depth//4}: {norms_no_sn[3*depth//4]:.6f}")
    print(f"  Final norm:   {final_no_sn:.6f}")
    print(f"  Ratio (final/initial): {ratio_no_sn:.6f}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("VERDICT:")
    print(f"{'='*70}")
    
    results = {
        'depth': depth,
        'dim': dim,
        'sn_initial': norms_sn[0],
        'sn_final': final_sn,
        'sn_ratio': ratio_sn,
        'no_sn_initial': norms_no_sn[0],
        'no_sn_final': final_no_sn,
        'no_sn_ratio': ratio_no_sn,
    }
    
    # Determine if SN helps
    if ratio_sn < 0.01 and ratio_no_sn < 0.01:
        print("❌ BOTH VANISHED - Linear chains are fundamentally unstable!")
        print("   Spectral normalization DOES NOT prevent vanishing in pure linear chains.")
        print("   THE DEPTH CLAIMS ARE INVALID FOR THIS ARCHITECTURE.")
        results['verdict'] = 'BOTH_VANISH'
        results['pass'] = False
        
    elif ratio_sn > 0.5 and ratio_no_sn < 0.1:
        print("✅ SN PRESERVES SIGNAL - Claims validated!")
        print(f"   With SN: {ratio_sn:.3f}× preservation")
        print(f"   Without SN: {ratio_no_sn:.6f}× (vanished)")
        results['verdict'] = 'SN_WORKS'
        results['pass'] = True
        
    elif ratio_sn > ratio_no_sn * 2:
        print("⚠️  SN HELPS BUT BOTH DECAY")
        print(f"   With SN: {ratio_sn:.3f}× preservation")
        print(f"   Without SN: {ratio_no_sn:.3f}× preservation")
        print(f"   SN is {ratio_sn/ratio_no_sn:.1f}× better, but not perfect")
        results['verdict'] = 'SN_HELPS'
        results['pass'] = True
        
    else:
        print("❓ INCONCLUSIVE - No clear difference")
        print(f"   With SN: {ratio_sn:.3f}×")
        print(f"   Without SN: {ratio_no_sn:.3f}×")
        results['verdict'] = 'INCONCLUSIVE'
        results['pass'] = False
    
    print(f"{'='*70}\n")
    
    return results


def main():
    """Run comprehensive depth tests."""
    print("\n" + "="*70)
    print("DEPTH SIGNAL PROPAGATION TEST")
    print("="*70)
    
    # Test different depths
    all_results = []
    
    for depth in [50, 100, 200, 500]:
        results = test_depth_signal_propagation(depth=depth, dim=64)
        all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for r in all_results:
        status = "✅ PASS" if r['pass'] else "❌ FAIL"
        print(f"{status} Depth {r['depth']:3d}: SN={r['sn_ratio']:.4f}, No-SN={r['no_sn_ratio']:.6f} ({r['verdict']})")
    
    # Final verdict
    all_pass = all(r['pass'] for r in all_results)
    
    print("\n" + "="*70)
    if all_pass:
        print("FINAL VERDICT: ✅ Spectral Normalization WORKS for depth")
        return 0
    else:
        print("FINAL VERDICT: ❌ Claims NOT validated - linear chains unsuitable")
        return 1


if __name__ == '__main__':
    sys.exit(main())
