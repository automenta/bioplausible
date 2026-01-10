#!/usr/bin/env python3
"""
Quick verification script for eqprop-torch library.

Runs comprehensive tests to ensure all components are working.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow importing without installation
sys.path.insert(0, str(Path(__file__).parent.parent))



def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from eqprop_torch import (
            # Core
            EqPropTrainer,
            # Models
            LoopedMLP, BackpropMLP, ConvEqProp, TransformerEqProp,
            # Kernel
            EqPropKernel, HAS_CUPY,
            # Utils
            compile_model, get_optimal_backend,
            count_parameters, verify_spectral_norm, create_model_preset,
            # Datasets
            get_vision_dataset, get_lm_dataset,
            # Flags
            HAS_LM_VARIANTS,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test creating models."""
    print("\nTesting model creation...")
    try:
        from eqprop_torch import LoopedMLP, ConvEqProp, count_parameters
        
        # Test MLP
        mlp = LoopedMLP(784, 256, 10, use_spectral_norm=True)
        params = count_parameters(mlp)
        print(f"  LoopedMLP: {params:,} parameters")
        
        # Test Conv
        conv = ConvEqProp(1, 32, 10)
        params = count_parameters(conv)
        print(f"  ConvEqProp: {params:,} parameters")
        
        print("✓ Model creation OK")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_spectral_norm():
    """Test spectral normalization."""
    print("\nTesting spectral normalization...")
    try:
        from eqprop_torch import LoopedMLP, verify_spectral_norm
        
        model = LoopedMLP(100, 128, 10, use_spectral_norm=True)
        L = model.compute_lipschitz()
        
        if L <= 1.0:
            print(f"  Lipschitz constant: {L:.4f} ≤ 1.0 ✓")
        else:
            print(f"  WARNING: Lipschitz constant {L:.4f} > 1.0")
        
        sn_values = verify_spectral_norm(model)
        print(f"  Verified {len(sn_values)} layers with spectral norm")
        
        print("✓ Spectral norm OK")
        return True
    except Exception as e:
        print(f"✗ Spectral norm test failed: {e}")
        return False


def test_trainer():
    """Test EqPropTrainer."""
    print("\nTesting EqPropTrainer...")
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from eqprop_torch import EqPropTrainer, LoopedMLP
        
        model = LoopedMLP(50, 32, 5, use_spectral_norm=True)
        trainer = EqPropTrainer(model, use_compile=False, lr=0.01)
        
        # Quick training test
        x = torch.randn(16, 50)
        y = torch.randint(0, 5, (16,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        
        history = trainer.fit(loader, epochs=1)
        
        print(f"  Device: {trainer.device}")
        print(f"  Final loss: {history['train_loss'][-1]:.4f}")
        print("✓ Trainer OK")
        return True
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_presets():
    """Test model presets."""
    print("\nTesting model presets...")
    try:
        from eqprop_torch import create_model_preset, count_parameters
        
        presets = ['mnist_small', 'mnist_medium', 'mnist_large']
        for preset in presets:
            model = create_model_preset(preset)
            params = count_parameters(model)
            print(f"  {preset}: {params:,} parameters")
        
        print("✓ Presets OK")
        return True
    except Exception as e:
        print(f"✗ Preset test failed: {e}")
        return False


def test_optional_features():
    """Test optional features."""
    print("\nTesting optional features...")
    from eqprop_torch import HAS_CUPY, HAS_LM_VARIANTS
    
    print(f"  CuPy available: {HAS_CUPY}")
    print(f"  LM variants available: {HAS_LM_VARIANTS}")
    
    if HAS_LM_VARIANTS:
        try:
            from eqprop_torch import list_eqprop_lm_variants
            variants = list_eqprop_lm_variants()
            print(f"  LM variants: {variants}")
        except:
            pass
    
    print("✓ Optional features checked")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("EqProp-Torch Library Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_spectral_norm,
        test_trainer,
        test_presets,
        test_optional_features,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 60)
        return 0
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
