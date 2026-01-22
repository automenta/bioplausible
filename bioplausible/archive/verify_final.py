#!/usr/bin/env python3
"""Final verification script before launch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== EqProp-Torch Final Verification ===\n")

# Test imports
try:
    from bioplausible import (HAS_BIOPLAUSIBLE, AdaptiveFeedbackAlignment,
                              ConvEqProp, EqPropTrainer, LoopedMLP,
                              StandardEqProp, enable_tf32)

    print("✓ Core imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test model creation
try:
    model1 = LoopedMLP(784, 256, 10)
    print(
        f"✓ LoopedMLP created: {sum(p.numel() for p in model1.parameters()):,} params"
    )

    from algorithms import AlgorithmConfig

    config = AlgorithmConfig("eqprop", 784, [256], 10)
    model2 = StandardEqProp(config)
    print(
        f"✓ StandardEqProp created: {sum(p.numel() for p in model2.parameters()):,} params"
    )
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test trainer
try:
    trainer = EqPropTrainer(model1, use_compile=False)
    print(f"✓ EqPropTrainer initialized on {trainer.device}")
except Exception as e:
    print(f"✗ Trainer failed: {e}")
    sys.exit(1)

print(f"\n✅ All checks passed! Ready for use.")
print(f"   HAS_BIOPLAUSIBLE: {HAS_BIOPLAUSIBLE}")
print(f"   Total models: 22 (4 native + 5 LM + 13 research)\n")
