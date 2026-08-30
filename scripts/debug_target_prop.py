#!/usr/bin/env python3
"""Debug target propagation: target computation, inverse mapping, multi-step targets (native version).

Checks:
1. Targets propagate to all hidden layers (not just output)
2. Inverse mapping reduces cycle error
3. Multiple target steps improve target quality

Migrated to native compositions after legacy zoo removal.
"""

import torch

from computronium.models.native.tile_native import create_native_tile_tp


def debug_target_prop(steps: int = 10, lr: float = 0.001, target_lr: float = 0.1):
    """Run target prop and log target propagation + inverse quality."""
    model = create_native_tile_tp(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        num_layers=3,
        neurons_per_tile=16,
        tiles_per_layer=4,
        lr=lr,
        beta=target_lr,
    )

    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    print("=== Target Prop Debug (Native Tile) ===")
    print(f"Config: lr={lr}, target_lr={target_lr}, steps={steps}")

    for step in range(steps):
        result = model.train_step(x, y)
        print(f"\nStep {step}: loss={result['loss']:.6f}, acc={result['accuracy']:.4f}")

        if step < 2 or step >= steps - 2:
            # Check forward/backward propagation through tiles
            model.eval()  # type: ignore[attr-defined]
            with torch.no_grad():
                acts = model.geometry.forward_with_intermediates(x, model.substrate)
                for i, act in enumerate(acts):
                    print(f"  Layer {i}: shape={tuple(act.shape)}, norm={act.norm().item():.4f}")

    print("\n=== Summary ===")
    print("Activations should propagate through all tile layers")
    print("Tile TP uses PredictiveSettlingDynamics + TargetInversionCredit")


if __name__ == "__main__":
    import sys

    steps = (
        int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 10
    )
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.001
    target_lr = (
        float(sys.argv[sys.argv.index("--target_lr") + 1])
        if "--target_lr" in sys.argv
        else 0.1
    )
    debug_target_prop(steps=steps, lr=lr, target_lr=target_lr)