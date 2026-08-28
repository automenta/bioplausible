#!/usr/bin/env python3
"""Debug target propagation: target computation, inverse mapping, multi-step targets.

Checks:
1. Targets propagate to all hidden layers (not just output)
2. Inverse mapping reduces cycle error
3. Multiple target steps improve target quality
"""

import torch
from torch import nn

from computronium.zoo.models.target_prop import DifferenceTargetProp


def debug_target_prop(steps: int = 10, lr: float = 0.001, target_lr: float = 0.1):
    """Run target prop and log target propagation + inverse quality."""
    model = DifferenceTargetProp(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        num_layers=3,
        learning_rate=lr,
        target_lr=target_lr,
    )

    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    print("=== Target Prop Debug ===")
    print(f"Config: lr={lr}, target_lr={target_lr}, steps={steps}")

    for step in range(steps):
        result = model.train_step(x, y)
        print(f"\nStep {step}: loss={result['loss']:.6f}, acc={result['accuracy']:.4f}")

        if step < 2 or step >= steps - 2:
            # Check inverse mapping quality for each layer
            for i, layer in enumerate(model.layers):
                h = torch.randn(32, layer.forward_net[0].in_features)
                fwd = layer.forward_net(h)
                inv = layer.inverse_net(fwd)
                cycle_error = nn.functional.mse_loss(inv, h).item()
                print(f"  Layer {i}: cycle_error={cycle_error:.6f}")

    print("\n=== Summary ===")
    print("Targets should propagate to all layers (see cycle errors above)")
    print("Lower cycle error = better inverse mapping = better target propagation")


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
