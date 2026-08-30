#!/usr/bin/env python3
"""Debug three-factor Hebbian: modulator signal, weight update direction (native version).

Checks:
1. Modulator is graded (continuous, not binary)
2. Weight updates correlate with error direction
3. Loss decreases over steps

Migrated to native compositions after legacy zoo removal.
"""

import torch

from computronium.models.native.tile_native import create_native_tile_hebbian


def debug_hebbian(steps: int = 10, lr: float = 0.005):
    """Run three-factor Hebbian and log modulator + weight changes."""
    model = create_native_tile_hebbian(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        num_layers=3,
        neurons_per_tile=16,
        tiles_per_layer=4,
        lr=lr,
    )

    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    print("=== Three-Factor Hebbian Debug (Native Tile) ===")
    print(f"Config: lr={lr}, steps={steps}")

    for step in range(steps):
        w_before = {n: p.clone() for n, p in model.geometry.params.items()}
        result = model.train_step(x, y)

        # Check modulator (error signal)
        with torch.no_grad():
            out = model(x)  # type: ignore[operator]
            pred_probs = torch.softmax(out, dim=1)
            y_onehot = torch.zeros_like(out)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            modulator = (y_onehot - pred_probs).mean().item()

        print(
            f"\nStep {step}: loss={result['loss']:.6f}, acc={result['accuracy']:.4f}, "
            f"mean_modulator={modulator:.6f}"
        )

        if step < 2 or step >= steps - 2:
            for name, p in model.geometry.params.items():
                if "output_proj" in name or "tile_weight" in name:
                    delta = (p - w_before[name]).norm().item()
                    if delta > 0:
                        print(f"  {name}: delta_norm={delta:.6f}")

    print("\n=== Summary ===")
    print("Modulator should be non-zero (graduated error signal, not binary)")
    print("Weight deltas should be non-zero for error-modulated updates")


if __name__ == "__main__":
    import sys

    steps = (
        int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 10
    )
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.005
    debug_hebbian(steps=steps, lr=lr)
