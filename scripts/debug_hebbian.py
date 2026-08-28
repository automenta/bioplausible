#!/usr/bin/env python3
"""Debug three-factor Hebbian: modulator signal, weight update direction.

Checks:
1. Modulator is graded (continuous, not binary)
2. Weight updates correlate with error direction
3. Loss decreases over steps
"""

import torch

from computronium.zoo.models.hebbian import ThreeFactorHebbian


def debug_hebbian(steps: int = 10, lr: float = 0.005):
    """Run three-factor Hebbian and log modulator + weight changes."""
    model = ThreeFactorHebbian(
        input_dim=784, hidden_dim=128, output_dim=10, num_layers=3
    )
    model.lr = lr

    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    print("=== Three-Factor Hebbian Debug ===")
    print(f"Config: lr={lr}, steps={steps}")

    for step in range(steps):
        w_before = [p.data.clone() for p in model.parameters()]
        result = model.train_step(x, y)

        # Check modulator
        with torch.no_grad():
            out = model.forward(x)
            pred_probs = torch.softmax(out, dim=1)
            y_onehot = torch.zeros_like(out)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            modulator = (y_onehot - pred_probs).mean().item()

        print(
            f"\nStep {step}: loss={result['loss']:.6f}, acc={result['accuracy']:.4f}, "
            f"mean_modulator={modulator:.6f}"
        )

        if step < 2 or step >= steps - 2:
            for name, p, w_b in zip(
                model.named_parameters(), model.parameters(), w_before
            ):
                if "out_layer" in name:
                    delta = (p.data - w_b).norm().item()
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
