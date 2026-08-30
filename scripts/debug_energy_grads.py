#!/usr/bin/env python3
"""Debug energy-contrastive EqProp gradient flow (native version).

The train_step uses manual updates (p -= ...), so we check:
1. _energy_grads returns non-zero gradients for ALL parameters
2. train_step actually updates all parameters (via delta norms)
3. Accuracy improves over steps

Migrated to native compositions after legacy zoo removal.
"""

import torch

from computronium.models.native.eqprop_native import create_native_eqprop_mlp


def debug_energy_grads(steps: int = 10, lr: float = 0.01, beta: float = 2.0):
    """Run energy-contrastive computation and log gradient norms."""
    model = create_native_eqprop_mlp(
        input_dim=784,
        hidden_dim=512,
        output_dim=10,
        num_layers=2,
        beta=beta,
        settle_steps=5,
        lr=lr,
    )

    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    print("=== Energy-Contrastive EqProp Gradient Debug (Native) ===")
    print(f"Config: beta={beta}, lr={lr}, steps={steps}")

    prev_params = {n: p.clone() for n, p in model.geometry.params.items()}

    for step in range(steps):
        result = model.train_step(x, y)
        print(
            f"\nStep {step}: loss={result.get('loss', 'N/A'):.6f}, "
            f"accuracy={result.get('accuracy', 'N/A'):.4f}"
        )

        for name, p in model.geometry.params.items():
            delta = (p - prev_params[name]).norm().item()
            if delta > 0:
                print(f"  {name}: delta_norm={delta:.6f} (UPDATED)")
        prev_params = {n: p.clone() for n, p in model.geometry.params.items()}

    # Direct test of energy computation on a fixed state
    print("\n=== Energy computation direct output ===")
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        acts = model.geometry.forward_with_intermediates(x, model.substrate)
        h_fixed = acts[-2] if len(acts) > 1 else acts[0]

    # Compute energy
    energy = model.dynamics.compute_energy(
        type("State", (), {"free_state": acts, "activations": acts, "x": x})(),
        model.geometry
    )
    print(f"  Energy: {energy.item():.6f}")

    return model


if __name__ == "__main__":
    import sys

    steps = (
        int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 10
    )
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.01
    beta = (
        float(sys.argv[sys.argv.index("--beta") + 1]) if "--beta" in sys.argv else 2.0
    )
    debug_energy_grads(steps=steps, lr=lr, beta=beta)