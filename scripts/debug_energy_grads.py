#!/usr/bin/env python3
"""Debug energy-contrastive EqProp gradient flow.

The train_step uses manual updates (p -= ...), so we check:
1. _energy_grads returns non-zero gradients for ALL parameters
2. train_step actually updates all parameters (via delta norms)
3. Accuracy improves over steps
"""
import torch
from computronium.config.unified import ModelConfig
from computronium.zoo.models.eqprop._energy import EquilibriumMLP


def debug_energy_grads(steps: int = 10, lr: float = 0.01, beta: float = 2.0):
    """Run energy-contrastive computation and log gradient norms."""
    config = ModelConfig(
        name="eqprop_debug",
        input_dim=784,
        output_dim=10,
        hidden_dims=[512, 256],
        learning_rate=lr,
        beta=beta,
        max_steps=5,
        convergence_threshold=0.01,
        use_spectral_norm=False,
    )
    model = EquilibriumMLP(config=config, gradient_method="equilibrium")

    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))

    print("=== Energy-Contrastive EqProp Gradient Debug ===")
    print(f"Config: beta={beta}, lr={lr}, steps={steps}")

    prev_params = {n: p.clone() for n, p in model.named_parameters()}

    for step in range(steps):
        result = model.train_step(x, y)
        print(f"\nStep {step}: loss={result.get('loss', 'N/A'):.6f}, "
              f"accuracy={result.get('accuracy', 'N/A'):.4f}")

        for name, p in model.named_parameters():
            delta = (p - prev_params[name]).norm().item()
            if delta > 0:
                print(f"  {name}: delta_norm={delta:.6f} (UPDATED)")
        prev_params = {n: p.clone() for n, p in model.named_parameters()}

    # Direct test of _energy_grads on a fixed hidden state
    print("\n=== _energy_grads direct output ===")
    h_fixed = torch.randn(32, 512)
    gf = model._energy_grads(h_fixed, x)
    all_nonzero = True
    for (name, p), grad in zip(model.named_parameters(), gf):
        grad_norm = grad.norm().item()
        status = "OK" if grad_norm > 0 else "ZERO"
        print(f"  {name}: grad_norm={grad_norm:.6f} [{status}]")
        if grad_norm == 0:
            all_nonzero = False
    print(f"\nAll gradients non-zero: {all_nonzero}")

    return model


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 10
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.01
    beta = float(sys.argv[sys.argv.index("--beta") + 1]) if "--beta" in sys.argv else 2.0
    debug_energy_grads(steps=steps, lr=lr, beta=beta)
