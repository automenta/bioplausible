"""
Model Introspection Demo

Demonstrates how to use the forward() method with return_dynamics=True to inspect
internal convergence dynamics of Equilibrium Propagation models.
"""

import torch
from bioplausible.zoo.models.eqprop import DirectedEP, StandardEqProp


def run_introspection(model, name):
    print(f"\n--- Introspecting {name} ---")

    # Generate random input
    x = torch.randn(5, model.input_dim)

    # Get Dynamics via model.forward() with return_dynamics=True
    print("Running equilibrium dynamics...")
    model.eval()
    with torch.no_grad():
        _output, dynamics = model.forward(
            x, return_trajectory=True, return_dynamics=True
        )

    # Analyze
    deltas = dynamics["deltas"]
    final_delta = dynamics["final_delta"]
    trajectory = dynamics["trajectory"]

    print(f"Convergence steps: {len(deltas)}")
    print(f"Final Delta (Rate of Change): {final_delta:.6f}")

    # Check if converging (deltas should decrease)
    is_converging = deltas[-1] < deltas[0]
    print(f"Converging: {is_converging}")
    print(f"Deltas (first 5): {[f'{d:.4f}' for d in deltas[:5]]}")
    print(f"Deltas (last 5): {[f'{d:.4f}' for d in deltas[-5:]]}")

    # Trajectory shape
    # trajectory is list of list of tensors (activations for each layer)
    # len(trajectory) = steps + 1 (initial)
    # trajectory[0] is list of layer activations at step 0
    print(f"Trajectory captured: {len(trajectory)} steps")
    layers_count = len(trajectory[0])
    print(f"Model depth: {layers_count} layers (including input)")


def main():
    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    # 1. Standard EqProp (Symmetric)
    model_std = StandardEqProp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        equilibrium_steps=30,
        beta=0.1,
    )
    run_introspection(model_std, "Standard EqProp")

    # 2. Directed EqProp (Asymmetric/Relaxed)
    model_deep = DirectedEP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        equilibrium_steps=30,
        beta=0.1,
    )
    run_introspection(model_deep, "Directed EqProp (Deep EP)")


if __name__ == "__main__":
    main()
