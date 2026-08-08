"""Test energy-contrastive EqProp gradient flow.

Locks down that all non-output parameters receive non-zero energy gradients.
W_out is excluded because it gets a separate supervised update.
"""
import torch
from bioplausible.core.config import ModelConfig
from bioplausible.zoo.models.eqprop._energy import EquilibriumMLP


def test_energy_grads_all_params_nonzero():
    """All hidden-layer params must receive non-zero energy gradients."""
    config = ModelConfig(
        name="eqprop_test",
        input_dim=784,
        output_dim=10,
        hidden_dims=[64],
        learning_rate=0.01,
        beta=2.0,
        max_steps=5,
        convergence_threshold=0.01,
        use_spectral_norm=False,
    )
    model = EquilibriumMLP(config=config, gradient_method="equilibrium")

    x = torch.randn(8, 784)
    h_fixed = torch.randn(8, 64)

    gf = model._energy_grads(h_fixed, x)

    for (name, p), grad in zip(model.named_parameters(), gf):
        # W_out gets a supervised update path, not energy gradient
        if "W_out" in name:
            continue
        assert grad.norm().item() > 0, f"{name} has zero gradient — W_in/W_rec not updating!"


def test_train_step_updates_all_params():
    """train_step must update all parameters (no zero deltas)."""
    config = ModelConfig(
        name="eqprop_test",
        input_dim=784,
        output_dim=10,
        hidden_dims=[64],
        learning_rate=0.01,
        beta=2.0,
        max_steps=5,
        convergence_threshold=0.01,
        use_spectral_norm=False,
    )
    model = EquilibriumMLP(config=config, gradient_method="equilibrium")

    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))

    prev = {n: p.clone() for n, p in model.named_parameters()}
    model.train_step(x, y)

    for name, p in model.named_parameters():
        delta = (p - prev[name]).norm().item()
        assert delta > 0, f"{name} did not update — zero delta!"


def test_train_step_decreases_loss():
    """Loss should decrease over multiple steps on random data."""
    config = ModelConfig(
        name="eqprop_test",
        input_dim=64,
        output_dim=10,
        hidden_dims=[32],
        learning_rate=0.1,
        beta=2.0,
        max_steps=5,
        convergence_threshold=0.01,
        use_spectral_norm=False,
    )
    model = EquilibriumMLP(config=config, gradient_method="equilibrium")

    torch.manual_seed(42)
    x = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))

    losses = []
    for _ in range(10):
        result = model.train_step(x, y)
        losses.append(result["loss"])

    # Loss should be decreasing overall
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} → {losses[-1]}"


if __name__ == "__main__":
    test_energy_grads_all_params_nonzero()
    test_train_step_updates_all_params()
    test_train_step_decreases_loss()
    print("All tests passed!")
