"""Test energy-contrastive EqProp gradient flow.

Locks down that all non-output parameters receive non-zero contrastive
updates after a ``train_step`` (the consolidated deep-eqprop engine, so the
self-recurrent ``W_rec`` layers are exercised too). W_out is excluded from the
"hidden must move" assertion because it gets the separate supervised update.
"""

import torch
from bioplausible.core.config import ModelConfig
from bioplausible.zoo.models.eqprop._energy import EquilibriumMLP


def _make_config(**overrides):
    defaults = dict(
        name="eqprop_test",
        input_dim=64,
        output_dim=10,
        hidden_dims=[32],
        learning_rate=0.1,
        beta=2.0,
        max_steps=5,
        convergence_threshold=0.01,
        use_spectral_norm=False,
        extra={"gradient_method": "contrastive"},
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_energy_grads_all_params_nonzero():
    """All hidden-layer params must move after a contrastive train_step."""
    config = _make_config(
        input_dim=784,
        hidden_dims=[64],
        learning_rate=0.01,
    )
    model = EquilibriumMLP(config=config)

    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))

    before = {n: p.clone() for n, p in model.named_parameters()}
    model.train_step(x, y)

    for name, p in model.named_parameters():
        if (
            "layers." in name
            and name.endswith(".weight")
            and name.startswith("layers." + str(len(model.layers) - 1) + ".")
        ):
            continue  # W_out has a separate supervised update
        delta = (p - before[name]).norm().item()
        assert delta > 0, f"{name} has zero update after train_step"


def test_train_step_updates_all_params():
    """train_step must update all parameters (no zero deltas)."""
    config = _make_config(
        input_dim=784,
        hidden_dims=[64],
        learning_rate=0.01,
    )
    model = EquilibriumMLP(config=config)

    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))

    prev = {n: p.clone() for n, p in model.named_parameters()}
    model.train_step(x, y)

    for name, p in model.named_parameters():
        delta = (p - prev[name]).norm().item()
        assert delta > 0, f"{name} did not update — zero delta!"


def test_train_step_decreases_loss():
    """Loss should decrease over multiple steps on random data."""
    config = _make_config()
    model = EquilibriumMLP(config=config)

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
