"""Numerical parity snapshots for GATE-0 (REFACTOR4).

Pins the known-drifting values from the xfail'ed parity tests so a semantic
regression during LOOP step 3 / RULE fails loudly instead of silently passing.
Mirrors the exact harness of the backing tests (test_backprop_parity /
test_equilibrium_parity) so numbers stay comparable.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

SNAPSHOT_DIR = pathlib.Path(__file__).parent / "parity_snapshots"

# GATE-0 baseline (verified 2026-08-14). Refresh only after an intentional,
# verified training-semantics change.
SNAPSHOTS = {
    "eqprop_mlp": {"accuracy": 0.198},
    "directed_ep": {"accuracy": 0.114},
    "mlp_gradient_parity": {"loss_bptt": 1.487, "loss_eqprop": 1.538},
}


def _synthetic_task():
    torch.manual_seed(42)
    n, input_dim, n_classes = 500, 64, 10
    x = torch.randn(n, input_dim)
    y = torch.randint(0, n_classes, (n,))
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            d = torch.randn(input_dim)
            d = d / d.norm() * 2.0
            x[mask] += d * 0.8
    return x, y, input_dim, n_classes


def _train_model(model, x, y, epochs=3, batch_size=32):
    """Exact harness from test_backprop_parity._train_model."""
    from torch import nn, optim

    model.train()
    has_custom = False
    if hasattr(model, "train_step"):
        has_custom = model.train_step(x[:batch_size], y[:batch_size]) is not None
    if has_custom:
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), batch_size):
                model.train_step(x[perm[i : i + batch_size]], y[perm[i : i + batch_size]])
    else:
        opt = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                loss = criterion(model(x[idx]), y[idx])
                loss.backward()
                opt.step()
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def _eqprop_mlp_acc():
    x, y, input_dim, n_classes = _synthetic_task()
    torch.manual_seed(456)
    from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

    m = LoopedMLP(
        input_dim=input_dim, hidden_dim=64, output_dim=n_classes,
        use_spectral_norm=True, max_steps=20,
        gradient_method="contrastive", backend="pytorch",
    )
    m.hebbian_lr = 0.008
    m.beta = 0.03
    return _train_model(m, x, y)


def _directed_ep_acc():
    x, y, input_dim, n_classes = _synthetic_task()
    torch.manual_seed(456)
    from bioplausible.config.unified import ModelConfig
    from bioplausible.zoo.models.eqprop.deep_ep import DirectedEP

    m = DirectedEP(
        ModelConfig(
            name="directed_ep", input_dim=input_dim, output_dim=n_classes,
            hidden_dims=[64, 64], learning_rate=0.03, beta=0.3, max_steps=20,
        )
    )
    return _train_model(m, x, y)


def _mlp_gradient_losses():
    torch.manual_seed(42)
    from torch import nn
    from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

    input_dim, hidden_dim, output_dim, batch_size, max_steps = 10, 20, 5, 4, 100
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    m_bptt = LoopedMLP(input_dim, hidden_dim, output_dim, max_steps=max_steps, gradient_method="bptt", use_spectral_norm=False)
    m_eq = LoopedMLP(input_dim, hidden_dim, output_dim, max_steps=max_steps, gradient_method="equilibrium", use_spectral_norm=False)
    m_eq.load_state_dict(m_bptt.state_dict())
    crit = nn.CrossEntropyLoss()
    m_bptt.zero_grad()
    loss_bptt = crit(m_bptt(x), y)
    loss_bptt.backward()
    m_eq.zero_grad()
    loss_eq = crit(m_eq(x), y)
    loss_eq.backward()
    return {"loss_bptt": loss_bptt.item(), "loss_eqprop": loss_eq.item()}


@pytest.mark.parametrize("name", ["eqprop_mlp", "directed_ep"])
def test_accuracy_snapshot(name) -> None:
    actual = {"eqprop_mlp": _eqprop_mlp_acc, "directed_ep": _directed_ep_acc}[name]()
    ref = SNAPSHOTS[name]["accuracy"]
    assert abs(actual - ref) < 1e-3, f"{name} accuracy {actual:.6f} != pinned {ref:.6f}"


def test_mlp_gradient_loss_snapshot() -> None:
    actual = _mlp_gradient_losses()
    ref = SNAPSHOTS["mlp_gradient_parity"]
    for k in ("loss_bptt", "loss_eqprop"):
        assert abs(actual[k] - ref[k]) < 1e-3, f"mlp_gradient {k} {actual[k]:.6f} != {ref[k]:.6f}"
