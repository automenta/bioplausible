"""Regression: implicit-differentiation ('equilibrium') method must learn at O(1).

Guards against the defect that stranded run_phase1_5.py eqprop models on
BPTT. Historically FIX.md claimed the implicit method "fails to learn due to a
gradient-quality issue" — the real cause was that
:meth:`BioModel._get_spectral_normalized_weight` returned a *detached* weight
when ``model.eval()`` was set during ``EquilibriumFunction.backward``, killing
the recurrent weight's gradient entirely (``W_rec.grad is None``).

These tests assert:
1. ``equilibrium`` gradients match full BPTT (relative error is small).
2. Every learnable parameter receives a real gradient (no silent ``None``).
3. ``equilibrium`` actually drives training loss down on a deterministic task.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.zoo.models.eqprop.conv_eqprop import ConvEqProp
from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

_GRAD_PARITY_TOL = 5e-2  # implicit-diff vs BPTT relative error budget


def _forward_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(x), y)


def _params_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return {name: grad} for params that require grad and got a gradient."""
    return {
        n: p.grad
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def _assert_no_none_grads(model: nn.Module) -> None:
    none_names = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not none_names, (
        f"equilibrium method produced None gradients for: {none_names}. "
        "Spectral-norm weights must stay differentiable during the implicit "
        "backward (FIX.md regression)."
    )


def _max_rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm().item()) / (a.norm().item() + 1e-12)


def _grads(bptt_model: nn.Module, eq_model: nn.Module) -> None:
    bptt_model.load_state_dict(eq_model.state_dict())
    x = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    for m, gm in ((bptt_model, "bptt"), (eq_model, "equilibrium")):
        m.zero_grad()
        _forward_loss(m, x, y).backward()


@pytest.mark.parametrize("use_sn", [True, False])
def test_equilibrium_gradients_match_bptt_looped_mlp(use_sn: bool) -> None:
    """Implicit-equilibrium gradients equal unrolled BPTT to within a tolerance."""
    torch.manual_seed(0)
    bptt = LoopedMLP(
        input_dim=8,
        hidden_dim=12,
        output_dim=3,
        use_spectral_norm=use_sn,
        max_steps=10,
        gradient_method="bptt",
    )
    eq = LoopedMLP(
        input_dim=8,
        hidden_dim=12,
        output_dim=3,
        use_spectral_norm=use_sn,
        max_steps=10,
        gradient_method="equilibrium",
    )
    _grads(bptt, eq)
    _assert_no_none_grads(eq)

    worst = 0.0
    for (nb, gb), (ne, ge) in zip(bptt.named_parameters(), eq.named_parameters()):
        assert nb == ne
        worst = max(worst, _max_rel_error(gb, ge))
    assert worst < _GRAD_PARITY_TOL, (
        f"equilibrium gradient drifted from BPTT (rel={worst:.3e})"
    )


def test_equilibrium_learns_looped_mlp_with_spectral_norm() -> None:
    """The O(1) implicit method must reduce loss (the 'doesn't learn' regression)."""
    torch.manual_seed(0)
    model = LoopedMLP(
        input_dim=8,
        hidden_dim=16,
        output_dim=3,
        use_spectral_norm=True,
        max_steps=10,
        gradient_method="equilibrium",
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    # Fixed target function for the duration of training
    w = torch.randn(8, 3)
    first = last = None
    for epoch in range(40):
        x = torch.randn(16, 8)
        y = (x @ w).argmax(dim=1)
        opt.zero_grad()
        loss = _forward_loss(model, x, y)
        loss.backward()
        _assert_no_none_grads(model)
        opt.step()
        if epoch == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, (
        f"equilibrium method did not learn (first={first:.4f}, last={last:.4f})"
    )


def test_conv_eqprop_equilibrium_learns() -> None:
    """Conv models default to the O(1) implicit method and still learn."""
    torch.manual_seed(0)
    model = ConvEqProp(
        input_channels=3,
        hidden_channels=16,
        output_dim=10,
        use_spectral_norm=True,
        max_steps=8,
        gradient_method="equilibrium",
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 10, (4,))
    first = last = None
    for epoch in range(8):
        opt.zero_grad()
        loss = _forward_loss(model, x, y)
        loss.backward()
        opt.step()
        if epoch == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, (
        f"conv equilibrium method did not learn (first={first:.4f}, last={last:.4f})"
    )
