"""Tests for the EqProp family propagator wrappers.

Covers: EqProp, HolomorphicEqProp, FiniteNudgeEqProp, LazyEqProp
in bioplausible.zoo.propagators.eqprop.
"""

import pytest
import torch
from torch import nn

from bioplausible.zoo.propagators.eqprop import (
    EqProp,
    FiniteNudgeEqProp,
    HolomorphicEqProp,
    LazyEqProp,
)


# =============================================================================
# Fixtures
# =============================================================================


class SameDimMLP(nn.Module):
    """Equal-dim MLP so _compute_ep_gradient can reach all weight params."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SameDimMLP(dim=8)


@pytest.fixture
def params(model):
    return list(model.parameters())


@pytest.fixture
def x():
    return torch.randn(4, 8)


@pytest.fixture
def target():
    return torch.randint(0, 8, (4,))


# =============================================================================
# EqProp Tests
# =============================================================================


class TestEqProp:
    """Tests for standard Equilibrium Propagation wrapper."""

    def test_step_updates_params(self, params, model, x, target):
        torch.manual_seed(42)
        opt = EqProp(params, model, lr=0.01, beta=0.5, settle_steps=2, settle_lr=0.01)
        old = [p.clone() for p in params]
        opt.step(x, target)
        assert any(not torch.equal(o, p) for o, p in zip(old, params)), (
            "EqProp step should update params"
        )

    def test_step_raises_no_target(self, params, model, x):
        opt = EqProp(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)

    def test_settle_returns_pairs(self, params, model, x):
        opt = EqProp(params, model)
        pairs = opt._settle(x, target=None, beta=0.0)
        assert isinstance(pairs, list)
        for p in pairs:
            assert isinstance(p, tuple) and len(p) == 2
            inp, out = p
            assert isinstance(inp, torch.Tensor)
            assert isinstance(out, torch.Tensor)
            assert inp.shape[0] == out.shape[0]

    def test_get_layers_returns_weighted_only(self, params, model):
        opt = EqProp(params, model)
        layers = opt._get_layers()
        assert len(layers) == 2
        assert all(isinstance(l, (nn.Linear, nn.Conv2d)) for l in layers)

    def test_compute_ep_gradient_sets_grad(self, params, model, x, target):
        torch.manual_seed(42)
        opt = EqProp(params, model, beta=0.5)
        free = opt._settle(x, target=None, beta=0.0)
        nudged = opt._settle(x, target=target, beta=0.5)
        for p in params:
            p.grad = None
        opt._compute_ep_gradient(free, nudged)
        # Only params with ndim >= 2 AND index < len(pairs) get grads
        pairs_len = len(free)
        reachable = [p for i, p in enumerate(params) if p.ndim >= 2 and i < pairs_len]
        assert all(p.grad is not None for p in reachable)
        for p in reachable:
            assert p.grad.shape == p.shape


# =============================================================================
# HolomorphicEqProp Tests
# =============================================================================


class TestHolomorphicEqProp:
    """Tests for HolomorphicEqProp wrapper."""

    def test_step_updates_params(self, params, model, x, target):
        torch.manual_seed(42)
        opt = HolomorphicEqProp(params, model, lr=0.1)
        old = [p.clone() for p in params]
        opt.step(x, target)
        assert any(not torch.equal(o, p) for o, p in zip(old, params)), (
            "HolomorphicEqProp step should update params"
        )

    def test_step_raises_no_target(self, params, model, x):
        opt = HolomorphicEqProp(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)


# =============================================================================
# FiniteNudgeEqProp Tests
# =============================================================================


class TestFiniteNudgeEqProp:
    """Tests for FiniteNudgeEqProp wrapper."""

    def test_step_updates_params(self, params, model, x, target):
        torch.manual_seed(42)
        loss = nn.functional.cross_entropy(model(x), target)
        loss.backward()
        opt = FiniteNudgeEqProp(params, model, lr=0.1, beta=2.0)
        old = [p.clone() for p in params]
        opt.step(x, target)
        assert any(not torch.equal(o, p) for o, p in zip(old, params)), (
            "FiniteNudgeEqProp step should update params"
        )

    def test_step_scales_grad_by_beta(self, params, model, x, target):
        torch.manual_seed(42)
        loss = nn.functional.cross_entropy(model(x), target)
        loss.backward()
        grads_before = [p.grad.clone() for p in params if p.grad is not None]
        opt = FiniteNudgeEqProp(params, model, lr=0.0, beta=3.0)
        opt.step(x, target)
        for gb, p in zip(grads_before, params):
            if p.grad is not None:
                assert torch.allclose(p.grad, gb * 3.0, atol=1e-6)

    def test_step_raises_no_target(self, params, model, x):
        opt = FiniteNudgeEqProp(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)


# =============================================================================
# LazyEqProp Tests
# =============================================================================


class TestLazyEqProp:
    """Tests for LazyEqProp wrapper."""

    def test_step_updates_on_first_call(self, params, model, x, target):
        torch.manual_seed(42)
        opt = LazyEqProp(params, model, lr=0.1)
        assert opt.last_inputs is None
        old = [p.clone() for p in params]
        opt.step(x, target)
        assert opt.last_inputs is not None
        assert any(not torch.equal(o, p) for o, p in zip(old, params)), (
            "LazyEqProp should update on first call"
        )

    def test_should_update_returns_true_large_change(self, params, model):
        opt = LazyEqProp(params, model, threshold=0.01)
        x1 = torch.randn(4, 8)
        x2 = x1 + 10.0
        opt.last_inputs = x1
        assert opt._should_update(x2)

    def test_should_update_returns_false_small_change(self, params, model, x):
        opt = LazyEqProp(params, model, threshold=100.0)
        opt.last_inputs = x
        assert not opt._should_update(x.clone())

    def test_step_no_target_does_nothing(self, params, model, x):
        opt = LazyEqProp(params, model, lr=0.1)
        old = [p.clone() for p in params]
        opt.step(x)
        assert opt.last_inputs is not None
        assert all(torch.equal(o, p) for o, p in zip(old, params)), (
            "params should not change when target is None"
        )
