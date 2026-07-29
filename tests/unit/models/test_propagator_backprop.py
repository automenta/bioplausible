"""Tests for the Backprop propagator wrapper.

Covers: Backprop class in bioplausible.zoo.propagators.backprop.
"""

import pytest
import torch
from torch import nn

from bioplausible.zoo.propagators.backprop import Backprop

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


@pytest.fixture
def model_sigmoid():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
        nn.Sigmoid(),
    )


@pytest.fixture
def params(model):
    return list(model.parameters())


@pytest.fixture
def x():
    return torch.randn(4, 8)


@pytest.fixture
def target_cls():
    return torch.randint(0, 4, (4,))


@pytest.fixture
def target_mse():
    return torch.randn(4, 4)


@pytest.fixture
def target_bce():
    return (torch.rand(4, 4) > 0.5).float()


# =============================================================================
# Backprop Tests
# =============================================================================


class TestBackprop:
    """Tests for the Backprop propagator."""

    def test_step_cross_entropy(self, params, model, x, target_cls):
        torch.manual_seed(42)
        opt = Backprop(params, model, lr=0.1)
        old = [p.clone() for p in params]
        opt.step(x, target_cls)
        assert all(not torch.equal(o, p) for o, p in zip(old, params)), (
            "params should change after step"
        )

    def test_step_mse(self, params, model, x, target_mse):
        torch.manual_seed(42)
        opt = Backprop(params, model, lr=0.1, loss_fn="mse")
        old = [p.clone() for p in params]
        opt.step(x, target_mse)
        assert all(not torch.equal(o, p) for o, p in zip(old, params)), (
            "params should change after step (mse)"
        )

    def test_step_binary_cross_entropy(self, model_sigmoid, x, target_bce):
        torch.manual_seed(42)
        params = list(model_sigmoid.parameters())
        opt = Backprop(params, model_sigmoid, lr=0.1, loss_fn="binary_cross_entropy")
        old = [p.clone() for p in params]
        opt.step(x, target_bce)
        assert all(not torch.equal(o, p) for o, p in zip(old, params)), (
            "params should change after step (bce)"
        )

    def test_step_raises_no_target(self, params, model, x):
        opt = Backprop(params, model, lr=0.1)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)

    def test_step_raises_unknown_loss(self, params, model, x, target_cls):
        opt = Backprop(params, model, lr=0.1, loss_fn="not_a_loss")
        with pytest.raises(ValueError, match="Unknown loss function"):
            opt.step(x, target_cls)

    def test_zero_grad_clears_gradients(self, params, model, x, target_cls):
        opt = Backprop(params, model, lr=0.1)
        opt.step(x, target_cls)
        assert all(p.grad is not None for p in params if p.requires_grad)
        opt.zero_grad()
        assert all(
            p.grad is None or p.grad.sum().item() == pytest.approx(0.0) for p in params
        )

    def test_step_owns_backward(self, params, model, x, target_cls):
        """Step() owns the backward pass — no loss.backward() needed."""
        for p in params:
            p.grad = None
        opt = Backprop(params, model, lr=0.1)
        opt.step(x, target_cls)
        assert any(p.grad is not None for p in params), (
            "step should compute gradients internally"
        )

    @pytest.mark.parametrize(
        ("loss_fn", "model_fixture", "tgt_fixture"),
        [
            ("cross_entropy", "model", "target_cls"),
            ("mse", "model", "target_mse"),
            ("binary_cross_entropy", "model_sigmoid", "target_bce"),
        ],
    )
    def test_all_loss_fns(self, loss_fn, model_fixture, tgt_fixture, x, request):
        torch.manual_seed(42)
        model = request.getfixturevalue(model_fixture)
        tgt = request.getfixturevalue(tgt_fixture)
        params = list(model.parameters())
        opt = Backprop(params, model, lr=0.1, loss_fn=loss_fn)
        old = [p.clone() for p in params]
        opt.step(x, tgt)
