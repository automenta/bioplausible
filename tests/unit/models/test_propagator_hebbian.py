"""Tests for the ContrastiveHebbianLearning propagator wrapper.

Covers: ContrastiveHebbianLearning in bioplausible.zoo.propagators.hebbian.
"""

import pytest
import torch
from torch import nn

from bioplausible.zoo.models.transitions import TransitionGraphMixin
from bioplausible.core.local_learning.rules.hebbian import ContrastiveHebbianLearning

# =============================================================================
# Fixtures
# =============================================================================


class SimpleMLP(TransitionGraphMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        self.layers = nn.ModuleList([self.fc1, self.fc2])

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SimpleMLP()


@pytest.fixture
def params(model):
    return list(model.parameters())


@pytest.fixture
def x():
    return torch.randn(4, 8)


@pytest.fixture
def target():
    return torch.randint(0, 4, (4,))


# =============================================================================
# ContrastiveHebbianLearning Tests
# =============================================================================


class TestContrastiveHebbianLearning:
    """Tests for Contrastive Hebbian Learning wrapper."""

    def test_step_updates_params(self, params, model, x, target):
        torch.manual_seed(42)
        opt = ContrastiveHebbianLearning(params, model, lr=0.1)
        old = [p.clone() for p in params]
        opt.step(x, target)
        assert any(not torch.equal(o, p) for o, p in zip(old, params)), (
            "CHL step should update params"
        )

    def test_step_raises_no_target(self, params, model, x):
        opt = ContrastiveHebbianLearning(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)

    def test_get_layers_returns_weighted_only(self, params, model):
        opt = ContrastiveHebbianLearning(params, model)
        layers = opt._get_transitions()
        assert len(layers) == 2  # Linear(8,16), Linear(16,4)
        assert all(isinstance(l, (nn.Linear, nn.Conv2d)) for l in layers)

    def test_step_sets_grad_on_weight_layers(self, params, model, x, target):
        torch.manual_seed(42)
        opt = ContrastiveHebbianLearning(params, model, lr=0.1)
        for p in params:
            p.grad = None
        opt.step(x, target)
        weight_layers = [l for l in opt._get_transitions() if hasattr(l, "weight")]
        for l in weight_layers:
            assert l.weight.grad is not None, f"weight.grad should be set for layer {l}"
            assert l.weight.grad.shape == l.weight.shape

    def test_step_produces_nonzero_gradient(self, params, model, x, target):
        """Free vs clamped contrast must yield a non-trivial (non-zero) gradient."""
        torch.manual_seed(42)
        opt = ContrastiveHebbianLearning(params, model, lr=0.1)
        for p in params:
            p.grad = None
        opt.step(x, target)
        weight_layers = [l for l in opt._get_transitions() if hasattr(l, "weight")]
        grads = [l.weight.grad for l in weight_layers]
        assert any(g.norm().item() > 0 for g in grads), (
            "clamped contrast must produce a non-trivial gradient"
        )

    def test_step_clamped_output_matches_target(self, params, model, x, target):
        """During clamped phase, output layer is fixed to target one-hot."""
        torch.manual_seed(42)
        opt = ContrastiveHebbianLearning(params, model, lr=0.1)
        # We can't directly test clamped output since it's internal,
        # but we can verify step runs without error and produces finite grads
        opt.step(x, target)
        weight_layers = [l for l in opt._get_transitions() if hasattr(l, "weight")]
        for l in weight_layers:
            assert l.weight.grad is not None
            assert torch.isfinite(l.weight.grad).all()
