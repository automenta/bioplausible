"""Tests for the ContrastiveHebbianLearning propagator wrapper.

Covers: ContrastiveHebbianLearning in bioplausible.zoo.propagators.hebbian.
"""

import pytest
import torch
from torch import nn

from bioplausible.zoo.models.transitions import TransitionGraphMixin
from bioplausible.zoo.propagators.hebbian import ContrastiveHebbianLearning

# =============================================================================
# Fixtures
# =============================================================================


class SimpleMLP(TransitionGraphMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

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

    def test_forward_capture_returns_states(self, params, model, x):
        opt = ContrastiveHebbianLearning(params, model)
        states = opt._forward_capture(x)
        assert isinstance(states, list)
        # states[0] = input, states[1] = after layer1+relu, states[2] = after layer2+relu
        assert len(states) == 3
        assert states[0].shape == (4, 8)
        assert states[1].shape == (4, 16)
        assert states[2].shape == (4, 4)

    def test_get_layers_returns_weighted_only(self, params, model):
        opt = ContrastiveHebbianLearning(params, model)
        layers = opt._get_transitions()
        assert len(layers) == 2  # Linear(8,16), Linear(16,4)
        assert all(isinstance(l, (nn.Linear, nn.Conv2d)) for l in layers)

    def test_hebbian_update_sets_grad(self, params, model, x):
        torch.manual_seed(42)
        opt = ContrastiveHebbianLearning(params, model, lr=0.1)
        for p in params:
            p.grad = None
        free = opt._forward_capture(x)
        clamped = opt._forward_capture(x)
        opt._hebbian_update(free, clamped)
        weight_layers = [l for l in opt._get_transitions() if hasattr(l, "weight")]
        for l in weight_layers:
            assert l.weight.grad is not None, f"weight.grad should be set for layer {l}"
            assert l.weight.grad.shape == l.weight.shape
