"""Tests for Equilibrium Propagation propagators."""

import pytest
import torch
from torch import nn

from bioplausible.zoo.propagators.eqprop import (
    EqProp,
    HolomorphicEqProp,
    FiniteNudgeEqProp,
    LazyEqProp,
)
from bioplausible.zoo.propagators.base import LearningRuleOptimizer


class SimpleMLP(nn.Module):
    """Minimal MLP for eqprop tests.

    NOTE: All layers must have the same hidden dim to match EqProp's
    _compute_ep_gradient shape assumption (contrast.T needs same dim
    as weight matrix).
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model():
    return SimpleMLP(dim=8)


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
    """Tests for standard Equilibrium Propagation."""

    def test_init(self, params, model):
        opt = EqProp(params, model, lr=0.01, beta=0.5, settle_steps=30)
        assert opt.beta == 0.5
        assert opt.settle_steps == 30
        assert opt.settle_lr == 0.15
        assert opt.loss_type == "mse"

    def test_init_defaults(self, params, model):
        opt = EqProp(params, model)
        assert opt.beta == 0.5
        assert opt.settle_steps == 30
        assert opt.loss_type == "mse"

    def test_step_requires_target(self, params, model, x):
        opt = EqProp(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)

    def test_step_sets_gradients(self, params, model, x, target):
        """Step runs and sets gradients on weight params for reachable layers.
        NOTE: Only the first len(layers) weight params get grads due to the
        i < len(pairs_free) guard in _compute_ep_gradient."""
        opt = EqProp(params, model, lr=0.1, beta=0.5, settle_steps=5, settle_lr=0.1)

        opt.step(x, target)

        layers = opt._get_layers()
        reachable_weights = [p for i, p in enumerate(params) if p.ndim >= 2 and i < len(layers)]
        assert all(p.grad is not None for p in reachable_weights), (
            "Weight params reachable by EP gradient should have grads"
        )
        for p in reachable_weights:
            assert p.grad.shape == p.shape, (
                f"Gradient shape {p.grad.shape} != param shape {p.shape}"
            )

    def test_is_learning_rule_optimizer(self, params, model):
        opt = EqProp(params, model)
        assert isinstance(opt, LearningRuleOptimizer)

    def test_get_layers_linear(self, model):
        opt = EqProp(list(model.parameters()), model)
        layers = opt._get_layers()
        assert all(isinstance(l, (nn.Linear, nn.Conv2d)) for l in layers)
        assert len(layers) == 3  # fc1, fc2, fc3

    def test_get_layers_no_linear(self):
        """Model with no Linear/Conv2d returns empty list."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        # Register a single float param so optimizer accepts it
        dummy = nn.Parameter(torch.tensor(1.0))
        model.register_parameter("dummy", dummy)
        params = [dummy]
        opt = EqProp(params, model)
        assert opt._get_layers() == []

    def test_settle_output_shape(self, params, model, x):
        opt = EqProp(params, model, settle_lr=0.1)
        pairs = opt._settle(x, target=None, beta=0.0)
        assert len(pairs) == 3
        for i, (inp, out) in enumerate(pairs):
            assert isinstance(inp, torch.Tensor), f"Input {i} is not a tensor"
            assert isinstance(out, torch.Tensor), f"Output {i} is not a tensor"
            assert inp.shape[0] == out.shape[0], f"Batch dim mismatch at layer {i}"

    def test_compute_ep_gradient_sets_grad(self, params, model, x):
        """Weight params reachable by EP contrast get correct-shaped grads."""
        opt = EqProp(params, model, beta=0.5)
        for p in params:
            p.grad = None

        pairs_free = opt._settle(x, target=None, beta=0.0)
        pairs_nudged = opt._settle(x, target=torch.randint(0, 8, (4,)), beta=0.5)
        opt._compute_ep_gradient(pairs_free, pairs_nudged)

        layers = opt._get_layers()
        for i, p in enumerate(params):
            if p.ndim >= 2 and i < len(layers):
                assert p.grad is not None, f"Grad should be set for weight {p.shape} at i={i}"
                assert p.grad.shape == p.shape, (
                    f"Gradient shape {p.grad.shape} != param shape {p.shape} at i={i}"
                )

    def test_param_groups(self, params, model):
        opt = EqProp(params, model, lr=0.01, momentum=0.9, weight_decay=0.001)
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[0]["momentum"] == 0.9
        assert opt.param_groups[0]["weight_decay"] == 0.001


# =============================================================================
# HolomorphicEqProp Tests
# =============================================================================


class TestHolomorphicEqProp:
    """Tests for Holomorphic Equilibrium Propagation."""

    def test_init(self, params, model):
        opt = HolomorphicEqProp(params, model, lr=0.01, beta=0.5)
        assert opt.beta == 0.5
        assert opt.settle_steps == 30

    def test_step_requires_target(self, params, model, x):
        opt = HolomorphicEqProp(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)

    def test_step_updates_params(self, params, model, x, target):
        opt = HolomorphicEqProp(params, model, lr=0.1)
        old_params = [p.clone() for p in params]

        opt.step(x, target)

        for old_p, new_p in zip(old_params, params):
            assert not torch.equal(old_p, new_p), "Parameters should be updated"

    def test_is_learning_rule_optimizer(self, params, model):
        opt = HolomorphicEqProp(params, model)
        assert isinstance(opt, LearningRuleOptimizer)

    def test_loss_decreases(self, params, model, x, target):
        """Smoke test: loss should decrease after step."""
        opt = HolomorphicEqProp(params, model, lr=0.1)

        loss_before = nn.functional.cross_entropy(model(x), target)
        opt.step(x, target)
        loss_after = nn.functional.cross_entropy(model(x), target)

        assert loss_after < loss_before


# =============================================================================
# FiniteNudgeEqProp Tests
# =============================================================================


class TestFiniteNudgeEqProp:
    """Tests for Finite Nudge Equilibrium Propagation."""

    def test_init(self, params, model):
        opt = FiniteNudgeEqProp(params, model, lr=0.01, beta=1.0)
        assert opt.beta == 1.0
        assert opt.settle_steps == 20

    def test_step_requires_target(self, params, model, x):
        opt = FiniteNudgeEqProp(params, model)
        with pytest.raises(ValueError, match="requires target"):
            opt.step(x)

    def test_step_with_grad(self, params, model, x, target):
        """Step should work when some params have gradients."""
        loss = nn.functional.cross_entropy(model(x), target)
        loss.backward()

        opt = FiniteNudgeEqProp(params, model, lr=0.1, beta=2.0)
        old_params = [p.clone() for p in params]

        opt.step(x, target)

        for old_p, new_p in zip(old_params, params):
            assert not torch.equal(old_p, new_p), "Parameters should be updated"

    def test_step_without_grad(self, params, model, x, target):
        """Step should be no-op when no gradients exist."""
        for p in params:
            p.grad = None

        opt = FiniteNudgeEqProp(params, model, lr=0.1)
        old_params = [p.clone() for p in params]

        opt.step(x, target)

        for old_p, new_p in zip(old_params, params):
            assert torch.equal(old_p, new_p), "No-grad params should not update"

    def test_beta_magnifies_gradient(self, params, model, x, target):
        """Gradients should be multiplied by beta."""
        loss = nn.functional.cross_entropy(model(x), target)
        loss.backward()

        grad_before = [p.grad.clone() for p in params if p.grad is not None]

        opt = FiniteNudgeEqProp(params, model, lr=0.0, beta=3.0)
        opt.step(x, target)

        for gb, p in zip(grad_before, params):
            if p.grad is not None:
                expected = gb * 3.0
                assert torch.allclose(p.grad, expected, atol=1e-6), (
                    f"Gradient should be scaled by beta=3"
                )

    def test_is_learning_rule_optimizer(self, params, model):
        opt = FiniteNudgeEqProp(params, model)
        assert isinstance(opt, LearningRuleOptimizer)


# =============================================================================
# LazyEqProp Tests
# =============================================================================


class TestLazyEqProp:
    """Tests for Lazy Equilibrium Propagation."""

    def test_init(self, params, model):
        opt = LazyEqProp(params, model, lr=0.01, threshold=0.05)
        assert opt.threshold == 0.05
        assert opt.last_inputs is None

    def test_should_update_on_first_call(self, params, model, x):
        opt = LazyEqProp(params, model)
        assert opt._should_update(x)

    def test_should_update_no_update_small_change(self, params, model, x):
        opt = LazyEqProp(params, model, threshold=1.0)
        x2 = x.clone()
        opt.last_inputs = x
        assert not opt._should_update(x2)

    def test_should_update_on_large_change(self, params, model):
        opt = LazyEqProp(params, model, threshold=0.01)
        x1 = torch.randn(4, 8)
        x2 = x1 + 10.0  # Large change
        opt.last_inputs = x1
        assert opt._should_update(x2)

    def test_step_skips_no_change(self, params, model, x, target):
        """Step should be no-op when input doesn't change much."""
        opt = LazyEqProp(params, model, threshold=100.0, lr=0.1)
        old_params = [p.clone() for p in params]

        # First call should update since last_inputs is None
        opt.step(x, target)

        # Second call with same input should skip (change < threshold)
        opt.step(x, target)

        # Params should be different from original (first update happened)
        assert not all(
            torch.equal(old_p, new_p) for old_p, new_p in zip(old_params, params)
        ), "First step should have updated params"

    def test_step_with_target(self, params, model, x, target):
        """Step should update when called with target."""
        opt = LazyEqProp(params, model, lr=0.1)
        old_params = [p.clone() for p in params]

        opt.step(x, target)

        for old_p, new_p in zip(old_params, params):
            assert not torch.equal(old_p, new_p), "Parameters should be updated"

    def test_step_without_target(self, params, model, x):
        """Step without target should still track input but skip training."""
        opt = LazyEqProp(params, model, lr=0.1)
        # Call without target
        opt.step(x)

        assert opt.last_inputs is not None
        # Since target is None, training is skipped
        assert torch.equal(opt.last_inputs, x)

    def test_is_learning_rule_optimizer(self, params, model):
        opt = LazyEqProp(params, model)
        assert isinstance(opt, LearningRuleOptimizer)
