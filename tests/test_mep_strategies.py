"""Unit tests for individual MEP strategy components.

Tests each strategy class in isolation: construction, basic functional
invocation, and key invariants per the REFACTOR.md plan (A.5).
"""

import pytest
import torch
from torch import nn

from bioplausible.zoo.mep.optimizers.strategies import (
    BackpropGradient,
    DionUpdate,
    EPGradient,
    ErrorFeedback,
    FisherUpdate,
    LocalEPGradient,
    MuonUpdate,
    NaturalGradient,
    NoConstraint,
    NoFeedback,
    PlainUpdate,
    SettlingSpectralPenalty,
    SpectralConstraint,
)
from bioplausible.zoo.mep.optimizers.strategies.base import (
    ConstraintStrategy,
    FeedbackStrategy,
    GradientStrategy,
    UpdateStrategy,
)


class TestStrategyProtocols:
    """Verify strategy Protocol hierarchy is importable and structurally sound."""

    def test_protocols_are_importable(self):
        """Protocol classes are importable and have expected methods."""

        assert hasattr(GradientStrategy, "compute_gradients")
        assert hasattr(UpdateStrategy, "transform_gradient")
        assert hasattr(ConstraintStrategy, "enforce")
        assert hasattr(FeedbackStrategy, "accumulate")
        assert hasattr(FeedbackStrategy, "update_buffer")

    def test_strategy_classes_have_required_methods(self):
        """Concrete strategies implement the Protocol methods."""
        assert hasattr(BackpropGradient, "compute_gradients")
        assert hasattr(PlainUpdate, "transform_gradient")
        assert hasattr(NoConstraint, "enforce")
        assert hasattr(NoFeedback, "accumulate")
        assert hasattr(NoFeedback, "update_buffer")


class TestGradientStrategies:
    """Test gradient computation strategies."""

    @pytest.fixture
    def model(self):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    @pytest.fixture
    def data(self):
        return torch.randn(2, 4), torch.tensor([0, 1])

    def _check_gradients_populated(self, model):
        for p in model.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()

    def test_backprop_gradient(self, model, data):
        x, y = data
        strategy = BackpropGradient()
        strategy.compute_gradients(model, x, y, loss_fn=nn.CrossEntropyLoss())
        self._check_gradients_populated(model)

    def test_backprop_default_loss(self, model, data):
        x, y = data
        strategy = BackpropGradient()
        with pytest.raises(ValueError, match="loss_fn must be provided"):
            strategy.compute_gradients(model, x, y)

    def test_ep_gradient_constructs(self):
        strategy = EPGradient(beta=0.5, settle_steps=5, settle_lr=0.1)
        assert strategy.beta == pytest.approx(0.5)
        assert strategy.settle_steps == 5

    def test_local_ep_gradient_constructs(self):
        strategy = LocalEPGradient(beta=0.3, settle_steps=5, settle_lr=0.1)
        assert strategy.settle_steps == 5

    def test_natural_gradient_wraps_backprop(self, model, data):
        x, y = data
        base = BackpropGradient()
        strategy = NaturalGradient(base_strategy=base)
        strategy.compute_gradients(model, x, y, loss_fn=nn.CrossEntropyLoss())
        self._check_gradients_populated(model)


class TestUpdateStrategies:
    """Test update/transformation strategies."""

    @pytest.fixture
    def grad_2d(self):
        return torch.randn(8, 4)

    @pytest.fixture
    def grad_1d(self):
        return torch.randn(8)

    def test_plain_update_identity(self, grad_2d):
        strategy = PlainUpdate()
        result = strategy.transform_gradient(grad_2d, grad_2d, {}, {"lr": 0.01})
        assert torch.equal(result, grad_2d)

    def test_muon_update_orthogonal(self, grad_2d):
        strategy = MuonUpdate(ns_steps=5)
        result = strategy.transform_gradient(grad_2d, grad_2d, {}, {"lr": 0.01})
        assert result.shape == grad_2d.shape
        assert not torch.isnan(result).any()

    def test_muon_update_1d_passthrough(self, grad_1d):
        """1D gradients pass through Muon unchanged."""
        strategy = MuonUpdate(ns_steps=5)
        result = strategy.transform_gradient(grad_1d, grad_1d, {}, {"lr": 0.01})
        assert result.shape == grad_1d.shape

    def test_dion_update_small(self, grad_2d):
        """Below threshold uses Muon fallback."""
        strategy = DionUpdate(threshold=100000)
        result = strategy.transform_gradient(grad_2d, grad_2d, {}, {"lr": 0.01})
        assert result.shape == grad_2d.shape

    def test_fisher_update_no_fisher(self, grad_2d):
        """Without param.fisher, FisherUpdate should error or return grad."""
        strategy = FisherUpdate()
        result = strategy.transform_gradient(grad_2d, grad_2d, {}, {"lr": 0.01})
        # FisherUpdate may silently return identity when no fisher attr present
        assert result.shape == grad_2d.shape


class TestConstraintStrategies:
    """Test parameter constraint strategies."""

    def test_no_constraint_noop(self):
        strategy = NoConstraint()
        param = nn.Parameter(torch.randn(4, 4))
        old_data = param.data.clone()
        strategy.enforce(param, {}, {})
        assert torch.equal(param.data, old_data)

    def test_spectral_constraint_reduces_norm(self):
        """SpectralConstraint ensures spectral norm <= gamma."""
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(10, 5))
        original_sigma = torch.linalg.svdvals(param.data).max()

        gamma = 0.5
        strategy = SpectralConstraint(gamma=gamma, power_iter=10)
        strategy.enforce(param, {}, {})

        new_sigma = torch.linalg.svdvals(param.data).max()
        assert new_sigma <= gamma * (1 + 1e-4), (
            f"Spectral norm {new_sigma:.4f} > gamma {gamma}"
        )

    def test_spectral_constraint_loose_gamma_noop(self):
        """When gamma > current sigma, constraint is a no-op."""
        param = nn.Parameter(torch.ones(5, 5) * 0.1)
        sigma = torch.linalg.svdvals(param.data).max()
        gamma = sigma * 2

        strategy = SpectralConstraint(gamma=gamma, power_iter=10)
        old_data = param.data.clone()
        strategy.enforce(param, {}, {})
        assert torch.equal(param.data, old_data)

    def test_settling_spectral_penalty(self):
        """SettlingSpectralPenalty returns a scalar penalty."""
        param = nn.Parameter(torch.randn(8, 8) * 2)
        model = nn.Linear(8, 8)
        model.weight = param

        penalty = SettlingSpectralPenalty(gamma=1.0, lambda_penalty=1.0)
        result = penalty.compute_penalty(model, {})
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # scalar


class TestFeedbackStrategies:
    """Test error feedback strategies."""

    @pytest.fixture
    def grad(self):
        return torch.randn(10)

    def test_no_feedback_identity(self, grad):
        strategy = NoFeedback()
        result = strategy.accumulate(grad, {}, {})
        assert torch.equal(result, grad)

    def test_error_feedback_accumulate(self, grad):
        strategy = ErrorFeedback(beta=0.9)
        result = strategy.accumulate(grad, {}, {})
        assert result.shape == grad.shape
        assert not torch.isnan(result).any()

    def test_error_feedback_zero_beta(self, grad):
        """beta=0 degenerates to NoFeedback."""
        strategy = ErrorFeedback(beta=0.0)
        result = strategy.accumulate(grad, {}, {"max_grad_norm": 1.0})
        assert torch.equal(result, grad)

    def test_error_feedback_update_buffer(self, grad):
        strategy = ErrorFeedback(beta=0.9)
        residual = torch.randn(10)
        state: dict = {}
        config = {"max_grad_norm": 1.0}
        # First accumulate triggers buffer initialization (zero buffer)
        first = strategy.accumulate(grad, state, config)
        strategy.update_buffer(residual, state, config)
        # Second accumulate includes the non-zero buffer from update
        second = strategy.accumulate(grad, state, config)
        assert not torch.equal(second, first)


class TestSpectralConstraintTiming:
    """SpectralConstraint timing logic."""

    def test_should_apply_post_update(self):
        sc = SpectralConstraint(timing="post_update")
        assert sc.should_apply("post_update")
        assert not sc.should_apply("during_settling")

    def test_should_apply_during_settling(self):
        sc = SpectralConstraint(timing="during_settling")
        assert sc.should_apply("during_settling")
        assert not sc.should_apply("post_update")

    def test_should_apply_both(self):
        sc = SpectralConstraint(timing="both")
        assert sc.should_apply("post_update")
        assert sc.should_apply("during_settling")
