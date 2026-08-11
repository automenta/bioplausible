"""End-to-end wiring tests: concrete zoo model → concrete strategy permutation.

Validates ``requires_energy`` forwarding (``StrategyOptimizer.step`` calls
``compute_gradients`` with ``x`` and ``target``) and ``target_lr`` / ``loss_fn``
plumbing through ``create_strategy_optimizer``.
"""

import pytest
import torch
from torch import nn

from bioplausible.core.optimization import (
    StrategyConfig,
    StrategyOptimizerConfig,
    create_strategy_optimizer,
)
from bioplausible.zoo.models.fa import StandardFA
from bioplausible.zoo.models.target_prop import DifferenceTargetProp


class TestTargetPropWiring:
    """DifferenceTargetProp → TargetPropGradient permutation."""

    @pytest.fixture
    def model(self):
        return DifferenceTargetProp(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            num_layers=2,
        )

    @pytest.fixture
    def x_y(self):
        torch.manual_seed(42)
        return torch.randn(4, 4), torch.tensor([0, 1, 2, 1])

    def _make_optimizer(self, model, target_lr=0.1, lr=0.05):
        config = StrategyOptimizerConfig(
            gradient=StrategyConfig(
                name="target_prop",
                kwargs={
                    "loss_fn": model.criterion,
                    "target_lr": target_lr,
                },
            ),
            update=StrategyConfig(name="plain"),
            lr=lr,
            momentum=0.0,
            weight_decay=0.0,
        )
        return create_strategy_optimizer(config, model=model)

    def test_requires_energy_forwarded(self, model, x_y):
        """step(x=, target=) is forwarded to compute_gradients."""
        x, y = x_y
        opt = self._make_optimizer(model)

        assert opt.gradient.requires_energy is True

        w_before = model.out_layer.weight.detach().clone()
        opt.step(x=x, target=y)
        w_after = model.out_layer.weight.detach().clone()

        assert not torch.allclose(w_before, w_after)

    def test_target_lr_plumbing(self, model):
        """target_lr reaches the strategy constructor."""
        opt = self._make_optimizer(model, target_lr=0.999)
        assert opt.gradient.target_lr == pytest.approx(0.999)

    def test_loss_fn_plumbing(self, model):
        """loss_fn from the strategy reaches compute_gradients."""
        opt = self._make_optimizer(model, target_lr=0.1)
        assert opt.gradient.loss_fn is model.criterion

    def test_returns_none_loss(self, model, x_y):
        """step returns None when no closure (energy path)."""
        x, y = x_y
        opt = self._make_optimizer(model)
        result = opt.step(x=x, target=y)
        assert result is None

    def test_missing_x_raises(self, model, x_y):
        """Without x, energy-based strategy raises."""
        _, y = x_y
        opt = self._make_optimizer(model)
        with pytest.raises(ValueError, match="require an x tensor"):
            opt.step(target=y)

    def test_missing_target_raises(self, model, x_y):
        """Without target, energy-based strategy raises."""
        x, _ = x_y
        opt = self._make_optimizer(model)
        with pytest.raises(ValueError, match="require a target"):
            opt.step(x=x)


class TestHebbianWiring:
    """StandardFA + HebbianGradient(use_oja) permutations."""

    @pytest.fixture
    def model(self):
        return StandardFA(
            input_dim=4,
            hidden_dim=6,
            output_dim=3,
        )

    @pytest.fixture
    def x_y(self):
        torch.manual_seed(0)
        return torch.randn(4, 4), torch.tensor([0, 1, 2, 0])

    def _make_optimizer(self, model, use_oja=True, lr=0.05):
        config = StrategyOptimizerConfig(
            gradient=StrategyConfig(
                name="hebbian",
                kwargs={"hebbian_lr": 0.05, "use_oja": use_oja},
            ),
            update=StrategyConfig(name="plain"),
            lr=lr,
            momentum=0.0,
            weight_decay=0.0,
        )
        return create_strategy_optimizer(config, model=model)

    @pytest.mark.parametrize("use_oja", [True, False])
    def test_oja_plumbing(self, model, use_oja):
        """use_oja flag reaches HebbianGradient constructor."""
        opt = self._make_optimizer(model, use_oja=use_oja)
        assert opt.gradient.use_oja is use_oja

    def test_requires_energy_forwarded(self, model, x_y):
        """step(x=, target=) is forwarded to HebbianGradient.compute_gradients."""
        x, y = x_y
        opt = self._make_optimizer(model)

        assert opt.gradient.requires_energy is True

        head_before = model.layers[-1].weight.detach().clone()
        opt.step(x=x, target=y)
        head_after = model.layers[-1].weight.detach().clone()

        assert not torch.allclose(head_before, head_after)

    def test_hebbian_lr_from_model(self, model):
        """HebbianGradient falls back to model.hebbian_lr."""
        model.hebbian_lr = 0.42
        opt = self._make_optimizer(model, lr=0.01)
        assert opt.gradient._get_hebbian_lr(model) == pytest.approx(0.42)

    def test_transition_modules_discovered(self, model):
        """BioModel auto-discovers self.layers as transition_modules."""
        modules = model.transition_modules()
        assert len(modules) == len(model.layers)
        assert all(isinstance(m, nn.Linear) for m in modules)


class TestBackpropWiring:
    """StandardFA + BackpropGradient via closure (requires_energy=False)."""

    @pytest.fixture
    def model(self):
        return StandardFA(
            input_dim=4,
            hidden_dim=6,
            output_dim=3,
        )

    @pytest.fixture
    def x_y(self):
        torch.manual_seed(1)
        return torch.randn(4, 4), torch.tensor([0, 1, 2, 0])

    def test_backprop_via_closure(self, model, x_y):
        """BackpropGradient (requires_energy=False) uses closure path."""
        x, y = x_y
        config = StrategyOptimizerConfig(
            gradient=StrategyConfig(
                name="backprop",
                kwargs={"loss_fn": nn.CrossEntropyLoss()},
            ),
            update=StrategyConfig(name="plain"),
            lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
        )
        opt = create_strategy_optimizer(config, model=model)

        assert getattr(opt.gradient, "requires_energy", False) is False

        w_before = model.layers[-1].weight.detach().clone()

        def closure():
            opt.zero_grad()
            logits = model(x)
            loss = opt.gradient.loss_fn(logits, y)
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            return float(loss.item())

        opt.step(closure=closure)
        w_after = model.layers[-1].weight.detach().clone()

        assert not torch.allclose(w_before, w_after)
