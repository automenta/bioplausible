"""Tests for TargetPropGradient and HebbianGradient strategies."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn

from computronium.core.optimization import (
    HebbianGradient,
    StrategyConfig,
    StrategyOptimizer,
    StrategyOptimizerConfig,
    TargetPropGradient,
    create_strategy_optimizer,
)


class _DTPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(in_features, out_features), nn.Tanh()
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(out_features, in_features), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_net(x)


class _TargetPropModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _DTPLayer(input_dim, hidden_dim),
            _DTPLayer(hidden_dim, hidden_dim),
        ])
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer.forward_net(h)
        return self.out_layer(h)


class _HebbianLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_oja: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_oja = use_oja
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight, gain=1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight.addmm_(y.T, x, alpha=self.use_oja / (y.size(0) * self.use_oja))


class _HebbianModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hebbian_lr = 0.01
        self.use_oja = True
        self.layers = nn.ModuleList([
            _HebbianLayer(input_dim, hidden_dim),
            _HebbianLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ])

    def transition_modules(self) -> list[nn.Module]:
        return list(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class TestTargetPropGradient:
    def test_output_target_computation(self) -> None:
        model = _TargetPropModel(4, 8, 3)
        grad = TargetPropGradient(loss_fn=nn.CrossEntropyLoss(), target_lr=0.1)
        x = torch.randn(2, 4)
        y = torch.tensor([0, 2])
        grad.compute_gradients(model, x, y)
        # All params must have grads
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_requires_loss_fn(self) -> None:
        model = _TargetPropModel(4, 8, 3)
        grad = TargetPropGradient()
        x = torch.randn(2, 4)
        y = torch.tensor([0, 2])
        try:
            grad.compute_gradients(model, x, y)
        except ValueError as e:
            assert "loss_fn" in str(e)
        else:
            raise AssertionError("should require loss_fn")

    def test_requires_target(self) -> None:
        model = _TargetPropModel(4, 8, 3)
        grad = TargetPropGradient(loss_fn=nn.CrossEntropyLoss())
        x = torch.randn(2, 4)
        try:
            grad.compute_gradients(model, x, None)
        except ValueError as e:
            assert "target" in str(e)
        else:
            raise AssertionError("should require target")

    def test_unsupported_model(self) -> None:
        model = nn.Linear(4, 3)
        grad = TargetPropGradient(loss_fn=nn.CrossEntropyLoss())
        try:
            grad.compute_gradients(model, torch.randn(2, 4), torch.tensor([0, 1]))
        except ValueError as e:
            assert "layers" in str(e)
        else:
            raise AssertionError("should reject non-target-prop model")


class TestHebbianGradient:
    def test_local_hebbian_update(self) -> None:
        model = _HebbianModel(4, 8, 3)
        grad = HebbianGradient()
        x = torch.randn(2, 4)
        y = torch.tensor([1, 0])
        before = [p.clone() for p in model.parameters()]
        grad.compute_gradients(model, x, y)
        # Hebbian layers changed via in-place updates
        assert any(not torch.allclose(b, p) for b, p in zip(before, model.parameters()))

    def test_requires_transition_modules(self) -> None:
        model = nn.Linear(4, 3)
        grad = HebbianGradient()
        try:
            grad.compute_gradients(model, torch.randn(2, 4), torch.tensor([0, 1]))
        except (ValueError, AttributeError) as e:
            assert "transition_modules" in str(e)
        else:
            raise AssertionError("should require transition_modules")

    def test_requires_target(self) -> None:
        model = _HebbianModel(4, 8, 3)
        grad = HebbianGradient()
        try:
            grad.compute_gradients(model, torch.randn(2, 4), None)
        except ValueError as e:
            assert "target" in str(e)
        else:
            raise AssertionError("should require target")


class TestFactoryIntegration:
    def _make_model(self) -> nn.Module:
        return _TargetPropModel(4, 8, 3)

    def test_factory_creates_target_prop_optimizer(self) -> None:
        model = self._make_model()
        config = StrategyOptimizerConfig(
            gradient=StrategyConfig(
                name="target_prop", kwargs={"loss_fn": nn.CrossEntropyLoss()}
            ),
            update=StrategyConfig(name="plain"),
            lr=0.001,
        )
        opt = create_strategy_optimizer(config, model=model)
        assert isinstance(opt, StrategyOptimizer)
        assert isinstance(opt.gradient, TargetPropGradient)

    def test_factory_creates_hebbian_optimizer(self) -> None:
        model = _HebbianModel(4, 8, 3)
        config = StrategyOptimizerConfig(
            gradient=StrategyConfig(name="hebbian"),
            update=StrategyConfig(name="plain"),
            lr=0.001,
        )
        opt = create_strategy_optimizer(config, model=model)
        assert isinstance(opt, StrategyOptimizer)
        assert isinstance(opt.gradient, HebbianGradient)

    def test_unknown_gradient_registers(self) -> None:
        # Ensure the new names are resolvable via the registry.
        from computronium.core.optimization.factory import StrategyRegistry

        assert "target_prop" in StrategyRegistry
        assert "hebbian" in StrategyRegistry
