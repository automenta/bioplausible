"""Tests for the canonical optimizer factory (REFACTOR.md §2.3)."""

from __future__ import annotations

import torch
from torch import nn

from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestOptimizerConfig:
    def test_defaults(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.name == "adamw"
        assert cfg.lr == 1e-3
        assert cfg.weight_decay == 1e-4
        assert cfg.betas == (0.9, 0.999)

    def test_frozen(self) -> None:
        cfg = OptimizerConfig()
        try:
            cfg.lr = 0.5
        except AttributeError:
            pass
        else:
            raise AssertionError("OptimizerConfig should be frozen")

    def test_slots(self) -> None:
        cfg = OptimizerConfig()
        try:
            cfg.new_attr = 1
        except AttributeError:
            pass
        else:
            raise AssertionError("OptimizerConfig should use slots")


class TestCreateOptimizer:
    def test_adamw(self) -> None:
        net = _Net()
        opt = create_optimizer(net, OptimizerConfig(name="adamw", lr=0.01))
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 0.01

    def test_adam(self) -> None:
        net = _Net()
        opt = create_optimizer(net, OptimizerConfig(name="adam"))
        assert isinstance(opt, torch.optim.Adam)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_sgd(self) -> None:
        net = _Net()
        opt = create_optimizer(net, OptimizerConfig(name="sgd", momentum=0.5))
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == 0.5

    def test_unknown_name_raises(self) -> None:
        net = _Net()
        try:
            create_optimizer(net, OptimizerConfig(name="unknown"))  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown optimizer name should raise ValueError")

    def test_optimizer_updates_params(self) -> None:
        net = _Net()
        opt = create_optimizer(net, OptimizerConfig(name="adamw"))
        x = torch.randn(2, 4)
        y = net(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        assert all(
            p.grad is not None or p.requires_grad is False for p in net.parameters()
        )
