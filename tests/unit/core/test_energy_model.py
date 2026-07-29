"""Tests for EnergyModel protocol and EBMTrainer."""

from typing import Protocol

import torch

from bioplausible.core.energy_model import EBMTrainer, EnergyModel


class _DummyEnergyModel(torch.nn.Module):
    """Minimal EnergyModel implementation for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)

    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.fc(x)
        if y is not None:
            return torch.nn.functional.cross_entropy(logits, y)
        return logits.norm()

    def settle(
        self,
        x: torch.Tensor,
        steps: int,
        beta: float = 0.0,
        y: torch.Tensor | None = None,
    ) -> object:
        return self.fc(x).detach()

    def contrastive_update(
        self,
        free_state: object,
        nudged_state: object,
        beta: float,
        lr: float = 1.0,
    ) -> None:
        pass


def test_energy_model_is_protocol() -> None:
    """EnergyModel should be a runtime-checkable Protocol."""
    assert isinstance(EnergyModel, type)
    assert issubclass(EnergyModel, Protocol)


def test_dummy_model_satisfies_protocol() -> None:
    """A class with all three required methods should satisfy the protocol."""
    model = _DummyEnergyModel()
    assert isinstance(model, EnergyModel)


class _MissingMethod(torch.nn.Module):
    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return x.norm()


def test_incomplete_model_does_not_satisfy_protocol() -> None:
    """A class missing settle() or contrastive_update() should not match."""
    model = _MissingMethod()
    assert not isinstance(model, EnergyModel)


def test_ebm_trainer_fallback_bptt() -> None:
    """EBMTrainer.train_step should fall back to BPTT for non-EnergyModel."""
    model = torch.nn.Linear(10, 5)
    trainer = EBMTrainer(model, lr=0.01)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    metrics = trainer.train_step(x, y)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_ebm_trainer_dispatch() -> None:
    """EBMTrainer with EnergyModel should run free/nudge/contrastive."""
    model = _DummyEnergyModel()
    trainer = EBMTrainer(model, lr=0.01, free_steps=3, nudged_steps=2, beta=0.05)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    metrics = trainer.train_step(x, y)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_ebm_fallback_metrics_valid() -> None:
    """Non-EnergyModel should get BPTT fallback with valid metrics."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
    )
    trainer = EBMTrainer(model, lr=0.01, clip_grad_norm=1.0)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    metrics = trainer.train_step(x, y)
    assert 0 <= metrics["accuracy"] <= 1.0
