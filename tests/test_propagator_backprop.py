"""Tests for the Backprop propagator implementation."""

import pytest
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo.models.eqprop import LoopedMLP


def test_backprop_propagator_registered():
    """Backprop propagator is registered in the zoo."""
    cls = Registry.get(ComponentCategory.PROPAGATOR, "backprop")
    assert cls is not None
    assert cls.__name__ == "Backprop"


def test_backprop_propagator_step():
    """Backprop propagator step computes gradients and updates params."""
    model = LoopedMLP(10, 20, 5, max_steps=5)
    params = list(model.parameters())

    from bioplausible.zoo.propagators.backprop import Backprop

    optim = Backprop(params, model=model, lr=0.1)

    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    weights_before = {n: p.clone() for n, p in model.named_parameters()}

    optim.step(x, y)

    weights_after = {n: p.clone() for n, p in model.named_parameters()}

    # Verify weights changed
    changed = any(
        not torch.equal(weights_before[n], weights_after[n]) for n in weights_before
    )
    assert changed, "Backprop step should update model parameters"


def test_backprop_propagator_different_loss_fns():
    """Backprop propagator supports different loss functions."""
    model = LoopedMLP(10, 20, 5, max_steps=5)
    params = list(model.parameters())

    from bioplausible.zoo.propagators.backprop import Backprop

    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    # Test MSE loss
    optim_mse = Backprop(params, model=model, lr=0.01, loss_fn="mse")
    y_onehot = torch.zeros(4, 5).scatter_(1, y.unsqueeze(1), 1.0)
    optim_mse.step(x, y_onehot)  # Should not raise

    # Test cross_entropy loss (default)
    optim_ce = Backprop(params, model=model, lr=0.01)
    optim_ce.step(x, y)  # Should not raise


def test_backprop_propagator_no_target_raises():
    """Backprop propagator raises ValueError when target is None."""
    model = LoopedMLP(10, 20, 5, max_steps=5)
    params = list(model.parameters())

    from bioplausible.zoo.propagators.backprop import Backprop

    optim = Backprop(params, model=model, lr=0.01)
    x = torch.randn(4, 10)

    with pytest.raises(ValueError, match="requires target"):
        optim.step(x)


def test_backprop_propagator_unknown_loss_fn():
    """Unknown loss function raises ValueError."""
    model = LoopedMLP(10, 20, 5, max_steps=5)
    params = list(model.parameters())

    from bioplausible.zoo.propagators.backprop import Backprop

    optim = Backprop(params, model=model, lr=0.01, loss_fn="invalid_loss")

    with pytest.raises(ValueError, match="Unknown loss function"):
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))
        optim.step(x, y)
