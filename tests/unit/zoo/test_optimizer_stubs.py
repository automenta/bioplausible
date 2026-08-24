"""Tests for the EWC and SpectralConstraint optimizer implementations."""

import torch
from torch import nn


def test_ewc_optimizer_step():
    """EWC optimizer can step without error."""
    model = nn.Linear(10, 5)
    from computronium.zoo.optimizers.ewc import EWC

    optim = EWC(model.parameters(), lr=0.01, ewc_lambda=0.1)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optim.step()  # Should not raise


def test_ewc_optimizer_zero_grad():
    """EWC optimizer can zero gradients."""
    model = nn.Linear(10, 5)
    from computronium.zoo.optimizers.ewc import EWC

    optim = EWC(model.parameters(), lr=0.01)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    assert any(
        p.grad.norm().item() > 0 for p in model.parameters() if p.grad is not None
    )
    optim.zero_grad()
    assert all(
        p.grad is None or p.grad.norm().item() == 0.0 for p in model.parameters()
    )


def test_ewc_optimizer_update_fisher():
    """EWC optimizer can compute Fisher information."""
    model = nn.Linear(10, 5)
    from computronium.zoo.optimizers.ewc import EWC

    optim = EWC(model.parameters(), lr=0.01)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    loader = [(x, y)]

    optim.update_fisher(model, loader, task_id=0)
    assert 0 in optim._fisher
    pid = id(next(model.parameters()))
    assert pid in optim._fisher[0]


def test_spectral_constraint_registered():
    """SpectralConstraint is registered under the PARAM_UPDATE category."""
    from computronium.core.registry import ComponentCategory, Registry

    cls = Registry.get(ComponentCategory.PARAM_UPDATE, "spectral")
    assert cls is not None
    assert cls.__name__ == "SpectralConstraint"


def test_spectral_constraint_step():
    """SpectralConstraint projects weights to have bounded spectral norm."""
    model = nn.Linear(10, 5, bias=False)
    from computronium.zoo.optimizers.spectral import SpectralConstraint

    optim = SpectralConstraint(model.parameters(), lr=0.01, max_norm=0.5)

    # Stepping with generated gradients
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.randn_like(p)
    optim.step()  # Should not raise

    # Verify spectral norm is bounded
    w = model.weight.data
    _, s, _ = torch.linalg.svd(w.reshape(w.shape[0], -1), full_matrices=False)
    assert s.max() <= 0.5 + 1e-5, "Spectral norm should be <= max_norm"
