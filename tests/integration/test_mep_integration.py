"""MEP Preset Integration Tests (ontology-native replacement).

The former tests checked preset availability through the Registry; the same
integration surface is now the presets themselves — each wraps a model's
parameters into a working optimizer that decreases loss on a tiny task.
"""

import pytest
import torch
from torch import nn

from computronium.mep.presets import (
    local_ep,
    muon_backprop,
    natural_ep,
    sdmep,
    smep,
    smep_fast,
)

DIM_IN = 8
DIM_HIDDEN = 8
DIM_OUT = 4
N = 32


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(DIM_IN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_OUT),
        ])
        self.net = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def transition_modules(self) -> list[nn.Module]:
        """EP-capable surface: linear layers the settle strategies traverse."""
        return [m for m in self.layers if isinstance(m, nn.Linear)]


@pytest.fixture(scope="module")
def dataset() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    x = torch.randn(N, DIM_IN)
    y = torch.randint(0, DIM_OUT, (N,))
    return x, y


@pytest.mark.parametrize(
    ("preset", "kwargs"),
    [
        (smep, {}),
        (smep_fast, {}),
        (sdmep, {}),
        (natural_ep, {}),
        (local_ep, {}),
        (muon_backprop, {}),
    ],
)
def test_preset_wraps_params_into_optimizer(preset, kwargs, dataset) -> None:
    model = TinyMLP()
    optimizer = preset(model.parameters(), model=model, lr=0.01, **kwargs)
    x, y = dataset
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step(x=x, target=y)
    moved = any(
        not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True)
    )
    assert moved, f"{preset.__name__} must update parameters"


@pytest.mark.parametrize(
    ("preset", "kwargs"),
    [
        (smep, {}),
        (smep_fast, {}),
        (sdmep, {}),
        (natural_ep, {}),
        (local_ep, {}),
        (muon_backprop, {}),
    ],
)
def test_preset_training_reduces_loss(preset, kwargs, dataset) -> None:
    torch.manual_seed(0)
    model = TinyMLP()
    optimizer = preset(model.parameters(), model=model, lr=0.01, **kwargs)
    x, y = dataset
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step(x=x, target=y)
        losses.append(loss.item())
    assert losses[-1] < losses[0], (
        f"{preset.__name__} must reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
