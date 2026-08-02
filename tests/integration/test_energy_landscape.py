"""Energy Landscape Visualization tests — Sprint 2.2.

Verifies the 2D energy-slice computation and PNG rendering for EqProp and
EquiTile families. Runs on CPU with tiny models; plotting is validated by
file existence, not pixel content, so the gate stays fast and headless-safe.
"""


import numpy as np
import pytest
import torch
from torch import nn

from bioplausible.analysis.energy_landscape import (
    compute_energy_landscape,
    plot_energy_landscape,
)


@pytest.fixture(scope="module")
def energy_task():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = torch.randint(0, 5, (64,))
    return x, y


def test_energy_landscape_finite(energy_task):
    """The energy grid is finite and the origin matches a direct evaluation."""
    x, y = energy_task
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 5))
    land = compute_energy_landscape(
        model, x, y, "mlp", "synthetic", radius=0.5, grid=9
    )
    assert land.energy.shape == (9, 9)
    assert np.isfinite(land.energy).all()
    mid = land.energy.shape[0] // 2
    direct = float(
        nn.functional.cross_entropy(model(x[:64]), y[:64]).item()
    )
    assert abs(land.energy[mid, mid] - direct) < 1e-4, (
        "origin energy should equal a direct forward evaluation"
    )
    assert land.param_count == sum(p.numel() for p in model.parameters())
    assert land.d1_norm > 0


def test_energy_landscape_eqprop(energy_task):
    """LoopedMLP (equilibrium propagation) produces a finite landscape."""
    from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

    x, y = energy_task
    model = LoopedMLP(
        16, 32, 5, max_steps=10, gradient_method="contrastive", backend="pytorch"
    )
    for _ in range(3):
        model.train_step(x, y)
    land = compute_energy_landscape(
        model, x, y, "eqprop_mlp", "synthetic", radius=0.5, grid=7
    )
    assert np.isfinite(land.energy).all()
    mid = land.energy.shape[0] // 2
    direct = float(
        nn.functional.cross_entropy(model(x[:64]), y[:64]).item()
    )
    assert abs(land.energy[mid, mid] - direct) < 1e-4


def test_energy_landscape_equitile(energy_task):
    """EquiTile produces a finite landscape (fallback cross-entropy proxy)."""
    from bioplausible.equitile import EquiTile

    x, y = energy_task
    model = EquiTile(input_dim=16, hidden_dim=32, output_dim=5, num_layers=2)
    land = compute_energy_landscape(
        model, x, y, "equitile", "synthetic", radius=0.5, grid=7
    )
    assert np.isfinite(land.energy).all()


def test_energy_landscape_plot_png(energy_task, tmp_path):
    """Plotting writes a valid PNG file."""
    x, y = energy_task
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 5))
    land = compute_energy_landscape(
        model, x, y, "mlp", "synthetic", radius=0.5, grid=9
    )
    out = plot_energy_landscape(land, tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_energy_landscape_npz_roundtrip(energy_task, tmp_path):
    """The .npz archive round-trips for downstream analysis."""
    x, y = energy_task
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 5))
    land = compute_energy_landscape(
        model, x, y, "mlp", "synthetic", radius=0.5, grid=5
    )
    npz = land.save(tmp_path / "land.npz")
    data = np.load(npz)
    assert data["energy"].shape == (5, 5)
    assert np.allclose(data["energy"], land.energy)
