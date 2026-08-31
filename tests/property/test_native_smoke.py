"""Smoke Tests for All 28 Native Models.

Verifies that each registered native model can:
1. Be instantiated
2. Run forward() without error
3. Run train_step() without error

These are minimal sanity checks - not full training tests.
"""

import pytest
import torch

from computronium.models.native.backprop_native import create_native_backprop_mlp
from computronium.models.native.diffusion_eqprop_native import (
    create_native_diffusion_eqprop,
)
from computronium.models.native.eqprop_native import create_native_eqprop_mlp
from computronium.models.native.fa_native import (
    create_native_fa_adaptive,
    create_native_fa_contrastive,
    create_native_fa_deep_dfa,
    create_native_fa_direct,
    create_native_fa_energy_guided,
    create_native_fa_energy_minimizing,
    create_native_fa_equilibrium_alignment,
    create_native_fa_layerwise_equilibrium,
    create_native_fa_mlp,
    create_native_fa_sign_symmetric,
    create_native_fa_stochastic,
)
from computronium.models.native.momentum_eqprop_native import (
    create_native_momentum_eqprop,
)
from computronium.models.native.pepita_native import create_native_pepita_mlp
from computronium.models.native.research_native import (
    create_native_directed_ep,
    create_native_finite_nudge_ep,
    create_native_holomorphic_ep,
)
from computronium.models.native.sparse_eqprop_native import create_native_sparse_eqprop
from computronium.models.native.ternary_eqprop_native import (
    create_native_ternary_eqprop,
)
from computronium.models.native.tile_native import (
    create_native_tile_ep,
    create_native_tile_fa,
    create_native_tile_gnn,
    create_native_tile_hebbian,
    create_native_tile_pc,
    create_native_tile_snn,
    create_native_tile_tp,
)

# Common test parameters
INPUT_DIM = 16
HIDDEN_DIM = 16
OUTPUT_DIM = 4
BATCH_SIZE = 4
DEVICE = "cpu"


def _make_batch(device: str = DEVICE):
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
    y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)
    return x, y


# =============================================================================
# Backprop Models
# =============================================================================


def test_native_backprop_mlp_smoke():
    """native_backprop_mlp: forward + train_step."""
    model = create_native_backprop_mlp(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, num_layers=1, lr=0.001
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


# =============================================================================
# EqProp Models
# =============================================================================


def test_native_eqprop_mlp_smoke():
    """native_eqprop_mlp: forward + train_step."""
    model = create_native_eqprop_mlp(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=0.5, settle_steps=10, lr=0.01
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


@pytest.mark.xfail(reason="DiffusionDynamics has gradient computation bug")
def test_native_diffusion_eqprop_smoke():
    """native_diffusion_eqprop: forward + train_step."""
    model = create_native_diffusion_eqprop(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=0.5, settle_steps=10, lr=0.01
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


def test_native_momentum_eqprop_smoke():
    """native_momentum_eqprop: forward + train_step."""
    model = create_native_momentum_eqprop(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=0.5, settle_steps=10, lr=0.01
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


def test_native_sparse_eqprop_smoke():
    """native_sparse_eqprop: forward + train_step."""
    model = create_native_sparse_eqprop(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=0.5, settle_steps=10, lr=0.01
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


def test_native_ternary_eqprop_smoke():
    """native_ternary_eqprop: forward + train_step."""
    model = create_native_ternary_eqprop(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=0.5, settle_steps=10, lr=0.01
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


# =============================================================================
# FA Models (12 variants)
# =============================================================================


@pytest.mark.parametrize(
    "factory",
    [
        create_native_fa_mlp,
        create_native_fa_adaptive,
        create_native_fa_stochastic,
        create_native_fa_contrastive,
        create_native_fa_sign_symmetric,
        create_native_fa_direct,
        create_native_fa_energy_guided,
        create_native_fa_energy_minimizing,
        create_native_fa_equilibrium_alignment,
        create_native_fa_layerwise_equilibrium,
        create_native_fa_deep_dfa,
    ],
)
def test_native_fa_variants_smoke(factory):
    """All native FA variants: forward + train_step."""
    model = factory(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, lr=0.001)
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


# =============================================================================
# PEPITA Model
# =============================================================================


def test_native_pepita_mlp_smoke():
    """native_pepita_mlp: forward + train_step."""
    model = create_native_pepita_mlp(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, num_layers=1, lr=0.01
    )
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


# =============================================================================
# Research EqProp Models
# =============================================================================


@pytest.mark.parametrize(
    "factory",
    [
        create_native_holomorphic_ep,
        create_native_directed_ep,
        create_native_finite_nudge_ep,
    ],
)
def test_native_research_eqprop_smoke(factory):
    """Research EqProp variants: forward + train_step."""
    model = factory(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, lr=0.001)
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


# =============================================================================
# Tile Models (7 variants)
# =============================================================================


_TILE_CRASH_XFAIL = {
    create_native_tile_ep: (
        "EnergyMinimizationDynamics on TileMesh raises TypeError: "
        "'Energy-based settling requires a layered geometry' — "
        "candidate invalid coordinate (R3.9)"
    ),
    create_native_tile_snn: (
        "SpikeIntegrationDynamics on TileMesh raises RuntimeError: "
        "tensor size mismatch (16 vs 212) at dim 1 — implementation bug (R3.4)"
    ),
    create_native_tile_gnn: (
        "EnergyMinimizationDynamics on TileMesh raises TypeError: "
        "'Energy-based settling requires a layered geometry' — "
        "candidate invalid coordinate (R3.9)"
    ),
}

_TILE_NO_LEARNING_XFAIL = {
    create_native_tile_fa: (
        "FA + InstantaneousDynamics on TileMesh yields no error signal: "
        "train_step leaves all params frozen"
    ),
    create_native_tile_tp: (
        "TargetInversionCredit + PredictiveSettlingDynamics on TileMesh "
        "yields no error signal: train_step leaves all params frozen"
    ),
    create_native_tile_hebbian: (
        "LocalGoodnessCredit on TileMesh returns empty pseudo-gradients: "
        "train_step leaves all params frozen"
    ),
    create_native_tile_pc: (
        "PredictiveSettlingDynamics on TileMesh yields no error signal: "
        "train_step leaves all params frozen"
    ),
}

_TILE_CRASH_FREE = (
    create_native_tile_fa,
    create_native_tile_tp,
    create_native_tile_hebbian,
    create_native_tile_pc,
)


@pytest.mark.parametrize(
    "factory",
    [
        *_TILE_NO_LEARNING_XFAIL,
        *(
            pytest.param(f, marks=pytest.mark.xfail(reason=r, strict=True))
            for f, r in _TILE_CRASH_XFAIL.items()
        ),
    ],
)
def test_native_tile_variants_smoke(factory):
    """Crash-free smoke for tile variants: forward + train_step run."""
    model = factory(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, lr=0.001)
    x, y = _make_batch()
    out = model(x)
    assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(f, marks=pytest.mark.xfail(reason=r, strict=True))
        for f, r in _TILE_NO_LEARNING_XFAIL.items()
    ],
)
def test_native_tile_learning_capability(factory):
    """Learning-capability lock: train_step must move parameters."""
    model = factory(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, lr=0.001)
    x, y = _make_batch()
    before = [p.detach().clone() for p in model.parameters()]
    model.train_step(x, y)
    assert any(
        not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
