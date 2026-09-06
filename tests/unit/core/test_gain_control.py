"""TODO12 A5 — settle-path gain homeostasis locks.

The gain_control primitive renormalizes hidden-layer activations at settle
emit: unit_rms is the μPC recipe (per-sample unit RMS), spectral rescales
each hidden layer's batch matrix to unit spectral norm. Input and output
layers pass through untouched; zeros and non-finite layers are never
fabricated into signal.
"""

import pytest
import torch

from computronium import (
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.ontology.dynamics._dynamics import _apply_gain_control
from computronium.ontology.system import SystemState


def _acts() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [
        torch.randn(8, 20),
        torch.randn(8, 16) * 50.0,
        torch.randn(8, 16) * 50.0,
        torch.randn(8, 4),
    ]


def test_none_is_passthrough():
    acts = _acts()
    out = _apply_gain_control(acts, "none")
    for a, b in zip(acts, out):
        assert torch.equal(a, b)


def test_unit_rms_per_sample_on_hidden_only():
    acts = _acts()
    out = _apply_gain_control(acts, "unit_rms")
    assert torch.equal(out[0], acts[0]), "input layer must pass through"
    assert torch.equal(out[-1], acts[-1]), "output layer must pass through"
    for i in (1, 2):
        rms = out[i].square().mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-3)


def test_spectral_unit_on_hidden_only():
    acts = _acts()
    out = _apply_gain_control(acts, "spectral")
    assert torch.equal(out[0], acts[0])
    assert torch.equal(out[-1], acts[-1])
    for i in (1, 2):
        sigma = torch.linalg.matrix_norm(out[i], ord=2)
        assert torch.isclose(sigma, torch.tensor(1.0), rtol=1e-3)


def test_zeros_and_nonfinite_pass_through():
    acts = [torch.randn(4, 8), torch.zeros(4, 8), torch.full((4, 8), float("inf"))]
    out = _apply_gain_control(acts, "unit_rms")
    assert torch.equal(out[1], acts[1])
    assert torch.equal(out[2], acts[2])
    out_s = _apply_gain_control(acts, "spectral")
    assert torch.isinf(out_s[2]).all()


def test_short_acts_untouched():
    acts = [torch.randn(4, 8), torch.randn(4, 8) * 99.0]
    assert torch.equal(_apply_gain_control(acts, "unit_rms")[1], acts[1])


@pytest.mark.parametrize("mode", ["unit_rms", "spectral"])
def test_instantaneous_settle_bounded(mode):
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=20, output_dim=4, hidden_dims=(16, 16))
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    dynamics = InstantaneousDynamics(
        StateDynamicsConfig.instantaneous(gain_control=mode)  # type: ignore[arg-type]
    )
    state = SystemState(x=torch.randn(8, 20))
    x = state.x
    assert x is not None
    settled = dynamics.settle(state, geometry, substrate, None)  # type: ignore[arg-type]
    acts = settled.activations
    assert isinstance(acts, list)
    stds = [float(a.std()) for a in acts]
    assert max(stds[1:-1]) < 5.0, stds
    assert torch.equal(acts[0], x)
    assert torch.equal(acts[-1], geometry.forward_with_intermediates(x, substrate)[-1])
