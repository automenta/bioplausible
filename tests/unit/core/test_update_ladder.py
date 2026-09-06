"""TODO12 A1 — the ablation-ladder update primitives (UnitRMS, LocalAdam).

Locks: per-tensor unit-RMS normalization identity, ladder rungs distinct
from Muon (direction preserved) and from Adam (within-tensor structure
preserved), system-scoped state reuse fails loud, snapshot state
round-trips bitwise through the get_state/load_state protocol.
"""

import math
from typing import TYPE_CHECKING, cast

import pytest
import torch

from computronium.ontology.update import (
    AdamUpdate,
    LocalAdamUpdate,
    ParameterUpdateConfig,
    RiemannianOrthogonalUpdate,
    UnitRMSUpdate,
)

if TYPE_CHECKING:
    from computronium.ontology.geometry import Geometry

GEOM = cast("Geometry", None)
STEPS = 0.01


def _params() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "layers.0.weight": torch.randn(8, 6),
        "layers.1.weight": torch.randn(5, 8),
    }


def _grads(params) -> list[torch.Tensor]:
    return [torch.randn_like(p) * 10.0 for p in params.values()]


def _update_rms(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after - before).square().mean().sqrt())


def test_unit_rms_step_is_unit_rms_per_tensor():
    u = UnitRMSUpdate(ParameterUpdateConfig.unit_rms(step_size=STEPS, momentum=0.0))
    params = _params()
    out = u.step(params, _grads(params), GEOM)
    for (name, before), after in zip(params.items(), out.values()):
        rms = _update_rms(before, after)
        assert math.isclose(rms, STEPS, rel_tol=1e-3), (name, rms)


def test_unit_rms_direction_is_not_orthogonalized():
    """A rank-1 gradient keeps its rank-1 update direction (Muon would not)."""
    u = UnitRMSUpdate(ParameterUpdateConfig.unit_rms(step_size=STEPS, momentum=0.0))
    params = {"layers.0.weight": torch.zeros(6, 6)}
    grad = torch.randn(6, 1) @ torch.randn(1, 6)
    out = u.step(params, [grad], GEOM)
    delta = out["layers.0.weight"]
    assert torch.linalg.matrix_rank(delta) == 1


def test_local_adam_preserves_within_tensor_structure():
    """Scalar per-tensor denominator: the step direction matches the
    pseudo-gradient direction (Adam's per-coordinate denominator does not)."""
    local = LocalAdamUpdate(ParameterUpdateConfig.local_adam(step_size=STEPS))
    adam = AdamUpdate(ParameterUpdateConfig.adam(step_size=STEPS))
    params = _params()
    grads = _grads(params)
    local_out = local.step(params, grads, GEOM)
    adam_out = adam.step(params, grads, GEOM)
    for name, grad in zip(params, grads):
        g = grad.flatten()
        d_local = (params[name] - local_out[name]).flatten()
        d_adam = (params[name] - adam_out[name]).flatten()
        cos_local = float(torch.dot(d_local, g) / (d_local.norm() * g.norm()))
        cos_adam = float(torch.dot(d_adam, g) / (d_adam.norm() * g.norm()))
        assert cos_local > 0.999, (name, "local", cos_local)
        assert cos_adam < 0.999, (name, "adam", cos_adam)


def test_unit_rms_matches_magnitude_of_momentum_not_direction_of_muon():
    torch.manual_seed(1)
    u = UnitRMSUpdate(ParameterUpdateConfig.unit_rms(step_size=STEPS))
    m = RiemannianOrthogonalUpdate(
        ParameterUpdateConfig.riemannian_orthogonal(step_size=STEPS)
    )
    params = _params()
    grads = _grads(params)
    out_u = u.step(params, grads, GEOM)
    out_m = m.step(params, grads, GEOM)
    cosines = []
    for name in params:
        d_u = (out_u[name] - params[name]).flatten()
        d_m = (out_m[name] - params[name]).flatten()
        cos = torch.dot(d_u, d_m) / (d_u.norm() * d_m.norm())
        cosines.append(abs(float(cos)))
    assert all(c < 0.99 for c in cosines), cosines


@pytest.mark.parametrize(
    "cls", [UnitRMSUpdate, LocalAdamUpdate], ids=lambda c: c.__name__
)
def test_state_reuse_fails_loud(cls):
    cfg = (
        ParameterUpdateConfig.unit_rms()
        if cls is UnitRMSUpdate
        else ParameterUpdateConfig.local_adam()
    )
    u = cls(cfg)
    big = {"layers.0.weight": torch.zeros(8, 6)}
    small = {"layers.0.weight": torch.zeros(4, 4)}
    u.step(big, [torch.randn(8, 6)], GEOM)
    with pytest.raises(RuntimeError, match="system-scoped"):
        u.step(small, [torch.randn(4, 4)], GEOM)


@pytest.mark.parametrize(
    "cls", [UnitRMSUpdate, LocalAdamUpdate], ids=lambda c: c.__name__
)
def test_snapshot_state_roundtrip(cls):
    cfg = (
        ParameterUpdateConfig.unit_rms(step_size=STEPS)
        if cls is UnitRMSUpdate
        else ParameterUpdateConfig.local_adam(step_size=STEPS)
    )
    u = cls(cfg)
    params = _params()
    grads = _grads(params)
    u.step(params, grads, GEOM)
    state = u.get_state()
    fresh = cls(cfg)
    fresh.load_state(state)
    out_a = u.step(params, grads, GEOM)
    out_b = fresh.step(params, grads, GEOM)
    for name in params:
        assert torch.equal(out_a[name], out_b[name]), name
