"""TODO12 B1 — learned PEPITA feedback locks.

The learned inverse projections B are credit-internal state trained by a
transport-free reconstruction objective: they move from their fixed-random
init, strictly reduce the local reconstruction error, never read forward
weights (L3), round-trip bitwise through the get_state/load_state snapshot
protocol, and fail loud on shape-mismatched reuse.
"""

from typing import TYPE_CHECKING, cast

import pytest
import torch
import torch.nn.functional as fn

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    LocalGoodnessCredit,
    SubstrateConfig,
    SystemState,
)
from computronium.ontology.credit import Phase

if TYPE_CHECKING:
    from computronium.ontology.geometry import Geometry


def _fixture(learned: bool, seed: int = 0):
    torch.manual_seed(seed)
    geometry: Geometry = cast(
        "Geometry",
        FeedforwardGeometry(
            GeometryConfig.feedforward(input_dim=20, output_dim=4, hidden_dims=(16, 16))
        ),
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    x = torch.randn(8, 20)
    y = torch.randint(0, 4, (8,))
    acts = geometry.forward_with_intermediates(x, substrate)
    free = SystemState(x=x, y=y)
    free.activations = acts
    nudged = SystemState(x=x, y=y)
    nudged.activations = [
        *acts[:-1],
        acts[-1] + 0.5 * (fn.one_hot(y, 4).float() - acts[-1]),
    ]
    credit = LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01,
            local_objective="pepita",
            learned_feedback=learned,
            feedback_lr=0.5,
        )
    )
    return credit, geometry, {Phase.FREE: free, Phase.NUDGED: nudged}


def _recon_error(credit: LocalGoodnessCredit, key, acts, e1) -> float:
    b = credit._learned[key]
    post = acts
    c = (b / 0.01).T
    return float((post @ c - e1).square().mean())


def _e1(free_acts, y) -> torch.Tensor:
    out = free_acts[-1].detach()
    onehot = fn.one_hot(y, out.shape[-1]).to(out.dtype)
    return onehot - torch.softmax(out, dim=-1)


def test_learned_b_moves_and_reconstruction_improves():
    credit, geometry, states = _fixture(learned=True)
    credit.compute_pseudo_gradient(states, None, geometry)  # init step
    free_acts = states[Phase.FREE].activations
    assert isinstance(free_acts, list)
    y = states[Phase.FREE].y
    assert y is not None
    e1 = _e1(free_acts, y)
    keys = sorted(credit._learned)
    assert len(keys) == 3
    init = {k: credit._learned[k].clone() for k in keys}

    def _act_idx(key) -> int:
        return [n for n in geometry.params if n.endswith("weight")].index(key[0]) + 1

    err_before = [_recon_error(credit, k, free_acts[_act_idx(k)], e1) for k in keys]
    for _ in range(4):
        credit.compute_pseudo_gradient(states, None, geometry)
    for k in keys:
        assert not torch.equal(credit._learned[k], init[k]), "B must move"
        err_after = _recon_error(credit, k, free_acts[_act_idx(k)], e1)
        assert err_after < err_before[keys.index(k)], (k, err_before, err_after)


def test_transport_free_b_trajectory_ignores_forward_weights():
    """L3 lock: identical activation streams + different W values ⇒
    identical B trajectory (the legacy AdaptiveFA defect must not recur)."""
    c1, g1, states1 = _fixture(learned=True, seed=0)
    c2, g2, states2 = _fixture(learned=True, seed=0)
    with torch.no_grad():
        for name in g2.params:
            g2.params[name].mul_(3.7).add_(0.5)
    for _ in range(3):
        c1.compute_pseudo_gradient(states1, None, g1)
        c2.compute_pseudo_gradient(states2, None, g2)
    assert c1._learned.keys() == c2._learned.keys()
    for k in c1._learned:
        assert torch.equal(c1._learned[k], c2._learned[k]), k


def test_snapshot_state_roundtrip_bitwise():
    c1, g1, states1 = _fixture(learned=True, seed=0)
    for _ in range(3):
        c1.compute_pseudo_gradient(states1, None, g1)
    state = c1.get_state()
    assert "learned_feedback" in state and state["step"]["counter"].item() == 3
    c2, g2, states2 = _fixture(learned=True, seed=0)
    c2.load_state(state)
    out_a = c1.compute_pseudo_gradient(states1, None, g1)
    out_b = c2.compute_pseudo_gradient(states2, None, g2)
    for a, b in zip(out_a, out_b, strict=True):
        assert torch.equal(a, b)


def test_load_state_shape_mismatch_fails_loud():
    c1, g1, states1 = _fixture(learned=True, seed=0)
    c1.compute_pseudo_gradient(states1, None, g1)
    c1.compute_pseudo_gradient(states1, None, g1)
    state = c1.get_state()
    key = next(iter(state["learned_feedback"]))
    state["learned_feedback"][key] = torch.zeros(3, 3)
    _, _, _ = _fixture(learned=True, seed=0)
    c2 = LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            local_objective="pepita", learned_feedback=True
        )
    )
    c2.compute_pseudo_gradient(states1, None, g1)  # populate cache
    with pytest.raises(RuntimeError, match="system-scoped"):
        c2.load_state(state)


def test_fixed_pepita_unchanged_when_learned_off():
    credit, geometry, states = _fixture(learned=False)
    out = credit.compute_pseudo_gradient(states, None, geometry)
    assert credit.get_state() == {}
    assert all(g.abs().sum() > 0 for g in out)
