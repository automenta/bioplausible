"""Causal-transformer geometry locks (R11: LM capability pull).

The TransformerGeometry is the G-axis realization of a language model:
input token ids [B, T], logits [B*T, V], transition-aligned acts so the
local-credit family (ff goodness, pepita inverse projections) works on
attention weights UNMODIFIED.
"""

import dataclasses
from typing import Literal

import pytest
import torch

from computronium import (
    AdamUpdate,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.core.system_trainer.factory import (
    _geometry_spec_parts,
    _restore_geometry_params,
)
from computronium.ontology.credit import _learnable_weight_names
from computronium.ontology.geometry import GeometryConfig, geometry_from_config

V, T, D, L, H = 65, 16, 64, 2, 4


def _geometry():
    return geometry_from_config(
        GeometryConfig.causal_transformer(
            vocab_size=V, d_model=D, n_layers=L, n_heads=H, seq_len=T
        )
    )


def test_acts_aligned_to_weight_transitions():
    g = _geometry()
    subs = DigitalSubstrate()
    ids = torch.randint(0, V, (2, T))
    acts = g.forward_with_intermediates(ids, subs)
    wn = _learnable_weight_names(g.params)
    assert len(acts) == len(wn) + 1
    for k, name in enumerate(wn):
        w = g.params[name]
        assert w.ndim == 2
        assert acts[k].shape[-1] == w.shape[1], f"act {k} vs {name}"


def test_causal_masking():
    g = _geometry()
    ids = torch.randint(0, V, (1, T))
    mid = T // 2
    l1 = g.forward(ids, DigitalSubstrate())
    ids2 = ids.clone()
    ids2[0, mid:] = (ids2[0, mid:] + 7) % V
    l2 = g.forward(ids2, DigitalSubstrate())
    assert torch.equal(l1[0, :mid], l2[0, :mid])


def test_spec_round_trip_bitwise():
    g = _geometry()
    serialized = dataclasses.asdict(g.config)
    cfg2, _ = _geometry_spec_parts(dict(serialized))
    g2 = geometry_from_config(cfg2)
    _restore_geometry_params(g2, {k: v.tolist() for k, v in g.params.items()})
    ids = torch.randint(0, V, (2, T))
    assert torch.equal(g.forward(ids, DigitalSubstrate()), g2.forward(ids, None))


def _run(credit: Literal["ff", "pepita", "bp"], steps: int = 120) -> list[float]:
    torch.manual_seed(0)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=_geometry(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=(
            BackpropCredit()
            if credit == "bp"
            else LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness(
                    local_objective=credit, feedback_scale=0.01
                )
            )
        ),
        update=AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
    )
    gen = torch.Generator().manual_seed(3)
    data = torch.arange(4096) % V
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(data) - T - 1, (8,), generator=gen)
        win = data[idx.unsqueeze(1) + torch.arange(T + 1)]
        x, y = win[:, :-1], win[:, 1:].reshape(-1)
        m = run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )
        losses.append(m["loss"])
    return losses


def test_bp_learns_structured_tokens():
    losses = _run("bp")
    import math

    assert losses[-1] < math.log(V) - 0.5, losses[-1]


@pytest.mark.parametrize("credit", ["ff", "pepita"])
def test_local_credit_reaches_transformer_weights(credit):
    """Mechanism lock: the local family produces NONZERO pseudo-gradients
    on attention weights (the flat-Linear positional contract holds)."""
    torch.manual_seed(0)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=_geometry(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                local_objective=credit, feedback_scale=0.01
            )
        ),
        update=AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
    )
    gen = torch.Generator().manual_seed(3)
    data = (torch.arange(4096) * 7 + 3) % V  # structured, learnable
    idx = torch.randint(0, len(data) - T - 1, (8,), generator=gen)
    win = data[idx.unsqueeze(1) + torch.arange(T + 1)]
    x, y = win[:, :-1], win[:, 1:].reshape(-1)
    from computronium.ontology.credit import Phase

    free = system.dynamics.settle(
        SystemState(x=x, y=y), system.geometry, system.substrate
    )
    nudged = system.dynamics.settle(
        SystemState(x=x, y=y), system.geometry, system.substrate, target=y
    )
    grads = system.credit.compute_pseudo_gradient(
        {Phase.FREE: free, Phase.NUDGED: nudged}, None, system.geometry
    )
    wn = _learnable_weight_names(system.geometry.params)
    norms = {n: float(g.norm()) for n, g in zip(wn, grads, strict=True)}
    nonzero = {n for n, v in norms.items() if v > 0}
    assert "blocks.0.in_proj.weight" in nonzero, norms
    assert len(nonzero) >= len(wn) // 2, norms
