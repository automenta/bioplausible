"""F5 — Resource-vector accounting: the physical advantage, measured honestly.

Implements the pre-registered schema (`scripts/probes/f5_resource_vector.py`
docstring, written before any measurement) at demo scale: per-arm,
per-depth MEASURED saved-for-backward bytes (peak == total here) via the
D4 instrument, and MEASURED FLOPs per train step via torch.profiler —
no schema-derived proxies in the record.

VERDICT (pinned 2026-09-06, depths 4/16, width 16, batch 16, CPU — the
pre-registered ~10x memory / ~5x energy targets are FALSIFIED at HEAD
for the D17-winner realization; reported, not hidden):

- Memory MISS: ff_hybrid (LocalGoodnessCredit, autograd-realized) SAVES
  MORE than backprop (177.5 vs 143.5 KiB at depth 16) — its per-layer
  local losses run through autograd, which stores exactly what a
  backward stores, all layers alive simultaneously (peak == total).
  The O(1)-memory CLASS is real: thermo (requires_autograd=False)
  saves EXACTLY 0 bytes at every depth — the algorithmic claim is
  realizable, the ff_hybrid realization does not realize it
  (standing caution: an implementation artifact, not an in-principle
  failure; a non-autograd local-goodness realization would store 0).
- Energy MISS: ff_hybrid costs ~1.3x bp/adam FLOPs (per-layer local
  backward + optimizer); the OrthoAdam-family premium is real but
  modest at this scale (muon 1.349 M vs euclid 1.218 M — "the
  optimizer is the cost"). thermo/euclid ties bp/adam (0.943 vs
  0.939 M) — no energy win at HEAD either.
- Claim A (credit locality — "forward-local credit with a single
  readout supervision term — no backward sweep through the hidden
  layers") is an ALGORITHMIC statement and stands; Claim B (physical
  advantage) is NOT demonstrated at HEAD — the record pins the miss.
"""

import json

import torch
from torch.profiler import ProfilerActivity, profile

from computronium import (
    CreditAssignmentConfig,
    GeometryConfig,
    ParameterUpdateConfig,
    PlasticityConfig,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.core.profiling import measure_saved_activation_bytes
from computronium.core.system_trainer.joint import (
    compose_joint_system_from_configs,
)
from computronium.visualization import bars_panel, figure_spec

DEPTHS = (4, 16)
WIDTH = 16
BATCH = 16
INPUT_DIM = OUTPUT_DIM = 8

ARMS: dict[str, tuple] = {
    "bp_adam": (
        CreditAssignmentConfig.gradient(),
        ParameterUpdateConfig.adam(step_size=0.01),
    ),
    "ff_hybrid_muon": (
        CreditAssignmentConfig.local_goodness(local_objective="ff", readout_error=True),
        ParameterUpdateConfig.riemannian_orthogonal(step_size=0.01),
    ),
    "ff_hybrid_euclid": (
        CreditAssignmentConfig.local_goodness(local_objective="ff", readout_error=True),
        ParameterUpdateConfig.euclidean(step_size=0.01),
    ),
    "thermo_euclid": (
        CreditAssignmentConfig.thermodynamic_contrast(),
        ParameterUpdateConfig.euclidean(step_size=0.01),
    ),
}


def _system(arm: str, depth: int):
    credit_cfg, update_cfg = ARMS[arm]
    return compose_joint_system_from_configs(
        SubstrateConfig.digital(),
        GeometryConfig.feedforward(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dims=(WIDTH,) * depth,
        ),
        StateDynamicsConfig.instantaneous(),
        PlasticityConfig.null(),
        credit_cfg,
        update_cfg,
        device="cpu",
    )


def _measure(arm: str, depth: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    torch.manual_seed(0)
    joint = _system(arm, depth)
    joint.train_step(x, y)  # warmup (lazy init, optimizer state)
    _, saved = measure_saved_activation_bytes(joint.train_step, x, y)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        joint.train_step(x, y)
    flops = sum(e.flops for e in prof.key_averages())
    state = joint.update.get_state() if hasattr(joint.update, "get_state") else {}
    opt_bytes = sum(
        v.numel() * v.element_size() for group in state.values() for v in group.values()
    )
    return {
        "saved_bytes": round(saved.total_bytes, 1),
        "flops": int(flops),
        "optimizer_state_bytes": opt_bytes,
    }


def test_demo_resource_vector(emit_run_record) -> None:
    torch.manual_seed(0)
    x = torch.randn(BATCH, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (BATCH,))

    cells: dict[str, dict] = {}
    for arm in ARMS:
        for depth in DEPTHS:
            cells[f"{arm}@d{depth}"] = _measure(arm, depth, x, y)

    record: dict = {
        "depths": list(DEPTHS),
        "width": WIDTH,
        "batch": BATCH,
        "cells": cells,
        "schema": "scripts/probes/f5_resource_vector.py (pre-registered)",
    }

    d16 = {arm: cells[f"{arm}@d16"] for arm in ARMS}

    # The O(1)-memory class is real: thermo stores nothing, at any depth.
    for depth in DEPTHS:
        assert cells[f"thermo_euclid@d{depth}"]["saved_bytes"] <= 0.0, (
            "thermo must save exactly nothing"
        )
    assert d16["bp_adam"]["saved_bytes"] > cells["bp_adam@d4"]["saved_bytes"] > 0.0, (
        "backprop's stored-activation sweep must be depth-scaled"
    )

    # The honest miss, pinned: ff_hybrid's autograd realization saves at
    # least as much as backprop — the ~10x memory target is falsified at
    # HEAD. If a future non-autograd ff_hybrid lands, this assert is the
    # ratchet it must flip.
    assert d16["ff_hybrid_muon"]["saved_bytes"] >= d16["bp_adam"]["saved_bytes"], (
        "ff_hybrid must still save >= backprop (the pinned miss)"
    )

    # The energy miss, pinned: local-autograd costs more FLOPs than bp,
    # and the orthogonalizing optimizer is a real premium on top.
    assert d16["ff_hybrid_muon"]["flops"] > d16["bp_adam"]["flops"]
    assert d16["ff_hybrid_muon"]["flops"] > d16["ff_hybrid_euclid"]["flops"], (
        "the ortho optimizer must cost more than the plain update"
    )

    record["verdict"] = {
        "memory_target_10x": "MISS at HEAD — ff_hybrid autograd realization "
        "saves more than backprop; O(1) class exists (thermo = 0 exactly)",
        "energy_target_5x": "MISS at HEAD — ff_hybrid ~1.3x bp FLOPs; the "
        "optimizer is the cost (muon > euclid); thermo ties bp",
    }

    arms_sorted = sorted(ARMS)
    record["figure"] = figure_spec(
        "F5 — resource-vector accounting: measured, at HEAD",
        bars_panel(
            {
                arm: {
                    "saved KiB @d16": d16[arm]["saved_bytes"] / 1024,
                    "MFLOPs @d16": d16[arm]["flops"] / 1e6,
                }
                for arm in arms_sorted
            },
            ylabel="measured cost (lower is better)",
            title="local-realization vs backprop (miss pinned; thermo = 0)",
        ),
    )

    print(json.dumps(record["verdict"], indent=1))
    emit_run_record("F5", "resource_vector", record)
