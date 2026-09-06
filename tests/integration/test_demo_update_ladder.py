"""D18 — The optimizer crutch, killed or mapped: the ablation-ladder verdict.

TODO12 Workstream A (A0–A2). The unifying hypothesis — *the credit
direction is approximately right; only the magnitude is broken* —
measured on the P4 width-fragile LM cells (the recorded explosion
regime: ePC w32/w64 under Muon at its registered lr; PEPITA w32
everywhere), with per-tensor magnitude normalization only:

1. **ePC is width-robust without Muon under UnitRMS (seeds 0-2):**
   momentum-EMA normalized to unit RMS per tensor (NO
   orthogonalization) trains ePC at the fragile widths —
   val_ppl w64: 28.1/28.4/28.5 across seeds (chance 65), w32 ~35 —
   where Muon at its registered lr explodes (P4: act std -> 2028).
   The magnitude-only rung REPLACES the orthogonalizer for ePC in
   this band: the crutch is dead for ePC at w32-64.
2. **PEPITA's runaway is structural (audit-backed):** under UnitRMS
   the fixed-B DFA realization explodes at every lr; the defect audit
   ruled out feedback_scale tuning (inert under per-tensor
   normalization — it scales the pseudo-gradient linearly), the
   constant one-hot term (centered-e1 explodes identically), and
   init-time activity (step-0 stds healthy). Probe
   `scripts/probes/p4_width_fragility.py` docstring carries the full
   audit record. PEPITA-in-principle is NOT condemned — the honest
   repairs are B1 (learned B changes the update's row space) or A5
   (settle-path gain homeostasis).

Fixed-step arms (600 steps — deterministic, unlike walltime budgets;
the gallery lock requires byte-stable records), LM tiny-Shakespeare
char task, val ppl over a fixed val window set, seeds 0-2, CPU.
"""

import math
import time

import pytest
import torch
from torch import nn

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.data.lm import get_lm_dataset
from computronium.ontology.update import RiemannianOrthogonalUpdate, UnitRMSUpdate
from computronium.visualization._demo_api import bars_panel, figure_spec

STEPS = 600
CTX = 32
BATCH = 32
DEPTH = 4
VOCAB = 65
VAL_WINDOWS = 1024
SEEDS = (0, 1, 2)
CHANCE = 65.0
DEVICE = "cpu"

# credit, update, width, lr — the fragile-cell grid of the A2 probe.
ARMS = {
    "epc_w32_unit_rms": ("epc_thermo", "unit_rms", 32, 3e-4),
    "epc_w64_unit_rms": ("epc_thermo", "unit_rms", 64, 3e-4),
    "epc_w32_muon": ("epc_thermo", "muon", 32, 0.01),
    "epc_w64_muon": ("epc_thermo", "muon", 64, 0.01),
    "pepita_w32_unit_rms": ("pepita", "unit_rms", 32, 3e-4),
}


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    train_ds = get_lm_dataset("tiny_shakespeare", seq_len=64, split="train")
    val_ds = get_lm_dataset("tiny_shakespeare", seq_len=64, split="validation")
    stoi = {c: i for i, c in enumerate(sorted(set(train_ds.idx_to_char.values())))}
    val_raw = val_ds.decode(val_ds.data)
    return train_ds.data.long(), torch.tensor([stoi[c] for c in val_raw])


def _val_windows(val_t: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(0)
    vidx = torch.randint(0, len(val_t) - CTX - 1, (VAL_WINDOWS,), generator=gen)
    offs = torch.arange(CTX + 1)
    vwin = val_t[vidx.unsqueeze(1) + offs]
    eye = torch.eye(VOCAB)
    return [(eye[w[:, :-1]].reshape(w.size(0), -1), w[:, -1]) for w in vwin.split(256)]


def _system(credit: str, update: str, width: int, lr: float):
    if credit == "epc_thermo":
        credit_obj = ThermodynamicContrast()
        dynamics = ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(max_steps=10, step_size=0.1)
        )
    else:
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="pepita"
            )
        )
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=CTX * VOCAB, output_dim=VOCAB, hidden_dims=(width,) * DEPTH
        )
    )
    if update == "unit_rms":
        update_obj = UnitRMSUpdate(
            ParameterUpdateConfig.unit_rms(step_size=lr, momentum=0.9)
        )
    else:
        update_obj = RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr, momentum=0.9)
        )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=DEVICE)),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit_obj,
        update=update_obj,
    )


def _run_arm(arm, seed: int, train_t, val_windows) -> float:
    credit, update, width, lr = arm
    torch.manual_seed(seed)
    system = _system(credit, update, width, lr)
    if hasattr(system.geometry, "to"):
        system.geometry.to(DEVICE)  # type: ignore[attr-defined]
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(STEPS):
        idx = torch.randint(0, len(train_t) - CTX - 1, (BATCH,), generator=gen)
        win = train_t[idx.unsqueeze(1) + torch.arange(CTX + 1)]
        x = (
            torch.nn.functional
            .one_hot(win[:, :-1], VOCAB)
            .float()
            .reshape(BATCH, CTX * VOCAB)
            .to(DEVICE)
        )
        y = win[:, -1].to(DEVICE)
        run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )
    tot = n = 0
    with torch.no_grad():
        for x, y in val_windows:
            state = system.dynamics.settle(
                SystemState(x=x),  # type: ignore[arg-type]
                system.geometry,
                system.substrate,
                None,
            )
            acts = state.activations
            logits = acts[-1] if isinstance(acts, list) else acts
            assert y is not None
            tot += nn.functional.cross_entropy(logits, y, reduction="sum").item()  # type: ignore[arg-type]
            n += y.numel()
    avg = tot / n
    return float(round(math.exp(min(avg, 20)), 2))


@pytest.mark.timeout(1200)
def test_demo_update_ladder(emit_run_record) -> None:
    t0 = time.time()
    train_t, val_t = _tokens()
    val_windows = _val_windows(val_t)

    record: dict = {"arms": {}, "seeds": list(SEEDS), "steps": STEPS}
    for name, arm in ARMS.items():
        ppls = [_run_arm(arm, seed, train_t, val_windows) for seed in SEEDS]
        mean = sum(ppls) / len(ppls)
        record["arms"][name] = {
            "mean": mean,
            "std": (sum((p - mean) ** 2 for p in ppls) / len(ppls)) ** 0.5,
            "seeds": ppls,
        }
        print(f"{name:>22}: {mean:>14.2f}  {ppls}", flush=True)

    arms = record["arms"]
    record["figure"] = figure_spec(
        "D18 — the ablation ladder: UnitRMS (magnitude-only) makes ePC "
        "width-robust without Muon; PEPITA's runaway is structural",
        bars_panel(
            groups={
                "val_ppl (lower is better)": {k: v["mean"] for k, v in arms.items()}
            },
            chance=CHANCE,
            chance_label="chance (65)",
            ylabel="val ppl",
            yerr={k: {"val_ppl (lower is better)": v["std"]} for k, v in arms.items()},
        ),
        figsize=[9.0, 4.5],
    )
    print(f"walltime: {round(time.time() - t0, 1)} s (printed, never recorded)")
    emit_run_record("D18", "update_ladder", record)

    # Headline: UnitRMS trains ePC at both fragile widths, multi-seed.
    for name in ("epc_w32_unit_rms", "epc_w64_unit_rms"):
        assert arms[name]["mean"] < 45.0, (
            f"{name}: UnitRMS must train ePC without Muon (mean "
            f"{arms[name]['mean']:.2f}, chance {CHANCE})"
        )
        assert max(arms[name]["seeds"]) < 60.0, (
            f"{name}: every seed must train (seeds {arms[name]['seeds']})"
        )
    # The crutch comparison: Muon at its registered lr explodes at w32
    # (P4's recorded failure mode) — UnitRMS strictly dominates there.
    assert arms["epc_w32_muon"]["mean"] > arms["epc_w32_unit_rms"]["mean"] * 2, (
        f"Muon w32 ({arms['epc_w32_muon']['mean']:.2f}) must clearly trail "
        f"UnitRMS ({arms['epc_w32_unit_rms']['mean']:.2f})"
    )
    # PEPITA control: the audit-backed structural runaway persists.
    assert arms["pepita_w32_unit_rms"]["mean"] > 100.0, (
        f"PEPITA w32 control must explode at HEAD (mean "
        f"{arms['pepita_w32_unit_rms']['mean']:.2f}) — if it trains, the "
        "A2 audit verdict is stale and must be re-run"
    )
