"""F4 — The credit-channel failure map (all eight faces, one figure).

The consolidated finding figure of the TODO12 repair program: the eight
measured ways the credit signal fails, each with its landed lever or an
honest OPEN cell. Two mechanisms are demonstrated LIVE at demo scale
(the attenuating channel with the A4 repair, and the blocked channel);
the rest are ratchet-locked against the pinned run records — F4 fails if
any pinned mechanism regresses (D18, D16, F1, F2, D14).

The map (failure mode → repair → status):

1. **Misaligned channel** (PEPITA fixed-B) — OPEN: diverges under every
   landed lever (ratchet: D18 pepita_w32_unit_rms at the divergence
   sentinel; five causes ruled out in TODO12's audit chain —
   feedback_scale, centered-e1, row space, hidden gain, output step
   shape). The faithful-forward-modulation realization is untested.
2. **Attenuating channel** (ePC ~4×/layer decay) — REPAIRED (A4):
   LIVE — credit_norm="spectral" flattens per-layer credit norms
   (~1.0, asserted) and lifts depth-8 learning over the unnormalized
   arm at matched lr (asserted).
3. **Unnormalized gain** (width fragility) — REPAIRED (A1): ratchet —
   D18 ePC trains at both fragile widths under unit_rms while Muon at
   its registered lr explodes (w64: 32.5 vs 101.2; w32: 42.5 vs 191.7).
4. **Disconnected channel** (pure FF error-blindness) — REPAIRED
   (readout_error, landed pre-TODO12): carried by D13's ff_hybrid row.
5. **Blocked channel** (sPC hidden credit exactly 0) — REPAIRED (ePC
   reparameterization): LIVE — sPC hidden norms asserted exactly 0.0
   while ePC's are > 0 (F1's audit, re-shown here).
6. **Train/inference objective gap** (P2 frozen-error) — OPEN: C1
   pending (contrastive path works; epc_thermo×Muon trains LM; the
   untried cells remain).
7. **Absent channel** (timing-STDP has no task term) — OPEN: F2
   ratchet — supervised_train_acc 0.048 ≈ chance; B5 (reward-modulated
   STDP) pending.
8. **Low-rank credit** (the optimizer crutch) — REPAIRED for ePC-width,
   mapped otherwise (A6): ratchets — D18 crutch-dead cells; D16
   unit_rms vision-quick boundary (regime-shaped rung); D14 faithful
   regime self-sufficient (mupc_beta10 0.828 at depth 20, no
   credit-side lever).

Status codes in the figure: 2 = repair demonstrated live here, 1 =
repair demonstrated in a pinned record, 0 = honest OPEN cell.
"""

import json
from itertools import islice
from pathlib import Path

import pytest
import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology.credit import Phase
from computronium.visualization import figure_spec, heatmap_panel

WIDTH = 32
DEPTH = 8
BATCH_CAP = 60
SETTLE_STEPS = 15
LR = 0.2
BETA = 0.5
CHANCE = 0.1

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = REPO_ROOT / "docs" / "figures" / "run_records"

# status: 2 = live repair, 1 = pinned repair, 0 = open
_ROWS: tuple[tuple[str, int], ...] = (
    ("misaligned (pepita fixed-B)", 0),
    ("attenuating (epc decay)", 2),
    ("unnormalized gain (width)", 1),
    ("disconnected (ff blindness)", 1),
    ("blocked (spc trapping)", 2),
    ("objective gap (p2 frozen-e)", 0),
    ("absent (stdp no task term)", 0),
    ("low-rank (optimizer crutch)", 1),
)


def _load(capability: str) -> dict:
    return json.loads((RECORDS_DIR / f"{capability}.json").read_text(encoding="utf-8"))[
        "data"
    ]


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _epc_system(credit_norm: str, substrate):
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * DEPTH,
        )
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(
            beta=BETA,
            credit_norm=credit_norm,  # type: ignore[arg-type]
        )
    )
    return compose_system(
        substrate=substrate,
        geometry=geometry,
        # the A4/F1-probe ePC regime: short aggressive settle (NOT the
        # F1-demo PredictiveSettling arm, which explodes under spectral)
        dynamics=ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(
                max_steps=5, step_size=0.5, beta=BETA
            )
        ),
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )


def _spc_system(substrate):
    """F1's sPC arm: the contrastive rule under the layered settle."""
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
        )
    )
    return compose_system(
        substrate=substrate,
        geometry=geometry,
        dynamics=PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=SETTLE_STEPS)
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )


def _train_acc(system, config, train_data) -> float:
    return SystemTrainer(system=system, config=config, train_data=train_data).fit()[-1][
        "train_acc"
    ]


def _credit_norms(system, substrate, batch) -> list[float]:
    x, y = batch
    free = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=None
    )
    nudged = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=y
    )
    grads = system.credit.compute_pseudo_gradient(
        {Phase.FREE: free, Phase.NUDGED: nudged}, None, system.geometry
    )
    return [g.norm().item() for g in grads]


def _run_live_arms(substrate, config, train_data) -> dict:
    """Live cells: attenuating channel + A4 repair; blocked channel."""
    epc_none = _epc_system("none", substrate)
    epc_spectral = _epc_system("spectral", substrate)
    acc_none = _train_acc(epc_none, config, train_data)
    acc_spectral = _train_acc(epc_spectral, config, train_data)
    norms_none = _credit_norms(epc_none, substrate, train_data[0])
    norms_spectral = _credit_norms(epc_spectral, substrate, train_data[0])
    print(
        f"epc depth {DEPTH}: none {acc_none:.3f} -> spectral {acc_spectral:.3f}"
        f"; norms none {norms_none} spectral {norms_spectral}"
    )
    assert acc_spectral > acc_none + 0.03, (
        f"A4 repair must lift depth-{DEPTH} ePC (spectral {acc_spectral:.3f} "
        f"vs none {acc_none:.3f})"
    )
    assert min(norms_spectral) > 0, (
        "spectral credit_norm must leave signal in every hidden layer"
    )
    spread_none = max(norms_none) / (min(norms_none) + 1e-12)
    spread_spectral = max(norms_spectral) / (min(norms_spectral) + 1e-12)
    assert spread_spectral < spread_none, (
        "spectral credit_norm must flatten the per-layer credit-norm spread "
        f"({spread_spectral:.1f} vs none {spread_none:.1f})"
    )

    spc = _spc_system(substrate)
    spc_norms = _credit_norms(spc, substrate, train_data[0])
    print(f"spc norms {spc_norms}")
    assert all(n == 0.0 for n in spc_norms[:-1]), (  # noqa: RUF069 — exact-zero IS the ratchet (F1 precedent)
        "sPC hidden credit norms must be exactly zero (the blocked channel)"
    )
    assert all(n > 0 for n in norms_none[:-1]), (
        "ePC reparameterization must reach every hidden layer"
    )
    return {
        "epc_d8_none": acc_none,
        "epc_d8_spectral": acc_spectral,
        "credit_norms": {
            "epc_none": norms_none,
            "epc_spectral": norms_spectral,
            "spc": spc_norms,
        },
    }


def _assert_record_ratchets() -> None:
    """The pinned mechanisms must not regress (F4's ratchet locks)."""
    d18 = _load("d18_update_ladder")["arms"]
    assert d18["epc_w64_unit_rms"]["mean"] < d18["epc_w64_muon"]["mean"], (
        "D18 ratchet: unit_rms must beat Muon on ePC w64 (crutch dead)"
    )
    assert d18["epc_w32_unit_rms"]["mean"] < d18["epc_w32_muon"]["mean"], (
        "D18 ratchet: unit_rms must beat Muon on ePC w32 (crutch dead)"
    )
    assert d18["pepita_w32_unit_rms"]["mean"] > 1e4, (
        "D18 ratchet: the PEPITA misaligned-channel cell stays diverged "
        "(honest OPEN row until the faithful realization is tested)"
    )
    d16 = _load("d16_uaxis_coverage")["arms"]
    for geo in ("mlp_d2_w64", "graph_grid8x4", "lattice3d"):
        assert d16[f"{geo}/unit_rms"]["mean"] < CHANCE + 0.05, (
            f"D16 ratchet: unit_rms vision-quick boundary holds on {geo}"
        )
    f1 = _load("f1_failure_manifesto")["arms"]
    assert f1["bp"]["train_acc"][0] - f1["bp"]["train_acc"][-1] > 0.4, (
        "F1 ratchet: backprop decays through depth (the faithful-regime "
        "attenuation contrast)"
    )
    assert f1["hebbian_runaway"]["norm_ratio"][-1] > 1e4, (
        "F1 ratchet: the unnormalized local chain still runaways at depth 100"
    )
    assert f1["oja_collapse"]["readout_acc"][-1] < CHANCE + 0.15, (
        "F1 ratchet: normalized Oja chain still collapses toward chance"
    )
    f2 = _load("f2_spiking_plateau")
    assert f2["supervised_train_acc"] < CHANCE + 0.02, (
        "F2 ratchet: timing-STDP supervised accuracy stays at chance "
        "(the absent channel, B5 pending)"
    )
    d14 = _load("d14_jpc_faithful_depth")["arms"]
    assert d14["mupc_beta10"]["test"] > 0.8, (
        "D14 ratchet: the faithful regime stays self-sufficient at depth 20 "
        "(no credit-side lever needed)"
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_demo_credit_channel_map(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    loader = task.get_dataloader("train")  # type: ignore[attr-defined]
    train_data = list(_flatten(loader, BATCH_CAP))
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)

    _assert_record_ratchets()
    live = _run_live_arms(substrate, config, train_data)

    record = {
        "rows": [{"failure_mode": name, "status": status} for name, status in _ROWS],
        "live_arms": {
            "epc_d8_none": live["epc_d8_none"],
            "epc_d8_spectral": live["epc_d8_spectral"],
        },
        "live_credit_norms": live["credit_norms"],
        "figure": figure_spec(
            "F4 — the credit-channel failure map: eight faces, repairs landed and open",
            heatmap_panel(
                grid=[[status] for _, status in _ROWS],
                row_labels=[name for name, _ in _ROWS],
                col_labels=["repair status"],
                cmap="viridis",
                annotate=True,
                vmin=0,
                vmax=2,
                title="2 = live repair, 1 = pinned repair, 0 = open (see record)",
            ),
            figsize=[7.0, 4.5],
        ),
    }
    emit_run_record("F4", "credit_channel_map", record)
