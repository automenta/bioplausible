"""D15 — The U-axis moves the depth wall: capacity-matched update swap.

F1's depth wall (BP at chance from depth 8) was measured with Euclidean
SGD at width 32. This demo re-measures the frontier at width 128 with
capacity-matched geometries (every comparison runs at IDENTICAL
parameter counts — same geometry, swapped update or credit rule),
mnist quick, 300 batches, 1 epoch, TEST accuracy:

1. **The headline, multi-seed verified (seeds 0-2):** at depth 16 /
   width 128 (349,450 params BOTH arms), BP×Euclidean is at chance
   (0.114 ± 0.000) while BP×Muon reaches 0.834 ± 0.036 — the depth wall
   is an optimizer artifact, not a backprop limit: orthogonalizing the
   UPDATE moves it beyond depth 16. Measured pilot: 0.114 vs 0.834.
2. **Local ≥ global, capacity-matched (seeds 0-2):** FF×Muon 0.930 ±
   0.001 vs BP×Muon 0.911 ± 0.007 at depth 4 / width 256 (400,906
   params both) — the layer-local goodness rule BEATS backprop at
   matched capacity under the same update.
3. **The depth curve, single-seed live (F1 convention):** BP degrades
   gracefully under Muon through depth 16 while Euclid cliffs — the
   frontier shifts, and the failure mode is update-conditioned.

Record carries per-arm parameter counts. Every compared pair is at
identical geometry — no capacity confound (the D8–D12 convention).
"""

from itertools import islice

import numpy as np
import pytest
import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate
from computronium.visualization._demo_api import bars_panel, figure_spec, lines_panel

BATCH_CAP = 300
LR_MUON = 0.02
LR_EUCLID = 0.2
MULTI_SEEDS = (0, 1, 2)
CHANCE = 0.1


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _run(credit: str, update: str, depth: int, width: int, seed: int) -> float:
    torch.manual_seed(seed)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(width,) * depth
        )
    )
    if credit == "bp":
        credit_obj = BackpropCredit()
    else:
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff"
            )
        )
    if update == "muon":
        update_obj = RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=LR_MUON, momentum=0.9)
        )
    else:
        update_obj = EuclideanUpdate(
            ParameterUpdateConfig.euclidean(step_size=LR_EUCLID)
        )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit_obj,
        update=update_obj,
    )
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=_TRAIN_DATA,
    ).fit()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in _TEST_BATCHES:
            x = batch_x.view(batch_x.size(0), -1)
            state = system.dynamics.settle(
                SystemState(x=x), system.geometry, system.substrate, None
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            correct += (out.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def _n_params(depth: int, width: int) -> int:
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(width,) * depth
        )
    )
    return sum(p.numel() for p in geometry.params.values())


_TASK = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
_TASK.setup()
torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
_TRAIN_DATA = list(_flatten(_TASK.get_dataloader("train"), BATCH_CAP))
_TEST_BATCHES = list(_TASK.get_dataloader("test"))


# Slow tier: ~240 s (3 multi-seed arms + 3 depth curves at width 128).
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_demo_uaxis_depth_frontier(emit_run_record) -> None:
    # Headline pairs, multi-seed (capacity-matched: identical geometry).
    headline: dict[str, list[float]] = {}
    for label, credit, update, depth, width in [
        ("bp/euclid d16 w128", "bp", "euclid", 16, 128),
        ("bp/muon d16 w128", "bp", "muon", 16, 128),
        ("bp/muon d4 w256", "bp", "muon", 4, 256),
        ("ff/muon d4 w256", "ff", "muon", 4, 256),
    ]:
        accs = [_run(credit, update, depth, width, s) for s in MULTI_SEEDS]
        headline[label] = accs
        print(f"{label:>20}: {np.mean(accs):.3f} ± {np.std(accs):.3f} {accs}")

    # Depth curves, single-seed live (F1 convention: comparative margins).
    depths = (2, 4, 8, 16)
    curves: dict[str, list[float]] = {}
    for series, credit, update in (
        ("bp_euclid", "bp", "euclid"),
        ("bp_muon", "bp", "muon"),
        ("ff_muon", "ff", "muon"),
    ):
        curve = [_run(credit, update, d, 128, 0) for d in depths]
        curves[series] = curve
        print(f"{series:>12} w128: {[round(a, 3) for a in curve]}")

    params = {d: _n_params(d, 128) for d in depths}
    params[256] = _n_params(4, 256)

    record: dict = {
        "seeds": list(MULTI_SEEDS),
        "batch_cap": BATCH_CAP,
        "lr_muon": LR_MUON,
        "lr_euclid": LR_EUCLID,
        "chance": CHANCE,
        "device": "cpu",
        "params": {str(k): v for k, v in params.items()},
        "headline": {k: list(v) for k, v in headline.items()},
        "depth_curves": {k: list(v) for k, v in curves.items()},
        "figure": figure_spec(
            "D15 — Muon moves the depth wall (capacity-matched, mnist)",
            bars_panel(
                {
                    "depth 16 / width 128 (349,450 params both)": {
                        "BP×Euclid": float(np.mean(headline["bp/euclid d16 w128"])),
                        "BP×Muon": float(np.mean(headline["bp/muon d16 w128"])),
                    },
                    "depth 4 / width 256 (400,906 params both)": {
                        "BP×Muon": float(np.mean(headline["bp/muon d4 w256"])),
                        "FF×Muon": float(np.mean(headline["ff/muon d4 w256"])),
                    },
                },
                xlabel="",
                ylabel="test accuracy",
                chance=CHANCE,
            ),
            lines_panel(
                curves,
                x=list(depths),
                xlabel="depth (width 128)",
                ylabel="test accuracy",
                series_labels=["BP×Euclid", "BP×Muon", "FF×Muon"],
                chance=CHANCE,
            ),
            figsize=[12.0, 4.5],
        ),
    }
    emit_run_record("D15", "uaxis_depth_frontier", record)

    # The wall MOVED: at identical params, Euclid is at chance at depth 16
    # while Muon generalizes. Per-seed variance-aware (min margin 0.6).
    euclid_d16 = headline["bp/euclid d16 w128"]
    muon_d16 = headline["bp/muon d16 w128"]
    assert max(euclid_d16) < 0.2, (
        "bp/euclid d16 must be at (near) chance — the wall must be live "
        "for the move-the-wall claim"
    )
    assert min(muon_d16) > 0.6, (
        f"bp/muon d16 per-seed floor broken: {muon_d16} — the depth wall "
        "moved back: re-audit"
    )

    # Local ≥ global at matched capacity under the same update.
    ff_d4 = headline["ff/muon d4 w256"]
    bp_d4 = headline["bp/muon d4 w256"]
    assert np.mean(ff_d4) >= np.mean(bp_d4), (
        f"ff/muon must retain its capacity-matched edge over bp/muon "
        f"({np.mean(ff_d4):.3f} vs {np.mean(bp_d4):.3f}) — re-audit"
    )
    assert min(ff_d4) > 0.9, f"ff/muon d4 w256 per-seed floor broken: {ff_d4}"

    # The curve claims: Muon degrades gracefully where Euclid cliffs.
    assert curves["bp_euclid"][-1] < 0.2 and curves["bp_muon"][-1] > 0.6, (
        "the depth-frontier contrast must hold on the single-seed curves"
    )
    assert curves["ff_muon"][0] > curves["bp_euclid"][0], (
        "ff/muon must beat bp/euclid at shallow depth too (U-axis lift)"
    )
