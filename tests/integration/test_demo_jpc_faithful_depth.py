"""D14 — depth 20 trains under the jpc-faithful regime, and the μPC lift
is real (Path A, R11.3.11 + F1 tail).

The prior "no μPC lift under our trainer" verdicts (R11.3.11b, F1) were
trainer-regime artifacts: Euclidean SGD, β=0.5, fixed 60 settle steps.
Under the source papers' regime — ePC error reparameterization, Adam on
weights, nudge β from the grid, inference steps = H, residual geometry —
this demo shows at depth 20 / width 128:

1. **μPC generalizes where default init memorizes:** mupc+β=10 reaches
   test ≈ 0.69–0.83 (seeds 0–2, probe) while default+β=10 memorizes
   (train ≈ 1.00, test ≈ 0.14–0.24) and default+β=1e3 is degenerate
   (train 1.000, test ≈ 0.09 — at or below chance). The (N·L)^{-1/2}
   hidden scale is load-bearing at depth, exactly as the paper claims.
2. **The F1 depth wall dissolves under the faithful regime:** depth 20
   trains to ≈ 0.9 train / ≈ 0.8 test where the sPC/thermo/Euclid
   instrument walled at chance from depth 8. The wall is
   solver+regime-specific, not physics.
3. **β matters:** β=1e3 (the paper's upper grid edge) pushes every init
   into the memorization corner here; β=10 is the working regime at
   this scale. Single-seed demo with comparative margins (the multi-seed
   probe numbers live in scripts/probes/jpc_faithful.py; the measured
   per-seed lift gap is ≥ 0.54).
"""

from itertools import islice

import pytest
import torch
from torch import optim

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    create_task,
)
from computronium.ontology._settle_kernel import extract_layered_params

WIDTH = 128
DEPTH = 20
BATCH_CAP = 150
EVAL_CAP = 20
ADAM_LR = 1e-3
ACTIVITY_STEP = 0.1
CHANCE = 0.1


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _task() -> object:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    return task


def _run_arm(init: str, beta: float, train_data) -> dict:
    torch.manual_seed(1)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * DEPTH,
            init_scheme=init,
            residual=True,
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=DEPTH,  # inference steps = H
            step_size=ACTIVITY_STEP,
            beta=beta,
            convergence_threshold=0.0,  # fixed budget, no early exit
            convergence_start=DEPTH + 1,
        )
    )
    layered = extract_layered_params(geometry)
    weights = [t[0] for t in layered.transitions]
    adam = optim.Adam(weights, lr=ADAM_LR)

    n = total = 0
    for x, y in train_data:
        dynamics.settle(SystemState(x=x), geometry, substrate, None)
        dynamics.settle(SystemState(x=x), geometry, substrate, y)
        eps = [e.detach() for e in dynamics._last_errors]

        # PC-native weight gradient: dE_nudged/dtheta with the settled
        # errors frozen — the paper's Δθᵢ ∝ (∂sᵢ/∂θᵢ)ᵀ εᵢ (one reverse-mode
        # sweep, unattenuated).
        with torch.enable_grad():
            _, y_hat = dynamics._build_forward_with_errors(
                x, layered.transitions, substrate, eps, residual=True
            )
            energy = beta * torch.nn.functional.cross_entropy(y_hat, y)
            grads = torch.autograd.grad(energy, weights)

        adam.zero_grad()
        for w, g in zip(weights, grads, strict=True):
            w.grad = g
        adam.step()
        n += (y_hat.argmax(1) == y).sum().item()
        total += y.shape[0]

    return {
        "train": n / total,
        "test": _eval(dynamics, geometry, substrate),
    }


def _eval(dynamics, geometry, substrate) -> float:
    """Held-out accuracy via the free (inference) settle."""
    task = _task()
    correct = total = 0
    for x, y in _flatten(task.get_dataloader("test"), EVAL_CAP):
        state = dynamics.settle(SystemState(x=x), geometry, substrate, None)
        correct += (state.activations[-1].argmax(1) == y).sum().item()
        total += y.shape[0]
    return correct / total


# Slow tier (R11.5.7 re-baseline): the three depth-20 arms are the demo
# suite's first slow-tier resident — the fast gate proves D1–D13+F1–F3 in
# ~190 s; run this with `pytest -m slow -k demo` (~107 s).
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_demo_jpc_faithful_depth(emit_run_record) -> None:
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    task = _task()
    train_data = list(_flatten(task.get_dataloader("train"), BATCH_CAP))

    record: dict = {
        "regime": {
            "width": WIDTH,
            "depth": DEPTH,
            "adam_lr": ADAM_LR,
            "activity_step": ACTIVITY_STEP,
            "inference_steps": DEPTH,
            "batch_cap": BATCH_CAP,
        },
        "arms": {},
    }
    for name, init, beta in (
        ("mupc_beta10", "mupc", 10.0),
        ("default_beta10", "default", 10.0),
        ("mupc_beta1e3", "mupc", 1e3),
    ):
        record["arms"][name] = _run_arm(init, beta, train_data)
        print(f"{name:>16}: {record['arms'][name]}")

    # Common demo API: the figure is declared IN the record, next to the
    # data it presents — one generic renderer owns styling, chance lines,
    # and value labels.
    record["figure"] = {
        "title": (
            "D14 — depth 20 under the jpc-faithful regime: "
            "μPC generalizes where default init memorizes"
        ),
        "figsize": [7.5, 4.5],
        "panels": [
            {
                "type": "bars",
                "groups": record["arms"],
                "series_labels": {"train": "train", "test": "test (held-out)"},
                "chance": CHANCE,
                "chance_label": "chance (0.1)",
                "ylabel": "accuracy",
                "ylim": [0, 1],
            }
        ],
    }

    emit_run_record("D14", "jpc_faithful_depth", record)

    mupc = record["arms"]["mupc_beta10"]
    default = record["arms"]["default_beta10"]
    assert mupc["test"] > 5 * CHANCE, (
        "μPC + jpc-faithful regime must generalize at depth 20 "
        f"(probe: 0.69–0.83 over seeds 0–2; got {mupc['test']:.3f})"
    )
    assert mupc["test"] > default["test"] + 0.3, (
        "the μPC lift over default init at depth 20 is the claim "
        "(probe gap ≥ 0.54 per seed)"
    )
    assert default["train"] > 0.9 and default["test"] < 0.4, (
        "default init must sit in the memorization corner "
        "(train ≫ test — the failure mode μPC fixes)"
    )
    assert record["arms"]["mupc_beta1e3"]["test"] < mupc["test"] - 0.2, (
        "β=1e3 must land in the memorization corner at this scale "
        "(β is a working regime knob, not a monotone dial)"
    )
