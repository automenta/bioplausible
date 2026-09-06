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
4. **Orthogonalized-momentum Adam (OrthoAdam, ortho_lr 1e-3) lifts the
   whole regime and shrinks the μPC lift** (hunt cell, probe
   scripts/probes/jpc_ortho_adam.py, seeds 0–2): mupc×ortho 0.914 /
   0.923 / 0.922 and default×ortho 0.859 / 0.838 / 0.856 — the
   default-init memorization corner (test ≈ 0.20 under Adam) is largely
   rescued, so the μPC-vs-default gap narrows from ≈ 0.58 to ≈ 0.07.
   Momentum-orthogonalization and depth-scaled init are partially
   interchangeable repairs of the same depth pathology; μPC still leads.
   The lr is sharp (3e-3 — the D16-calibrated value — degrades μPC to
   ≈ 0.52: a step-size artifact, the natural-gradient lesson again).
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
ORTHO_LR = 1e-3  # the jpc-regime calibration (probe lr sweep: 5e-4–1e-3 plateau, 3e-3 degrades)
ACTIVITY_STEP = 0.1
CHANCE = 0.1


class _OrthoAdamWeights:
    """Adam moments + SVD-polar matrix directions over a plain weight
    list — the OrthoAdamUpdate recipe for the manual jpc loop."""

    def __init__(self, weights: list[torch.Tensor], lr: float):
        self.weights = weights
        self.lr = lr
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.m = [torch.zeros_like(w) for w in weights]
        self.v = [torch.zeros_like(w) for w in weights]

    def step(self, grads: list[torch.Tensor]) -> None:
        self.t += 1
        bias1 = 1 - self.beta1**self.t
        bias2 = 1 - self.beta2**self.t
        with torch.no_grad():
            for w, g, m, v in zip(self.weights, grads, self.m, self.v, strict=True):
                m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                m_hat = m / bias1
                adam_step = m_hat / (v / bias2).sqrt().add_(self.eps)
                if w.ndim == 2:
                    U, _, Vh = torch.linalg.svd(m_hat, full_matrices=False)
                    ortho = U @ Vh
                    ortho *= adam_step.norm() / (ortho.norm() + 1e-8)
                    w.add_(ortho, alpha=-self.lr)
                else:
                    w.add_(adam_step, alpha=-self.lr)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _task() -> object:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    return task


def _run_arm(init: str, beta: float, train_data, optimizer: str = "adam") -> dict:
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
    ortho = _OrthoAdamWeights(weights, ORTHO_LR)

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

        if optimizer == "ortho_adam":
            ortho.step([g.detach() for g in grads])
        else:
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
    for name, init, beta, optimizer in (
        ("mupc_beta10", "mupc", 10.0, "adam"),
        ("default_beta10", "default", 10.0, "adam"),
        ("mupc_beta1e3", "mupc", 1e3, "adam"),
        ("mupc_ortho", "mupc", 10.0, "ortho_adam"),
        ("default_ortho", "default", 10.0, "ortho_adam"),
    ):
        record["arms"][name] = _run_arm(init, beta, train_data, optimizer)
        print(f"{name:>16}: {record['arms'][name]}")

    # Common demo API: the figure is declared IN the record, next to the
    # data it presents — one generic renderer owns styling, chance lines,
    # and value labels.
    record["figure"] = {
        "title": (
            "D14 — depth 20 under the jpc-faithful regime: μPC generalizes "
            "where default init memorizes; OrthoAdam lifts both inits"
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
    mupc_ortho = record["arms"]["mupc_ortho"]
    default_ortho = record["arms"]["default_ortho"]
    assert mupc_ortho["test"] > mupc["test"] + 0.05, (
        "OrthoAdam must lift the μPC regime beyond plain Adam "
        f"(probe gap +0.095 at this seed; got {mupc_ortho['test']:.3f} vs "
        f"{mupc['test']:.3f})"
    )
    assert default_ortho["test"] > default["test"] + 0.3, (
        "OrthoAdam must largely rescue the default-init memorization corner "
        f"(probe gap +0.6 at this seed; got {default_ortho['test']:.3f})"
    )
    assert mupc_ortho["test"] > default_ortho["test"] + 0.03, (
        "μPC still leads under OrthoAdam (probe per-seed gap ≥ 0.06; "
        "the lift narrows — the two repairs are partially interchangeable)"
    )
