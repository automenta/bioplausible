"""P4 — width fragility of local feedback on LM (Fundamental-Research Focus).

Measured facts to explain: ePC×Muon and pepita explode at w64 (smoke
shape) and train at w256+; pure FF is flat at every width. One sweep:
width ∈ {32, 64, 128, 256} × credit ∈ {ff, ff_hybrid, pepita, epc_thermo},
depth 4 (fixed — isolates width), Muon at each credit's tuned lm lr,
fixed 75 s budget per arm, ctx 32, batch 32.

Recorded per arm: final val ppl (chance 65), stability (finite final
loss), and per-layer hidden-activation std at the end (the signal-to-noise
reading: local feedback's usable signal vs the activity scale it rides).

Caveat (scope-honest): lrs are the registered per-credit values, not
re-tuned per width — the claim is fragility at the working regime, and
width-dependent lr sensitivity is itself part of the fragility.

Run: uv run python scripts/probes/p4_width_fragility.py
"""

import math
import time

import lm_comparison as lmc
import torch

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.ontology.update import RiemannianOrthogonalUpdate

WIDTHS = (32, 64, 128, 256)
DEPTH = 4
BUDGET_S = 75.0
CTX = 32
BATCH = 32
LR = {"ff": 0.01, "ff_hybrid": 0.02, "pepita": 5e-4, "epc_thermo": 0.01}


def build(credit: str, width: int):
    if credit == "ff_hybrid":
        credit_obj = LocalGoodnessCredit(
            lmc.CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff", readout_error=True
            )
        )
    elif credit == "pepita":
        credit_obj = LocalGoodnessCredit(
            lmc.CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="pepita"
            )
        )
    elif credit == "ff":
        credit_obj = LocalGoodnessCredit(
            lmc.CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff"
            )
        )
    else:
        credit_obj = ThermodynamicContrast()
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=CTX * lmc.VOCAB,
            output_dim=lmc.VOCAB,
            hidden_dims=(width,) * DEPTH,
        )
    )
    if credit == "epc_thermo":
        dynamics = ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(max_steps=10, step_size=0.1)
        )
    else:
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=lmc.DEVICE)),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit_obj,
        update=RiemannianOrthogonalUpdate(
            lmc.ParameterUpdateConfig.riemannian_orthogonal(
                step_size=LR[credit], momentum=0.9
            )
        ),
    )


def act_stds(system) -> list[float]:
    x = torch.randn(8, CTX * lmc.VOCAB, device=lmc.DEVICE)
    with torch.no_grad():
        state = system.dynamics.settle(
            SystemState(x=x), system.geometry, system.substrate, None
        )
    acts = state.activations
    return [float(a.std()) for a in (acts if isinstance(acts, list) else [acts])]


def run_arm(system, tokens, val, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    step = 0
    train_loss = float("nan")
    while time.time() - t0 < BUDGET_S:
        idx = torch.randint(0, len(tokens) - CTX - 1, (BATCH,), generator=gen)
        win = tokens[idx.unsqueeze(1) + torch.arange(CTX + 1)]
        x = (
            torch.nn.functional
            .one_hot(win[:, :-1], lmc.VOCAB)
            .float()
            .reshape(BATCH, CTX * lmc.VOCAB)
            .to(lmc.DEVICE)
        )
        y = win[:, -1].to(lmc.DEVICE)
        train_loss = run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )["loss"]
        step += 1
    stds = act_stds(system)
    stable = math.isfinite(train_loss) and train_loss < 1e4
    return {
        "steps": step,
        "train_loss": round(train_loss, 4),
        "stable": stable,
        "act_std": [round(s, 4) for s in stds],
    }


def main() -> None:
    torch.manual_seed(0)
    train_t, val_t = lmc.load_tokens()
    _, m_val = lmc._val_sets(val_t, CTX)
    print(f"{'credit':>12} {'width':>5}  val_ppl  act_std (hidden+out)")
    for credit in ("ff", "ff_hybrid", "pepita", "epc_thermo"):
        for width in WIDTHS:
            torch.manual_seed(0)
            system = build(credit, width)
            system.geometry.to(lmc.DEVICE)  # type: ignore[attr-defined]
            r = run_arm(system, train_t, m_val, seed=0)
            val = lmc._eval(system, m_val, "mlp")
            print(
                f"{credit:>12} {width:>5}  {val['val_ppl']:>7}  {r['act_std']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
