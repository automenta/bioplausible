"""P3 — Muon lr-matched controls on LM (Fundamental-Research Focus).

Both LM headline candidates (mlp/ff_hybrid/muon, mlp/epc_thermo/muon) are
×Muon, and optimizer effects have masqueraded as mechanism twice before
(F3 routing, D3). Protocol (the P-axis lr-matched pilot's, applied to LM):

1. Measure each arm's EFFECTIVE per-step displacement ||Δθ||_F / step from
   identical inits (Muon's direction normalization makes its effective step
   ≠ step_size; Euclid's is lr·||g||).
2. Match the Euclidean lr to Muon's measured effective step by log-interp
   on a displacement grid.
3. Run Muon vs Euclid@matched for a fixed wall-clock budget on each
   credit's verified shape (ff_hybrid at smoke w64×4; epc_thermo at
   registered w816×7 — it explodes at w64) and compare val ppl on the
   shared fixed val windows.

Outcome interpretation: equal ppl at matched step → the "Muon is
load-bearing" reading is an effective-lr artifact; Muon wins at matched
step → a direction-quality mechanism claim becomes quotable.

Run: uv run python scripts/probes/lm_muon_lr_matched.py
"""

import itertools
import math
import time

import lm_comparison as lmc
import torch

from computronium import (
    AdamUpdate,
    CreditAssignmentConfig,
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.ontology.update import RiemannianOrthogonalUpdate

BUDGET_MIN = 2.0
MEASURE_STEPS = 20
EUC_GRID = (0.01, 0.03, 0.1, 0.3)
ARMS = {
    # credit: (geometry cfg, muon lr of record)
    "ff_hybrid": (lmc.MLP_SMOKE, 0.02),
    "epc_thermo": (lmc.MLP_CFG, 0.01),
}


def build(credit: str, cfg: dict, update: str, lr: float):
    if credit == "ff_hybrid":
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff", readout_error=True
            )
        )
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    else:
        credit_obj = ThermodynamicContrast()
        dynamics = ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(max_steps=10, step_size=0.1)
        )
    if update == "muon":
        upd = RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr, momentum=0.9)
        )
    elif update == "adam":
        upd = AdamUpdate(ParameterUpdateConfig.adam(step_size=lr))
    else:
        upd = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr))
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=lmc.DEVICE)),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=cfg["ctx"] * lmc.VOCAB,
                output_dim=lmc.VOCAB,
                hidden_dims=cfg["hidden"],
            )
        ),
        dynamics=dynamics,
        credit=credit_obj,
        update=upd,
    )


def _batch(tokens, ctx, batch, gen):
    idx = torch.randint(0, len(tokens) - ctx - 1, (batch,), generator=gen)
    win = tokens[idx.unsqueeze(1) + torch.arange(ctx + 1)]
    x = (
        torch.nn.functional
        .one_hot(win[:, :-1], lmc.VOCAB)
        .float()
        .reshape(batch, ctx * lmc.VOCAB)
        .to(lmc.DEVICE)
    )
    return x, win[:, -1].to(lmc.DEVICE)


def eff_step(
    credit: str, cfg: dict, update: str, lr: float, tokens, seed: int
) -> float:
    """Mean per-step ||Δθ||_F over MEASURE_STEPS from an identical init."""
    torch.manual_seed(seed)
    system = build(credit, cfg, update, lr)
    system.geometry.to(lmc.DEVICE)
    names = list(system.geometry.params)
    before = [system.geometry.params[n].detach().clone() for n in names]
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(MEASURE_STEPS):
        x, y = _batch(tokens, cfg["ctx"], 32, gen)
        run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )
    delta = (
        sum(
            (system.geometry.params[n].detach() - b).norm().item() ** 2
            for n, b in zip(names, before, strict=True)
        )
        ** 0.5
    )
    return delta / MEASURE_STEPS


def run_arm(
    credit: str, cfg: dict, update: str, lr: float, tokens, val, seed: int
) -> dict:
    torch.manual_seed(seed)
    system = build(credit, cfg, update, lr)
    system.geometry.to(lmc.DEVICE)
    gen = torch.Generator().manual_seed(seed + 1)
    curve = []
    t0 = time.time()
    step = 0
    while time.time() - t0 < BUDGET_MIN * 60:
        x, y = _batch(tokens, cfg["ctx"], 32, gen)
        metrics = run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )
        step += 1
        if time.time() - t0 > lmc.EVAL_INTERVAL_S * (len(curve) + 1):
            curve.append({
                "t": round(time.time() - t0, 1),
                "train_loss": round(metrics["loss"], 4),
                **lmc._eval(system, val, "mlp"),
            })
    return {
        "arm": f"mlp/{credit}/{update}@lr{lr}",
        "steps": step,
        "curve": curve,
        "final": curve[-1] if curve else lmc._eval(system, val, "mlp"),
    }


def main() -> None:
    torch.manual_seed(0)
    train_t, val_t = lmc.load_tokens()
    _m_val = lmc._val_sets(val_t, lmc.MLP_SMOKE["ctx"])
    seed = 0
    for credit, (cfg, muon_lr) in ARMS.items():
        m_val_c = lmc._val_sets(val_t, cfg["ctx"])[1]
        s_muon = eff_step(credit, cfg, "muon", muon_lr, train_t, seed)
        print(f"\n=== {credit}: muon lr {muon_lr} → eff step {s_muon:.4f} ===")
        grid = {
            lr: eff_step(credit, cfg, "euclid", lr, train_t, seed) for lr in EUC_GRID
        }
        for lr, s in grid.items():
            print(f"  euclid lr {lr}: eff step {s:.4f}")
        # log-interp to match the muon displacement
        lrs = sorted(grid)
        matched = muon_lr
        for lo, hi in itertools.pairwise(lrs):
            if grid[lo] <= s_muon <= grid[hi]:
                frac = (math.log(s_muon) - math.log(grid[lo])) / (
                    math.log(grid[hi]) - math.log(grid[lo])
                )
                matched = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
                break
        if s_muon > grid[lrs[-1]]:
            matched = lrs[-1] * (s_muon / grid[lrs[-1]])
        print(f"  matched euclid lr: {matched:.4f}")
        for update, lr in (("muon", muon_lr), ("euclid", matched)):
            r = run_arm(credit, cfg, update, lr, train_t, m_val_c, seed)
            print(
                f"  {r['arm']:>40}: steps {r['steps']:>4} "
                f"final {r['final']}  curve {r['curve'][-3:]}"
            )


if __name__ == "__main__":
    main()
