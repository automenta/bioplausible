"""P1(b) — Dense-supervision isolation (Fundamental-Research Focus).

The MLP LM arms eat a ~9x tokens/s handicap (single target per window vs
dense CE over all T positions). A zero-block `TransformerGeometry`
(n_layers=0: embedding + head only, dense per-position logits) gives dense
supervision on a LOCAL-CREDIT-compatible geometry, isolating how much of
the transformer's advantage is attention vs supervision density.

Arms (2.5 min each, smoke transformer shape d64/3L/ctx32):
  full/bp/adam         — the conventional reference on this shape
  full/ff_hybrid/muon  — the known 6.74-ppl local rule (2.5-min table)
  zero/bp/adam         — dense supervision, NO hidden layers: the
                         supervision-density-only baseline (≈ unigram)
  zero/ff_hybrid/muon  — local credit + dense supervision, no hidden layers

Reading: if zero/ff_hybrid matches zero/bp, the local-credit machinery
expresses dense supervision fine and its deficit on the full transformer
is hidden-layer learning (attention), not supervision density. If
zero/ff_hybrid lags zero/bp, dense supervision alone doesn't rescue it.

Run: uv run python scripts/probes/p1b_dense_supervision.py
"""

import time

import lm_comparison as lmc
import torch

from computronium import (
    AdamUpdate,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.ontology.geometry import geometry_from_config
from computronium.ontology.update import RiemannianOrthogonalUpdate

BUDGET_MIN = 2.5
CFG = lmc.TRANSFORMER_SMOKE  # d64, 3 layers, 4 heads, ctx 32


def _build(n_layers: int, credit: str, d: int | None = None):
    geometry = geometry_from_config(
        GeometryConfig.causal_transformer(
            vocab_size=lmc.VOCAB,
            d_model=d or CFG["d"],
            n_layers=n_layers,
            n_heads=CFG["n_head"],
            seq_len=CFG["ctx"],
        )
    )
    if credit == "bp":
        credit_obj = BackpropCredit()
        update = AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3))
    else:
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff", readout_error=True
            )
        )
        update = RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=lmc.DEVICE)),
        geometry=geometry,
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit_obj,
        update=update,
    )


def run_arm(
    tag: str, n_layers: int, credit: str, tokens, val, seed: int, d: int | None = None
) -> dict:
    torch.manual_seed(seed)
    system = _build(n_layers, credit, d)
    system.geometry.to(lmc.DEVICE)  # type: ignore[attr-defined]
    n_params = sum(p.numel() for p in system.geometry.params.values())
    gen = torch.Generator().manual_seed(seed + 1)
    ctx = CFG["ctx"]
    curve = []
    t0 = time.time()
    step = 0
    while time.time() - t0 < BUDGET_MIN * 60:
        idx = torch.randint(0, len(tokens) - ctx - 1, (32,), generator=gen)
        win = tokens[idx.unsqueeze(1) + torch.arange(ctx + 1)]
        x, y = win[:, :-1].to(lmc.DEVICE), win[:, 1:].reshape(-1).to(lmc.DEVICE)
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
                **lmc._eval(system, val, "transformer"),
            })
    print(
        f"{tag:>24}: params {n_params:>8,} steps {step:>5} "
        f"final {curve[-1] if curve else 'n/a'}",
        flush=True,
    )
    return {"tag": tag, "params": n_params, "steps": step, "curve": curve}


def main() -> None:
    torch.manual_seed(0)
    train_t, val_t = lmc.load_tokens()
    t_val, _ = lmc._val_sets(val_t, CFG["ctx"])
    for tag, n_layers, credit in (
        ("full/bp/adam", CFG["n_layer"], "bp"),
        ("full/ff_hybrid/muon", CFG["n_layer"], "ff_hybrid"),
        ("zero/bp/adam", 0, "bp"),
        ("zero/ff_hybrid/muon", 0, "ff_hybrid"),
    ):
        run_arm(tag, n_layers, credit, train_t, t_val, seed=0)


if __name__ == "__main__":
    main()
