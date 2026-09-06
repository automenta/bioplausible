"""P2 — the jpc-faithful PC trainer for LM (Fundamental-Research Focus).

Port of the D14 regime (ePC settle steps=H, PC-native weight gradient with
frozen settled errors, Adam) to the LM task: the MLP arm (one-hot context
window -> next char, biases present as ePC requires), registered width
w816 x 7. This replaces the ÷β contrastive credit (ThermodynamicContrast)
— the theory-faithful energy-gradient readout — with the D14 recipe:

  1. free settle: ePC, max_steps = H (depth, fixed budget)
  2. nudged settle: ePC, beta from the grid, driven by CE on the target
  3. PC-native weight gradient: d(beta*CE)/dtheta with settled errors
     frozen — equals Delta theta_i ~ (ds_i/dtheta_i)^T eps_i
  4. torch.optim.Adam step

Capacity-matched baseline (same geometry, same val set, same seed, 2-min
budget): epc_thermo/muon val_ppl 28.01 (scripts/probes/lm_muon_lr_matched.py,
seed 0). If the jpc loop beats that at matched wall-clock, the energy-based
LM trainer gets its paper-grounded regime; if not, the contrastive credit
is the better instrument for this task.

Run: uv run python scripts/probes/p2_jpc_lm.py
"""

import math
import time

import lm_comparison as lmc
import torch
from lm_muon_lr_matched import _batch

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
)
from computronium.ontology._settle_kernel import extract_layered_params

ADAM_LR = 1e-3
BETAS = (1e3, 100.0, 10.0)
BUDGET_MIN = 2.0
GAMMA = 0.1


def run_arm(beta: float, tokens, val, seed: int) -> dict:  # ruff: ignore[too-many-locals]
    torch.manual_seed(seed)
    cfg = lmc.MLP_CFG
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=cfg["ctx"] * lmc.VOCAB,
            output_dim=lmc.VOCAB,
            hidden_dims=cfg["hidden"],
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device=lmc.DEVICE))
    depth = len(cfg["hidden"])
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=depth,  # inference steps = H
            step_size=GAMMA,
            beta=beta,
            convergence_threshold=0.0,
            convergence_start=depth + 1,
        )
    )
    geometry.to(lmc.DEVICE)
    layered = extract_layered_params(geometry)
    weights = [t[0] for t in layered.transitions]
    adam = torch.optim.Adam(weights, lr=ADAM_LR)
    gen = torch.Generator().manual_seed(seed + 1)
    curve = []
    t0 = time.time()
    step = 0
    while time.time() - t0 < BUDGET_MIN * 60:
        x, y = _batch(tokens, cfg["ctx"], 32, gen)
        dynamics.settle(SystemState(x=x), geometry, substrate, None)
        dynamics.settle(SystemState(x=x), geometry, substrate, y)
        eps = [e.detach() for e in dynamics._last_errors]
        with torch.enable_grad():
            _, y_hat = dynamics._build_forward_with_errors(
                x, layered.transitions, substrate, eps, residual=False
            )
            energy = beta * torch.nn.functional.cross_entropy(y_hat, y)
            grads = torch.autograd.grad(energy, weights)
        adam.zero_grad()
        for w, g in zip(weights, grads, strict=True):
            w.grad = g
        adam.step()
        step += 1
        if time.time() - t0 > lmc.EVAL_INTERVAL_S * (len(curve) + 1):
            curve.append({
                "t": round(time.time() - t0, 1),
                "train_loss": round(energy.item() / beta, 4),
                **lmc._eval(
                    system=(dynamics, geometry, substrate), val=val, geom="jpc"
                ),
            })
    print(f"beta {beta:<6}: steps {step:>4} curve {curve[-2:]}", flush=True)
    return {"beta": beta, "steps": step, "curve": curve}


def _eval_jpc(dynamics, geometry, substrate, val) -> dict:
    tot = n = 0
    with torch.no_grad():
        for x, y in val:
            state = dynamics.settle(
                SystemState(x=x.to(lmc.DEVICE)), geometry, substrate, None
            )
            acts = state.activations
            logits = acts[-1] if isinstance(acts, list) else acts
            loss = torch.nn.functional.cross_entropy(
                logits, y.to(lmc.DEVICE), reduction="sum"
            )
            tot += loss.item()
            n += y.numel()
    avg = tot / n
    return {"val_loss": round(avg, 4), "val_ppl": round(math.exp(min(avg, 20)), 2)}


def main() -> None:
    torch.manual_seed(0)
    train_t, val_t = lmc.load_tokens()
    _, m_val = lmc._val_sets(val_t, lmc.MLP_CFG["ctx"])
    # patch lmc._eval so the curve uses the jpc free-settle readout
    lmc._eval = lambda system, val, geom=None: _eval_jpc(*system[:3], val)  # ruff: ignore[unused-lambda-argument]
    for beta in BETAS:
        run_arm(beta, train_t, m_val, seed=0)


if __name__ == "__main__":
    main()
