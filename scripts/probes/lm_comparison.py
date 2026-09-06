"""Small-LM comparison harness: the promising configurations vs a
conventional backprop transformer (user directive 2026-09-05).

Prepares, but does NOT start, the 1-hour-per-configuration comparison.
Run modes:

    uv run python scripts/probes/lm_comparison.py --smoke      # ~10 min total
    uv run python scripts/probes/lm_comparison.py --minutes 60 # the real runs

Data: tiny Shakespeare char-level (computronium.data.lm.get_lm_dataset;
HF fallback verified: 1.004M train / 55.8K val chars, vocab 65). Both
splits are encoded through ONE train-built vocab (per-split vocab ids
are different ciphers — a real bug found by the first smoke run).

Architecture: BOTH families run through the 5-axis ontology pipeline.

- ``transformer`` arms — `TransformerGeometry` (G-axis primitive:
  causal pre-LN attention blocks, sinusoidal positions, dense
  next-token supervision over all T positions).
- ``mlp`` arms — `FeedforwardGeometry` over a one-hot context window
  (single next-char target per window).

Both share the same credit x update space: bp/adam (the conventional
SOTA arm), bp/muon, bp/ortho_adam, ff/muon, ff/ortho_adam,
pepita/muon. FAIRNESS: parameter counts are capacity-matched per
family pair and printed for EVERY arm in the table (asserted).

Metrics per arm: wall-clock budget, steps, tokens seen, train loss +
val loss/ppl curves on a FIXED val window set (materialized once,
shared across arms), chars/s. Output: stdout table + JSON in
benchmark_results/ (untracked by standing directive).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from computronium import (
    AdamUpdate,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    OrthoAdamUpdate,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.data.lm import get_lm_dataset
from computronium.ontology.geometry import geometry_from_config
from computronium.ontology.update import (
    ParameterUpdate,
    RiemannianOrthogonalUpdate,
)

CONTEXT = 64
VAL_WINDOWS = 1024
EVAL_INTERVAL_S = 20.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("benchmark_results")
VOCAB = 65

# --- configs of record (smoke variants in parentheses) -----------------
# capacity-matched pairs: transformer 2·V·d + 12·d²·L  vs
# mlp V·C·w + (L−1)·w² + V·w
TRANSFORMER_CFG = {"d": 320, "n_layer": 6, "n_head": 5, "ctx": 128}  # 7.41M
MLP_CFG = {"hidden": (816,) * 7, "ctx": 64}  # 7.43M (ratio 1.003)
TRANSFORMER_SMOKE = {"d": 64, "n_layer": 3, "n_head": 4, "ctx": 32}  # ~156k
MLP_SMOKE = {"hidden": (64,) * 4, "ctx": 32}  # ~150k (ratio 0.96)

# LM-tuned lrs (smoke sweeps 2026-09-05). Keys are arm names with '*'
# wildcards; pepita needs gentler steps (0.02 exploded; 0.002 the MLP
# stable value; 5e-4 the transformer stable value).
# NOTE: specific patterns BEFORE wildcards — _lr_for returns the first
# match, and a trailing wildcard would shadow an arm-specific override.
LR: dict[str, float] = {
    "transformer/pepita/muon": 5e-4,
    "mlp/pepita/muon": 5e-4,
    "*/ff_hybrid/muon": 0.02,
    "mlp/epc_thermo/muon": 0.01,
    "*/bp/adam": 1e-3,
    "*/bp/muon": 0.01,
    "*/bp/ortho_adam": 1e-3,
    "*/ff/muon": 0.01,
    "*/ff/ortho_adam": 1e-3,
}


def _lr_for(arm: str) -> float:
    for pattern, lr in LR.items():
        parts = pattern.split("/")
        key = arm.split("/")
        if len(parts) != len(key):
            continue
        if all(p == k for p, k in zip(parts, key, strict=True) if p != "*"):
            return lr
    raise KeyError(arm)


# ---------------------------------------------------------------- data


def load_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    """Both splits encoded through ONE train-built vocab.

    get_lm_dataset builds a CharDataset per split — each split's ids are
    a different cipher (found by the first smoke run: char sets differ
    across splits). Fetch raw texts, build the vocab on train, encode
    both."""
    train_ds = get_lm_dataset("tiny_shakespeare", seq_len=64, split="train")
    val_ds = get_lm_dataset("tiny_shakespeare", seq_len=64, split="validation")
    stoi = {c: i for i, c in enumerate(sorted(set(train_ds.idx_to_char.values())))}
    val_raw = val_ds.decode(val_ds.data)  # unknown chars KeyError loudly
    return train_ds.data.long(), torch.tensor([stoi[c] for c in val_raw])


def _val_sets(
    val_t: torch.Tensor, vctx: int
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]
]:
    """Fixed val batches (transformer dense, mlp window) shared by all arms."""
    gen = torch.Generator().manual_seed(0)
    vidx = torch.randint(0, len(val_t) - vctx - 1, (VAL_WINDOWS,), generator=gen)
    offs = torch.arange(vctx + 1)
    vwin = val_t[vidx.unsqueeze(1) + offs]
    t_val = [(w[:, :-1], w[:, 1:].reshape(-1)) for w in vwin.split(256)]
    eye = torch.eye(VOCAB)
    m_val = []
    for w in vwin.split(256):
        x = eye[w[:, :-1]].reshape(w.size(0), -1)
        m_val.append((x, w[:, -1]))
    return t_val, m_val


# ---------------------------------------------------------------- arms


def _build(geom: str, credit: str, update: str, cfg: dict):
    if geom == "transformer":
        geometry = geometry_from_config(
            GeometryConfig.causal_transformer(
                vocab_size=VOCAB,
                d_model=cfg["d"],
                n_layers=cfg["n_layer"],
                n_heads=cfg["n_head"],
                seq_len=cfg["ctx"],
            )
        )
    else:
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=cfg["ctx"] * VOCAB,
                output_dim=VOCAB,
                hidden_dims=cfg["hidden"],
            )
        )
    if credit == "bp":
        credit_obj = BackpropCredit()
    elif credit == "ff_hybrid":
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff", readout_error=True
            )
        )
    elif credit == "thermo":
        credit_obj = ThermodynamicContrast()
    else:
        objective: Literal["ff", "pepita"] = "ff" if credit == "ff" else "pepita"
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective=objective
            )
        )
    lr = _lr_for(f"{geom}/{credit}/{update}")
    updates: dict[str, Callable[[], ParameterUpdate]] = {
        "adam": lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=lr)),
        "muon": lambda: RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr, momentum=0.9)
        ),
        "ortho_adam": lambda: OrthoAdamUpdate(
            ParameterUpdateConfig.ortho_adam(step_size=LR["*/bp/adam"], ortho_lr=lr)
        ),
    }
    dynamics = (
        ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(max_steps=10, step_size=0.1)
        )
        if credit == "thermo"
        else InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=DEVICE)),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit_obj,
        update=updates[update](),
    )


def _eval(system, val: list[tuple[torch.Tensor, torch.Tensor]], geom: str) -> dict:
    tot = n = 0
    with torch.no_grad():
        for x, y in val:
            state = system.dynamics.settle(
                SystemState(x=x.to(DEVICE)), system.geometry, system.substrate, None
            )
            acts = state.activations
            logits = acts[-1] if isinstance(acts, list) else acts
            loss = nn.functional.cross_entropy(logits, y.to(DEVICE), reduction="sum")
            tot += loss.item()
            n += y.numel()
    avg = tot / n
    return {"val_loss": round(avg, 4), "val_ppl": round(math.exp(min(avg, 20)), 2)}


def run_arm(
    geom: str,
    credit: str,
    update: str,
    tokens: torch.Tensor,
    val: list[tuple[torch.Tensor, torch.Tensor]],
    minutes: float,
    seed: int,
    cfg: dict,
    batch: int = 32,
) -> dict:
    torch.manual_seed(seed)
    system = _build(geom, credit, update, cfg)
    system.geometry.to(DEVICE)  # type: ignore[attr-defined]
    n_params = sum(p.numel() for p in system.geometry.params.values())
    gen = torch.Generator().manual_seed(seed + 1)
    ctx = cfg["ctx"]
    curve: list[dict] = []
    t0 = time.time()
    step = tokens_seen = 0
    while time.time() - t0 < minutes * 60:
        idx = torch.randint(0, len(tokens) - ctx - 1, (batch,), generator=gen)
        win = tokens[idx.unsqueeze(1) + torch.arange(ctx + 1)]
        if geom == "transformer":
            x, y = win[:, :-1].to(DEVICE), win[:, 1:].reshape(-1).to(DEVICE)
        else:
            x = (
                torch.nn.functional
                .one_hot(win[:, :-1], VOCAB)
                .float()
                .reshape(batch, ctx * VOCAB)
                .to(DEVICE)
            )
            y = win[:, -1].to(DEVICE)
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
        tokens_seen += y.numel()
        if time.time() - t0 > EVAL_INTERVAL_S * (len(curve) + 1):
            curve.append({
                "t": round(time.time() - t0, 1),
                "train_loss": round(metrics["loss"], 4),
                **_eval(system, val, geom),
            })
    wall = time.time() - t0
    return {
        "arm": f"{geom}/{credit}/{update}",
        "params": n_params,
        "steps": step,
        "tokens": tokens_seen,
        "chars_per_s": round(tokens_seen / wall, 1),
        "curve": curve,
        "final": curve[-1] if curve else _eval(system, val, geom),
    }


# ---------------------------------------------------------------- main


CELLS = [
    ("transformer", "bp", "adam"),
    ("transformer", "bp", "muon"),
    ("transformer", "ff_hybrid", "muon"),
    ("transformer", "pepita", "muon"),
    ("mlp", "bp", "adam"),
    ("mlp", "ff_hybrid", "muon"),
    ("mlp", "epc_thermo", "muon"),
    ("mlp", "ff", "muon"),
    ("mlp", "ff", "ortho_adam"),
    ("mlp", "pepita", "muon"),
]


def _gated_cells(arms: str, tcfg: dict, mcfg: dict) -> list[tuple[str, str, str]]:
    """Capacity gate + arm selection."""
    t_params = _param_count("transformer", tcfg)
    m_params = _param_count("mlp", mcfg)
    ratio = max(t_params, m_params) / min(t_params, m_params)
    print(
        f"capacity match: transformer {t_params:,} vs mlp {m_params:,} "
        f"params (ratio {ratio:.3f}, gate < 1.05)"
    )
    if ratio >= 1.05:
        raise RuntimeError
    cells = CELLS
    if arms:
        keep = set(arms.split(","))
        cells = [c for c in cells if "/".join(c) in keep]
    return cells


def _param_count(geom: str, cfg: dict) -> int:
    if geom == "transformer":
        g = geometry_from_config(
            GeometryConfig.causal_transformer(
                vocab_size=VOCAB,
                d_model=cfg["d"],
                n_layers=cfg["n_layer"],
                n_heads=cfg["n_head"],
                seq_len=cfg["ctx"],
            )
        )
    else:
        g = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=cfg["ctx"] * VOCAB,
                output_dim=VOCAB,
                hidden_dims=cfg["hidden"],
            )
        )
    return sum(p.numel() for p in g.params.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", type=str, default="")
    args = parser.parse_args(argv)
    minutes = args.minutes or (1.0 if args.smoke else 60.0)

    train_t, val_t = load_tokens()
    tcfg = TRANSFORMER_SMOKE if args.smoke else TRANSFORMER_CFG
    mcfg = MLP_SMOKE if args.smoke else MLP_CFG
    t_val, m_val = _val_sets(val_t, max(tcfg["ctx"], mcfg["ctx"]))

    geoms: dict[str, tuple[dict, list]] = {
        "transformer": (tcfg, t_val),
        "mlp": (mcfg, m_val),
    }

    cells = _gated_cells(args.arms, tcfg, mcfg)

    results = _run_cells(geoms, cells, minutes, args.seed, train_t)

    _print_table(results)
    return 0


def _run_cells(
    geoms: dict[str, tuple[dict, list]],
    cells: list[tuple[str, str, str]],
    minutes: float,
    seed: int,
    train_t: torch.Tensor,
) -> list[dict]:
    results: list[dict] = []
    for geom, credit, update in cells:
        cfg, val = geoms[geom]
        arm = f"{geom}/{credit}/{update}"
        print(f"=== {arm} ({minutes} min, {DEVICE}) ===", flush=True)
        r = run_arm(geom, credit, update, train_t, val, minutes, seed, cfg)
        results.append(r)
        print(
            f"  params {r['params']:,}  steps {r['steps']}  "
            f"tokens {r['tokens']:,} ({r['chars_per_s']:.0f}/s)  "
            f"final val_loss {r['final']['val_loss']:.4f}  "
            f"val_ppl {r['final']['val_ppl']:.2f}",
            flush=True,
        )
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "lm_comparison.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def _print_table(results: list[dict]) -> None:
    print("\n=== PRELIMINARY TABLE (val ppl, lower is better) ===")
    for r in sorted(results, key=lambda r: r["final"]["val_ppl"]):
        print(
            f"  {r['arm']:>28}  ppl {r['final']['val_ppl']:8.2f}  "
            f"params {r['params']:>10,}  steps {r['steps']}"
        )


if TYPE_CHECKING:
    from collections.abc import Callable


if __name__ == "__main__":
    raise SystemExit(main())
