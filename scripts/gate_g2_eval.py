#!/usr/bin/env python
"""Gate G2 evaluation — feedback salvage test (Plan 8 Track B3).

Pre-registered protocol (Session 15.4):
- model: directed_ep with feedback_gain=0.5, w_rec_init=xavier
- task: digits, depth >= 3, 5 epochs, 3 seeds
- threshold: > 50% accuracy on digits with depth >= 3 after 5 epochs
- AND diagnostics show non-zero early-layer state deltas or gradient norms

Also runs vanilla eqprop as a negative control.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from computronium.core.presets import create_eqprop_mlp
from computronium.data.vision import get_vision_dataset
from computronium.models.native.research_native import create_native_directed_ep

if TYPE_CHECKING:
    from computronium.ontology import System

logger = logging.getLogger(__name__)

__all__ = ["evaluate_g2", "main"]


@dataclass(frozen=True, slots=True)
class _G2Config:
    model_name: str
    depth: int
    hidden_dim: int
    beta: float
    lr: float
    epochs: int
    batch_size: int
    seed: int
    device: str
    feedback_gain: float | None
    w_rec_init: str | None


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _build_model(cfg: _G2Config, input_dim: int, output_dim: int) -> System:
    """Build the pre-registered G2 model via the live native factories."""
    if cfg.model_name == "directed_ep":
        return create_native_directed_ep(
            input_dim,
            cfg.hidden_dim,
            output_dim,
            num_layers=cfg.depth,
            beta=cfg.beta,
            settle_steps=20,
            lr=cfg.lr,
            feedback_scale=cfg.feedback_gain if cfg.feedback_gain is not None else 0.01,
            device=cfg.device,
        )
    if cfg.model_name == "eqprop":
        return create_eqprop_mlp(
            input_dim,
            hidden_dims=(cfg.hidden_dim,) * cfg.depth,
            output_dim=output_dim,
            beta=cfg.beta,
            inference_steps=20,
            lr=cfg.lr,
            device=cfg.device,
        )
    raise ValueError(
        f"Unknown G2 model {cfg.model_name!r}; expected 'directed_ep' or 'eqprop'"
    )


def _resolve_dims(train_data: object) -> tuple[int, int]:
    x_sample, _ = train_data[0]  # type: ignore[index]
    input_dim = x_sample.numel()
    output_dim = int(max(y.item() for _, y in train_data)) + 1  # type: ignore[union-attr]
    return input_dim, output_dim


def _evaluate(
    train_data: object,
    model: System,
    cfg: _G2Config,
) -> dict[str, object]:
    """Train for cfg.epochs and return final accuracy + last-step diagnostics."""
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    step = 0
    last_diagnostics: dict[str, object] = {}
    final_acc = 0.0
    final_loss = 0.0

    for epoch in range(cfg.epochs):
        epoch_accs: list[float] = []
        epoch_losses: list[float] = []
        for batch in train_loader:
            x, y = (t.to(cfg.device) for t in batch)
            if x.dtype != torch.float32:
                x = x.float()
            result = model.train_step(x, y)
            if isinstance(result, dict):
                final_loss = float(result.get("loss", 0.0))
                final_acc = float(result.get("accuracy", 0.0))
                epoch_accs.append(final_acc)
                epoch_losses.append(final_loss)
                if "layer_diagnostics" in result:
                    last_diagnostics = {
                        "epoch": epoch,
                        "step": step,
                        "layer_diagnostics": result["layer_diagnostics"],
                        "global_diagnostics": result["global_diagnostics"],
                    }
            step += 1
        mean_acc = sum(epoch_accs) / len(epoch_accs) if epoch_accs else 0.0
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logger.info(
            "  %s depth=%d seed=%d epoch=%d/%d mean_acc=%.4f mean_loss=%.4f",
            cfg.model_name,
            cfg.depth,
            cfg.seed,
            epoch + 1,
            cfg.epochs,
            mean_acc,
            mean_loss,
        )

    return {
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "last_epoch_mean_accuracy": mean_acc,
        "last_epoch_mean_loss": mean_loss,
        "last_diagnostics": last_diagnostics,
    }


def evaluate_g2(cfg: _G2Config, output_dir: Path) -> dict:  # ruff: ignore[too-many-locals]
    """Run one G2 evaluation and write results."""
    torch.manual_seed(cfg.seed)

    dataset = get_vision_dataset(
        (cfg.model_name and "digits") or "digits", flatten=True
    )
    train_data = dataset[0] if isinstance(dataset, tuple) else dataset
    input_dim, output_dim = _resolve_dims(train_data)
    model = _build_model(cfg, input_dim, output_dim)

    t0 = time.time()
    result = _evaluate(train_data, model, cfg)
    elapsed = time.time() - t0

    # Check diagnostics for non-zero early-layer signal
    diag = result.get("last_diagnostics", {})
    layer_diags = diag.get("layer_diagnostics", [])
    early_layer = layer_diags[0] if layer_diags else {}
    early_post_delta = float(early_layer.get("post_state_delta_norm", 0))
    early_grad_norm = float(early_layer.get("weight_grad_norm", 0))
    global_diag = diag.get("global_diagnostics", {})
    output_delta = float(global_diag.get("output_state_delta_norm", 0))

    summary = {
        "model": cfg.model_name,
        "depth": cfg.depth,
        "hidden_dim": cfg.hidden_dim,
        "beta": cfg.beta,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "device": cfg.device,
        "feedback_gain": cfg.feedback_gain,
        "w_rec_init": cfg.w_rec_init,
        "final_accuracy": result["final_accuracy"],
        "final_loss": result["final_loss"],
        "last_epoch_mean_accuracy": result["last_epoch_mean_accuracy"],
        "last_epoch_mean_loss": result["last_epoch_mean_loss"],
        "early_post_delta_norm": early_post_delta,
        "early_grad_norm": early_grad_norm,
        "output_delta_norm": output_delta,
        "early_signal_nonzero": early_post_delta > 1e-6 or early_grad_norm > 1e-6,
        "elapsed_seconds": elapsed,
        "provenance": {
            "git_sha": _git_sha(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{cfg.model_name}_depth{cfg.depth}_seed{cfg.seed}"
    with (output_dir / f"{fname}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "G2: %s depth=%d seed=%d acc=%.4f early_delta=%.6f early_grad=%.6f",
        cfg.model_name,
        cfg.depth,
        cfg.seed,
        summary["final_accuracy"],
        early_post_delta,
        early_grad_norm,
    )
    return summary


def main() -> None:  # ruff: ignore[too-many-locals]
    """Run G2 evaluation per pre-registered protocol."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path("runs/gate_g2") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-registered G2 config
    depths = [3, 4]
    seeds = [0, 1, 2]
    hidden_dim = 256
    beta = 0.1
    lr = 0.05
    epochs = 5
    batch_size = 128
    device = "cuda"

    all_summaries = []

    # Arm 1: directed_ep with feedback (the salvage candidate)
    for depth in depths:
        for seed in seeds:
            cfg = _G2Config(
                model_name="directed_ep",
                depth=depth,
                hidden_dim=hidden_dim,
                beta=beta,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
                feedback_gain=0.5,
                w_rec_init="xavier",
            )
            logger.info("=== G2: directed_ep depth=%d seed=%d ===", depth, seed)
            s = evaluate_g2(cfg, output_dir)
            all_summaries.append(s)

    # Arm 2: vanilla eqprop (negative control)
    for depth in depths:
        for seed in seeds:
            cfg = _G2Config(
                model_name="eqprop",
                depth=depth,
                hidden_dim=hidden_dim,
                beta=beta,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
                feedback_gain=None,
                w_rec_init="xavier",
            )
            logger.info("=== G2 control: eqprop depth=%d seed=%d ===", depth, seed)
            s = evaluate_g2(cfg, output_dir)
            all_summaries.append(s)

    # Gate G2 verdict
    dep_results = [s for s in all_summaries if s["model"] == "directed_ep"]
    eq_results = [s for s in all_summaries if s["model"] == "eqprop"]

    dep_depth3 = [s for s in dep_results if s["depth"] >= 3]
    dep_pass = len(dep_depth3) >= 3  # at least 3 seeds at depth >= 3
    if dep_pass:
        mean_acc = sum(s["final_accuracy"] for s in dep_depth3) / len(dep_depth3)
        nonzero_signal = all(s["early_signal_nonzero"] for s in dep_depth3)
        g2_pass = mean_acc > 0.50 and nonzero_signal
    else:
        mean_acc = 0.0
        nonzero_signal = False
        g2_pass = False

    eq_depth3 = [s for s in eq_results if s["depth"] >= 3]
    eq_mean_acc = (
        sum(s["final_accuracy"] for s in eq_depth3) / len(eq_depth3)
        if eq_depth3
        else 0.0
    )

    verdict = {
        "g2_pass": g2_pass,
        "directed_ep_mean_acc_depth3plus": mean_acc,
        "directed_ep_nonzero_early_signal": nonzero_signal,
        "eqprop_mean_acc_depth3plus": eq_mean_acc,
        "threshold": 0.50,
        "n_seeds": len(dep_depth3),
        "verdict_text": (
            "G2 PASSED: directed_ep reaches >50% accuracy at depth>=3 with "
            "non-zero early-layer signal. Feedback EqProp is salvaged."
            if g2_pass
            else "G2 FAILED: directed_ep does not meet the >50% threshold or "
            "early-layer signal is zero. Deep EqProp salvage unsuccessful."
        ),
    }

    with (output_dir / "g2_verdict.json").open("w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    print("\n" + "=" * 72)
    print("GATE G2 VERDICT")
    print("=" * 72)
    print(f"directed_ep mean acc (depth>=3, {len(dep_depth3)} seeds): {mean_acc:.4f}")
    print(f"  threshold: > 0.50  →  {'PASS' if mean_acc > 0.50 else 'FAIL'}")
    print(f"  early signal non-zero: {nonzero_signal}")
    print(f"  G2 overall: {'PASS' if g2_pass else 'FAIL'}")
    print(f"eqprop (control) mean acc (depth>=3): {eq_mean_acc:.4f}")
    print(f"\n{verdict['verdict_text']}")
    print(f"\nResults in: {output_dir}")


if __name__ == "__main__":
    main()
