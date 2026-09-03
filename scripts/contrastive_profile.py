"""Contrastive profiling script (Plan 8 Track B1).

Reveals whether deep EqProp layers receive meaningful nudged/free state
differences, or suffer from vanishing contrastive signal.

Usage::

    uv run python scripts/contrastive_profile.py \
        --model eqprop \
        --task digits \
        --num-layers 3 \
        --hidden-dim 256 \
        --epochs 1 \
        --device cpu

    uv run python scripts/contrastive_profile.py \
        --model directed_ep \
        --task digits \
        --num-layers 3 \
        --hidden-dim 256 \
        --epochs 1 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from computronium.core.presets import create_eqprop_mlp
from computronium.data.vision import get_vision_dataset
from computronium.models.native.research_native import create_native_directed_ep

logger = logging.getLogger(__name__)

__all__ = ["main", "profile_model", "write_report"]


def _git_sha() -> str:
    """Current git HEAD short hash, or ``"unknown"`` outside a git repo."""
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


def _env_provenance() -> dict[str, str]:
    """Environment fingerprint for the diagnostics record."""
    return {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _build_model(args: _ProfileArgs, input_dim: int, output_dim: int) -> object:
    """Build the profiled model via the live native factories."""
    if args.model_name == "directed_ep":
        return create_native_directed_ep(
            input_dim,
            args.hidden_dim,
            output_dim,
            num_layers=args.num_layers,
            beta=args.beta,
            settle_steps=20,
            lr=args.learning_rate,
            feedback_scale=args.feedback_gain
            if args.feedback_gain is not None
            else 0.01,
            device=args.device,
        )
    if args.model_name == "eqprop":
        return create_eqprop_mlp(
            input_dim,
            hidden_dims=(args.hidden_dim,) * args.num_layers,
            output_dim=output_dim,
            beta=args.beta,
            inference_steps=20,
            lr=args.learning_rate,
            device=args.device,
        )
    raise ValueError(
        f"Unknown profile model {args.model_name!r}; expected 'eqprop' or 'directed_ep'"
    )


@dataclass(frozen=True, slots=True)
class _ProfileArgs:
    """Bundled profile parameters (keeps ``profile_model`` lean)."""

    model_name: str
    task: str
    num_layers: int
    hidden_dim: int
    beta: float
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int
    device: str
    output_dir: Path
    feedback_gain: float | None = None
    w_rec_init: str | None = None
    w_rec_gain: float | None = None
    update_scale: float | None = None
    update_scale_by_depth: float | None = None


_PROFILE_STEPS = 10

# Gate G1 (vanishing contrastive signal) thresholds, per Plan 8 Gate G1.
_G1_MIN_DEPTH = 3
_G1_RATIO_THRESHOLD = 1e-3
_G1_MIN_OUTPUT_DELTA = 1e-10


def _collect_step_diagnostics(
    all_diagnostics: list[dict],
    result: dict,
    *,
    step: int,
    epoch: int,
    batch_idx: int,
    elapsed: float,
) -> None:
    """Append per-layer and global diagnostics for one training step."""
    for layer_diag in result["layer_diagnostics"]:
        entry = {"step": step, "epoch": epoch, "batch": batch_idx}
        entry.update(layer_diag)
        entry["time"] = elapsed
        all_diagnostics.append(entry)

    global_diag = result["global_diagnostics"]
    all_diagnostics.append({
        "step": step,
        "epoch": epoch,
        "batch": batch_idx,
        "type": "global",
        "output_state_delta_norm": global_diag["output_state_delta_norm"],
        "beta": global_diag["beta"],
        "loss": global_diag["loss"],
        "accuracy": global_diag["accuracy"],
        "free_converged": global_diag.get("free_converged"),
        "nudged_converged": global_diag.get("nudged_converged"),
        "free_settle_residual": global_diag.get("free_settle_residual"),
        "nudged_settle_residual": global_diag.get("nudged_settle_residual"),
        "free_steps_taken": global_diag.get("free_steps_taken"),
        "nudged_steps_taken": global_diag.get("nudged_steps_taken"),
        "time": elapsed,
    })


def _resolve_dims(train_data: object) -> tuple[int, int]:
    """Return ``(input_dim, output_dim)`` from a vision dataset."""
    x_sample, _ = train_data[0]  # type: ignore[index]
    input_dim = x_sample.numel()
    output_dim = int(max(y.item() for _, y in train_data)) + 1  # type: ignore[union-attr]
    return input_dim, output_dim


def _write_summary(summary: dict, model_dir: Path) -> None:
    """Write diagnostics JSON and markdown summary for a profile run."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(summary, model_dir / "summary.md")


def profile_model(args: _ProfileArgs) -> dict:  # ruff: ignore[complex-structure, too-many-locals]
    """Profile a single model and write diagnostics."""
    torch.manual_seed(args.seed)

    # Load data
    dataset = get_vision_dataset(args.task, flatten=True)
    train_data = dataset[0] if isinstance(dataset, tuple) else dataset
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    input_dim, output_dim = _resolve_dims(train_data)
    model = _build_model(args, input_dim, output_dim)

    all_diagnostics: list[dict] = []
    step = 0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            x, y = (t.to(args.device) for t in batch)
            if x.dtype != torch.float32:
                x = x.float()

            t0 = time.time()
            result = model.train_step(x, y)  # type: ignore[attr-defined]
            elapsed = time.time() - t0

            if isinstance(result, dict) and "layer_diagnostics" in result:
                _collect_step_diagnostics(
                    all_diagnostics,
                    result,
                    step=step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    elapsed=elapsed,
                )

            step += 1
            if step >= _PROFILE_STEPS:
                break
        if step >= _PROFILE_STEPS:
            break

    # Check Gate G1: Vanishing Signal
    g1_triggered = _check_gate_g1(all_diagnostics, args.num_layers)
    # Per-step early/output ratios: the depth-scaling evidence record.
    signal_ratios = _per_step_signal_ratios(all_diagnostics, args.num_layers)

    # Summary
    config_extras: dict[str, object] = {}
    if args.feedback_gain is not None:
        config_extras["feedback_gain"] = args.feedback_gain
    if args.w_rec_init is not None:
        config_extras["w_rec_init"] = args.w_rec_init
    if args.update_scale is not None:
        config_extras["update_scale"] = args.update_scale
    if args.update_scale_by_depth is not None:
        config_extras["update_scale_by_depth"] = args.update_scale_by_depth

    summary = {
        "model": args.model_name,
        "task": args.task,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "steps_profiled": step,
        "gate_g1_vanishing_signal": g1_triggered,
        "signal_ratios": signal_ratios,
        "diagnostics": all_diagnostics,
        "provenance": _env_provenance(),
        "config_extras": config_extras,
    }

    _write_summary(
        summary,
        args.output_dir / f"{args.model_name}_depth{args.num_layers}_{args.task}",
    )

    logger.info(
        "Profiled %s depth=%d task=%s g1=%s steps=%d",
        args.model_name,
        args.num_layers,
        args.task,
        g1_triggered,
        step,
    )

    return summary


def _check_gate_g1(diagnostics: list[dict], num_layers: int) -> bool:
    """Check Gate G1: Vanishing Contrastive Signal.

    Triggered if, for depth >= 3:
    early_layer_post_state_delta_norm / output_state_delta_norm < 1e-3
    and early_layer_grad_norm / output_layer_grad_norm < 1e-3
    for the majority of early training steps.
    """
    if num_layers < _G1_MIN_DEPTH:
        return False

    layer_diags = [d for d in diagnostics if "layer" in d]
    global_diags = [d for d in diagnostics if d.get("type") == "global"]

    if not layer_diags or not global_diags:
        return False

    # Group by step
    steps = {d["step"] for d in layer_diags}
    vanishing_count = 0
    total_count = 0

    for step in steps:
        step_layer_diags = [d for d in layer_diags if d["step"] == step]
        step_global = [d for d in global_diags if d["step"] == step]

        if not step_global:
            continue

        output_delta = step_global[0].get("output_state_delta_norm", 0)
        if output_delta < _G1_MIN_OUTPUT_DELTA:
            continue

        # Early layer = first hidden layer (layer 0)
        early_layer = step_layer_diags[0] if step_layer_diags else None
        if early_layer is None:
            continue

        early_post_delta = early_layer.get("post_state_delta_norm", 0)
        early_grad_norm = early_layer.get("weight_grad_norm", 0)

        # Output layer = last layer
        output_layer = step_layer_diags[-1] if step_layer_diags else None
        if output_layer is None:
            continue
        output_grad_norm = output_layer.get("weight_grad_norm", 0)

        total_count += 1
        delta_ratio = early_post_delta / output_delta if output_delta > 0 else 0
        grad_ratio = early_grad_norm / output_grad_norm if output_grad_norm > 0 else 0

        if delta_ratio < _G1_RATIO_THRESHOLD and grad_ratio < _G1_RATIO_THRESHOLD:
            vanishing_count += 1

    if total_count == 0:
        return False
    return vanishing_count > total_count / 2


def _per_step_signal_ratios(diagnostics: list[dict], num_layers: int) -> list[dict]:
    """Return per-step early/output signal ratios (the depth-scaling record).

    Each entry records ``step``, ``delta_ratio`` (early-layer post-state
    delta ÷ output delta) and ``grad_ratio`` (early-layer weight-grad norm ÷
    output-layer grad norm) — the quantities Gate G1 summarises into a single
    boolean. Reporting the ratios per step keeps the depth trend visible even
    when the binary gate does not fire (Plan 8 notes §2).
    """
    layer_diags = [d for d in diagnostics if "layer" in d]
    global_diags = [d for d in diagnostics if d.get("type") == "global"]
    if not layer_diags or not global_diags:
        return []
    steps = sorted({d["step"] for d in layer_diags})
    ratios: list[dict] = []
    for step in steps:
        step_layers = [d for d in layer_diags if d["step"] == step]
        step_global = [d for d in global_diags if d["step"] == step]
        if not step_layers or not step_global:
            continue
        output_delta = step_global[0].get("output_state_delta_norm", 0)
        early = step_layers[0]
        out_layer = step_layers[-1] if len(step_layers) > 1 else early
        delta_ratio = (
            early.get("post_state_delta_norm", 0) / output_delta
            if output_delta > 0
            else 0
        )
        out_grad = out_layer.get("weight_grad_norm", 0)
        grad_ratio = early.get("weight_grad_norm", 0) / out_grad if out_grad > 0 else 0
        ratios.append({
            "step": step,
            "delta_ratio": delta_ratio,
            "grad_ratio": grad_ratio,
            "n_layers": len(step_layers),
        })
    return ratios


def _slope_vs_depth(points: list[tuple[int, float]]) -> dict[str, float]:
    """Fit log(metric) vs depth by ordinary least squares.

    ``points`` is ``(depth, value)`` with ``value > 0`` for all entries. The
    fitted slope is the exponential decay rate of ``value`` with depth — the
    Plan 8 notes §2 recommended evidence for vanishing contrastive signal (a
    consistently negative slope), replacing a noise-sensitive binary threshold.

    Returns the slope, intercept, R², and the Pearson correlation of
    ``log(value)`` with ``depth``.
    """
    import math

    xs = [float(d) for d, _ in points]
    ys = [math.log(v) for _, v in points]
    n = len(xs)
    if n < 3:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "n": n}
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "n": n}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx
    intercept = y_mean - slope * x_mean
    # R²
    y_hat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


def write_report(summary: dict, path: Path) -> None:
    """Write a human-readable markdown summary."""
    prov = summary.get("provenance", {})
    lines = [
        f"# Contrastive Profile: {summary['model']} depth={summary['num_layers']}",
        "",
        f"**Task:** {summary['task']}",
        f"**Hidden dim:** {summary['hidden_dim']}",
        f"**Beta:** {summary['beta']}",
        f"**Learning rate:** {summary['learning_rate']}",
        f"**Seed:** {summary['seed']}",
        f"**Device:** {summary['device']}",
        f"**Steps profiled:** {summary['steps_profiled']}",
        f"**Gate G1 (Vanishing Signal):** {'TRIGGERED' if summary['gate_g1_vanishing_signal'] else 'not triggered'}",
        f"**git SHA:** {prov.get('git_sha', 'unknown')}",
        f"**python/torch:** {prov.get('python_version', 'n/a')} / {prov.get('torch_version', 'n/a')}",
        "",
        "## Per-Step Diagnostics",
        "",
        "| Step | Layer | Pre Δ Norm | Post Δ Norm | Grad Norm | Update Scale |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for d in summary["diagnostics"]:
        if "layer" in d:
            lines.append(
                f"| {d['step']} | {d['layer']} | "
                f"{d.get('pre_state_delta_norm', 0):.6f} | "
                f"{d.get('post_state_delta_norm', 0):.6f} | "
                f"{d.get('weight_grad_norm', 0):.6f} | "
                f"{d.get('update_scale', 1.0):.4f} |"
            )

    lines.extend([
        "",
        "## Signal Ratios (early layer ÷ output)",
        "",
        "| Step | Delta Ratio | Grad Ratio |",
        "|---:|---:|---:|",
    ])

    for r in summary.get("signal_ratios", []):
        lines.append(
            f"| {r['step']} | {r['delta_ratio']:.4g} | {r['grad_ratio']:.4g} |"
        )

    lines.extend([
        "",
        "## Global Diagnostics",
        "",
        "| Step | Output Δ Norm | Beta | Loss | Accuracy | Free Conv | Nudge Conv | Free Residual | Nudge Residual |",
        "|---:|---:|---:|---:|---:|:--:|:--:|---:|---:|",
    ])

    for d in summary["diagnostics"]:
        if d.get("type") == "global":
            lines.append(
                f"| {d['step']} | "
                f"{d.get('output_state_delta_norm', 0):.6f} | "
                f"{d.get('beta', 0):.4f} | "
                f"{d.get('loss', 0):.4f} | "
                f"{d.get('accuracy', 0):.4f} | "
                f"{'Y' if d.get('free_converged') else 'N'} | "
                f"{'Y' if d.get('nudged_converged') else 'N'} | "
                f"{d.get('free_settle_residual', 0):.4g} | "
                f"{d.get('nudged_settle_residual', 0):.4g} |"
            )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _depth_scale_analysis(depth_summaries: list[dict]) -> dict:
    """Fit log(early-layer delta / output delta) vs depth across profile runs.

    ``depth_summaries`` are the per-run summary dicts (one per depth). For each
    run we take the mean ``delta_ratio`` from ``signal_ratios``; the fitted OLS
    slope vs depth is the exponential decay rate of contrastive signal — the
    Plan 8 notes §2 evidence, more robust than a single-depth binary gate.

    Returns the fitted slope/intercept/R² and the per-depth mean ratios.
    """
    points: list[tuple[int, float]] = []
    per_depth: list[dict] = []
    for s in depth_summaries:
        ratios = [r["delta_ratio"] for r in s.get("signal_ratios", [])]
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        if mean_ratio > 0:
            points.append((int(s["num_layers"]), mean_ratio))
        per_depth.append({
            "depth": int(s["num_layers"]),
            "mean_delta_ratio": mean_ratio,
            "g1": bool(s.get("gate_g1_vanishing_signal", False)),
            "n_steps": len(ratios),
        })
    fit = _slope_vs_depth(points)
    return {"per_depth": per_depth, "fit": fit}


def analyze_depths(output_dir: Path) -> None:
    """Aggregate all depth runs under ``output_dir`` into a depth-scaling report.

    Recursively scans ``output_dir`` for ``diagnostics.json`` summaries (the
    profiler nests each run under a timestamped subfolder), fits
    ``log(delta_ratio)`` vs depth, and writes
    ``output_dir/depth_scale_analysis.md``.
    """
    diag_paths = list(output_dir.rglob("diagnostics.json"))
    summaries: list[dict] = []
    for diag_path in diag_paths:
        summaries.append(json.loads(diag_path.read_text(encoding="utf-8")))

    keys = {(s.get("model"), s.get("task")) for s in summaries}
    if len(keys) > 1:
        logger.warning(
            "depth-scale analysis found runs from multiple (model, task) "
            "combinations under %s: %s — run it per model/task directory.",
            output_dir,
            sorted(keys),
        )
        return
    if len({s["num_layers"] for s in summaries}) < 3:
        logger.warning(
            "depth-scale analysis needs >=3 distinct depth runs under %s, found %d",
            output_dir,
            len({s["num_layers"] for s in summaries}),
        )
        return

    analysis = _depth_scale_analysis(summaries)
    fit = analysis["fit"]
    lines = [
        "# Depth-Scale Analysis (Plan 8 notes §2)",
        "",
        f"**Directory:** `{output_dir}`",
        f"**Fitted slope (log early/output delta per depth):** {fit['slope']:.4f}",
        f"**Intercept:** {fit['intercept']:.4f}",
        f"**R²:** {fit['r2']:.4f}",
        f"**N depths:** {fit['n']}",
        "",
        "A **consistently negative slope** is the signature of exponential "
        "contrastive-signal decay with depth (vanishing signal). A slope near "
        "zero or positive means early layers keep a meaningful nudged/free "
        "difference as depth grows.",
        "",
        "| Depth | Mean Delta Ratio | G1 | Steps |",
        "|---:|---:|:--:|---:|",
    ]
    for d in analysis["per_depth"]:
        lines.append(
            f"| {d['depth']} | "
            f"{d['mean_delta_ratio']:.4g} | "
            f"{'Y' if d['g1'] else 'N'} | "
            f"{d['n_steps']} |"
        )
    out = output_dir / "depth_scale_analysis.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote depth-scale analysis to %s", out)


def main() -> None:
    """Entry point for the contrastive profiling script."""
    parser = argparse.ArgumentParser(
        description="Contrastive profiling for EqProp models (Plan 8 Track B1)"
    )
    sub = parser.add_subparsers(dest="command")
    analyze = sub.add_parser(
        "analyze-depths",
        help="Aggregate existing profile runs into a depth-scaling report",
    )
    analyze.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--task", type=str, default="digits", choices=["digits", "mnist"]
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of hidden layers"
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--beta", type=float, default=0.1, help="Nudging strength")
    parser.add_argument(
        "--learning-rate", type=float, default=0.05, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/contrastive_profile",
        help="Output directory",
    )
    parser.add_argument(
        "--feedback-gain",
        type=float,
        default=None,
        help="Feedback gain for DirectedEP (null arm: 0.0)",
    )
    parser.add_argument(
        "--w-rec-init",
        type=str,
        default=None,
        choices=["zero", "xavier"],
        help="Recurrent weight initialization",
    )
    parser.add_argument(
        "--w-rec-gain",
        type=float,
        default=None,
        help="Gain for xavier recurrent init",
    )
    parser.add_argument(
        "--update-scale",
        type=float,
        default=None,
        help="Global update scale multiplier",
    )
    parser.add_argument(
        "--update-scale-by-depth",
        type=float,
        default=None,
        help="Geometric update scale factor per depth",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "analyze-depths":
        analyze_depths(Path(args.output_dir))
        return

    if args.model is None:
        parser.error("the following arguments are required: --model")

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_args = _ProfileArgs(
        model_name=args.model,
        task=args.task,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        feedback_gain=args.feedback_gain,
        w_rec_init=args.w_rec_init,
        w_rec_gain=args.w_rec_gain,
        update_scale=args.update_scale,
        update_scale_by_depth=args.update_scale_by_depth,
    )
    profile_model(profile_args)


if __name__ == "__main__":
    main()
