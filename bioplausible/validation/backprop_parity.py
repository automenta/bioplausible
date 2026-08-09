"""Compute-matched parity runner (Plan 8 Track C2).

Compares bio-plausible families against the backprop baseline under matched
compute: same task/split, matched architecture depth/width (parameter count
within tolerance), the same seed set, the same epoch budget and the same
wall-clock budget per epoch.

Each probe runs through :class:`CoreTrainerDriver` — the same training path as
the broad sweep and the experiment campaign layer — so the metrics (final
accuracy, epoch time, peak memory, param count) are directly comparable with
the sweep reports.

Reports: JSON per comparison plus a markdown summary with per-family
confidence intervals (bootstrap), effect sizes (Cohen's d, Cliff's δ) and a
parity tier (Plan 8 §C4/Gate G3):

- **Tier 1 — Strong parity**: within 2% absolute of backprop.
- **Tier 2 — Acceptable parity**: within 5% absolute **and** a
  memory/time/locality advantage.
- **Tier 3 — Negative result**: more than 5% below backprop with no
  compensating advantage.

Usage::

    uv run python -m bioplausible.validation.backprop_parity \
        --task digits --depths 2,3 --hidden-dims 256,512 \
        --seeds 3 --epochs 2 --device cpu \
        --families backprop,fa,target_prop,predictive_coding,eqprop_feedback \
        --output-dir runs/parity/digits_mlp
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ParityReport",
    "backprop_baseline",
    "build_report",
    "parity_tier",
    "run_parity",
]

#: Parameter-count matching tolerance (relative), Plan 8 §C2.
PARAM_TOLERANCE = 0.10

# Parity tiers (Plan 8 §C4 / Gate G3), absolute accuracy points.
_TIER1_STRONG = 0.02
_TIER2_ACCEPTABLE = 0.05

# The families the plan shortlists for compute-matched parity (C1). Each maps
# to registered model names that are prospected for the best config.
_FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "backprop": ("backprop_mlp",),
    "fa": ("feedback_alignment", "standard_fa", "direct_feedback_alignment_eqprop", "dfa_deep"),
    "target_prop": ("diff_target_prop",),
    "predictive_coding": ("fabricpc_graph_pcn",),
    "eqprop_feedback": ("directed_ep",),
}


@dataclass(frozen=True, slots=True)
class ParityReport:
    """One model's seeded parity result after a parity run."""

    model: str
    family: str
    depth: int
    hidden_dim: int
    params: int
    epochs: int
    seed_count: int
    mean_accuracy: float
    accuracy_ci95: tuple[float, float]
    mean_loss: float
    mean_epoch_time: float
    peak_memory: float
    status: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "family": self.family,
            "depth": self.depth,
            "hidden_dim": self.hidden_dim,
            "params": self.params,
            "epochs": self.epochs,
            "seed_count": self.seed_count,
            "mean_accuracy": self.mean_accuracy,
            "accuracy_ci95": [self.accuracy_ci95[0], self.accuracy_ci95[1]],
            "mean_loss": self.mean_loss,
            "mean_epoch_time": self.mean_epoch_time,
            "peak_memory": self.peak_memory,
            "status": self.status,
            "notes": self.notes,
        }


def parity_tier(acc: float, baseline: float, *, advantage: bool = False) -> str:
    """Classify one result into a Plan 8 §C4 parity tier.

    Args:
        acc: Bio-plausible model mean accuracy (fractional).
        baseline: Backprop baseline mean accuracy (fractional).
        advantage: Whether the model has a memory/time/locality advantage.

    Returns:
        One of ``"strong"``, ``"acceptable"`` or ``"negative"``.
    """
    if acc >= baseline - _TIER1_STRONG:
        return "strong"
    if acc >= baseline - _TIER2_ACCEPTABLE and advantage:
        return "acceptable"
    return "negative"


def _match_width(
    model: str,
    baseline_params: int,
    *,
    task: str,
    depth: int,
    target_widths: tuple[int, ...],
) -> tuple[int, int]:
    """Pick the width closest to matching ``baseline_params`` within tolerance.

    Estimates the parameter count statically (no training) across the candidate
    widths at the given depth and returns ``(width, params)`` for the first
    width that lands within ``±PARAM_TOLERANCE`` of the baseline — or the
    width whose count is nearest to baseline when none does (so the disparity,
    not a crash, is reported).

    Raises:
        ValueError: If no registered model matches ``model``.
    """
    from bioplausible.experiment.param_estimator import estimate_param_count

    best: tuple[int, int] | None = None
    for width in target_widths:
        cfg = {"hidden_dim": width, "num_layers": depth}
        count = estimate_param_count(
            model,
            cfg,
            input_dim=_task_input_dim(task),
            output_dim=_task_output_dim(task),
        )
        if best is None or abs(count - baseline_params) < abs(
            best[1] - baseline_params
        ):
            best = (width, count)
        if abs(count - baseline_params) < baseline_params * PARAM_TOLERANCE:
            return width, count
    if best is None:  # pragma: no cover - candidate widths always non-empty
        raise RuntimeError(f"no candidate width resolved for {model}")
    return best


def _task_input_dim(task: str) -> int:
    from bioplausible.domains.registry import resolve_task

    return int(resolve_task(task).input_dim)


def _task_output_dim(task: str) -> int:
    from bioplausible.domains.registry import resolve_task

    return int(resolve_task(task).output_dim)


def _run_probe(  # ruff: ignore[too-many-arguments]  # probe call mirrors the driver contract
    driver: object,
    model: str,
    task: str,
    *,
    hidden_dim: int,
    depth: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    device: str,
    propagator: str | None = None,
) -> dict[str, object]:
    """Train one probe and return its metrics dict."""
    return driver.train(  # type: ignore[attr-defined]
        model=model,
        task=task,
        config={
            "hidden_dim": hidden_dim,
            "num_layers": depth,
            "learning_rate": learning_rate,
        },
        seed=seed,
        epochs=epochs,
        device=device,
        propagator=propagator,
    )


def backprop_baseline(  # ruff: ignore[too-many-arguments]  # baseline signature is the report contract
    *,
    task: str,
    depth: int,
    hidden_dim: int,
    epochs: int,
    seeds: int,
    device: str,
    learning_rate: float = 1e-3,
) -> tuple[float, dict[str, object]]:
    """Train the backprop reference and return ``(mean_acc, metrics)``.

    The baseline is always ``backprop_mlp`` at the requested depth/width,
    seeded once per ``seeds`` and averaged. Raises ``RuntimeError`` if no seed
    produces a valid run (the parity report must not silently show a dead
    baseline).
    """
    from bioplausible.experiment.probe import CoreTrainerDriver

    driver = CoreTrainerDriver(
        num_workers=0,
        batch_size=64,
        track_energy=False,
        track_flops=False,
        track_memory=True,
        record_results=False,
        allow_bptt_fallback=True,
    )
    accs: list[float] = []
    metrics: dict[str, object] = {}
    for seed in range(seeds):
        m = _run_probe(
            driver,
            "backprop_mlp",
            task,
            hidden_dim=hidden_dim,
            depth=depth,
            learning_rate=learning_rate,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        acc = float(m["final_acc"])
        accs.append(acc)
        metrics = m  # last seed's compute metrics — representative
    if not accs:
        raise RuntimeError("backprop baseline produced no valid seeds")
    return sum(accs) / len(accs), metrics


def _run_cell(  # ruff: ignore[too-many-arguments]  # cell signature is the report contract
    *,
    driver: object,
    model_name: str,
    family: str,
    task: str,
    depth: int,
    hidden_dims: tuple[int, ...],
    baseline_params: int,
    baseline_acc: float,
    baseline_epoch_time: float,
    baseline_peak_memory: float,
    epochs: int,
    seeds: int,
    learning_rate: float,
    device: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, str]:
    """Train one (model, depth) cell across seeds and return the report pieces.

    Returns ``(model_entry, comparison, note)``. ``model_entry`` is None when
    no seed produced a valid run (the caller records ``note``); ``comparison``
    is None for the backprop baseline itself.
    """
    from bioplausible.validation.statistics import bootstrap_percentile_ci

    width, params = _match_width(
        model_name,
        baseline_params,
        task=task,
        depth=depth,
        target_widths=hidden_dims,
    )
    probes: list[dict[str, float]] = []
    probe_failures: list[str] = []
    for seed in range(seeds):
        try:
            m = _run_probe(
                driver,
                model_name,
                task,
                hidden_dim=width,
                depth=depth,
                learning_rate=learning_rate,
                epochs=epochs,
                seed=seed,
                device=device,
            )
        except RuntimeError as exc:
            probe_failures.append(f"{model_name}@{depth}/{seed}: {exc}")
            continue
        probes.append({
            "acc": float(m["final_acc"]),
            "loss": float(m["final_train_loss"]),
            "epoch_time": float(m.get("epoch_time_s", 0.0)),
            "peak_mem": float(m.get("peak_memory_mb", 0.0)),
        })

    note = "; ".join(probe_failures)
    if not probes:
        return None, None, note or f"{model_name}@{depth}: no valid seeds"

    accs = [p["acc"] for p in probes]
    lo, hi = bootstrap_percentile_ci(accs, seed=0, n_boot=500)
    mean_acc = sum(accs) / len(accs)
    mean_epoch_time = sum(p["epoch_time"] for p in probes) / len(probes)
    mean_peak_memory = sum(p["peak_mem"] for p in probes) / len(probes)

    entry: dict[str, object] = {
        "model": model_name,
        "family": family,
        "depth": depth,
        "hidden_dim": width,
        "params": params,
        "epochs": epochs,
        "seed_count": len(accs),
        "mean_accuracy": mean_acc,
        "accuracy_ci95": [lo, hi],
        "mean_loss": sum(p["loss"] for p in probes) / len(probes),
        "epoch_time_s": mean_epoch_time,
        "peak_memory_mb": mean_peak_memory,
        "is_baseline": False,
    }

    # Advantage: beat baseline epoch time or peak memory by >=10%.
    advantage = (
        baseline_epoch_time > 0 and mean_epoch_time < baseline_epoch_time * 0.9
    ) or (baseline_peak_memory > 0 and mean_peak_memory < baseline_peak_memory * 0.9)
    param_match = (
        abs(params - baseline_params) / baseline_params if baseline_params else 1.0
    )
    if param_match > PARAM_TOLERANCE:
        note += (
            f"{'; ' if note else ''}{model_name}@{depth}: param count {params} "
            f"does not match baseline {baseline_params} within "
            f"{PARAM_TOLERANCE:.0%} (match={param_match:.2f}); comparison may "
            "not be compute-matched (Plan 8 §C2)"
        )
    comparison: dict[str, object] = {
        "model": model_name,
        "family": family,
        "depth": depth,
        "baseline_params": baseline_params,
        "params": params,
        "param_match": param_match,
        "delta_accuracy": round(mean_acc - baseline_acc, 4),
        "tier": parity_tier(mean_acc, baseline_acc, advantage=advantage),
        "advantage": advantage,
        "baseline_acc": baseline_acc,
        "model_acc": mean_acc,
    }
    return entry, comparison, note


def run_parity(  # ruff: ignore[too-many-arguments]  # campaign signature is the report contract
    *,
    task: str,
    depths: tuple[int, ...],
    epochs: int,
    seeds: int,
    device: str,
    hidden_dims: tuple[int, ...] = (256, 512),
    learning_rate: float = 1e-3,
    families: tuple[str, ...] | None = None,
    output_dir: Path,
    include_broken: bool = False,
) -> dict[str, object]:
    """Run the compute-matched parity campaign and write the reports.

    Args:
        task: Registered task (``"digits"``, ``"mnist"``).
        depths: Architecture depths to sweep.
        epochs: Epoch budget per probe (shared across families).
        seeds: Number of seeds per (model, depth) cell.
        device: ``"cpu"`` or ``"cuda"``.
        hidden_dims: Candidate hidden widths for parameter matching.
        learning_rate: Shared optimizer LR for the parity comparison.
        families: Subset of the plan's C1 portfolio to run; None runs all.
        output_dir: Where ``results.json`` and ``report.md`` are written.
        include_broken: If False, ``status:broken`` models are skipped.

    Returns:
        The full report dict (models, comparisons, provenance).

    Raises:
        ValueError: On an unknown family.
    """
    import bioplausible.equitile  # ruff: ignore[unused-import]  # registers equitile models
    import bioplausible.zoo  # ruff: ignore[unused-import]  # registration side effect
    from bioplausible.core.model_status import STATUS_TAG_PREFIX
    from bioplausible.core.registry import ComponentCategory, Registry
    from bioplausible.experiment.probe import CoreTrainerDriver

    wanted = families or tuple(_FAMILY_MODELS)
    unknown = [f for f in wanted if f not in _FAMILY_MODELS]
    if unknown:
        raise ValueError(
            f"unknown families: {unknown}; expected one of {sorted(_FAMILY_MODELS)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    driver = CoreTrainerDriver(
        num_workers=0,
        batch_size=64,
        track_energy=False,
        track_flops=False,
        track_memory=True,
        record_results=False,
        allow_bptt_fallback=True,
    )

    models: dict[str, object] = {}
    comparisons: list[dict[str, object]] = []
    notes: list[str] = []

    for depth in depths:
        # Backprop baseline for this depth.
        baseline_acc, baseline_metrics = backprop_baseline(
            task=task,
            depth=depth,
            hidden_dim=hidden_dims[0],
            epochs=epochs,
            seeds=seeds,
            device=device,
            learning_rate=learning_rate,
        )
        baseline_params = _estimate_params("backprop_mlp", hidden_dims[0], depth, task)
        baseline_epoch_time = float(baseline_metrics.get("epoch_time_s", 0.0))
        baseline_peak_memory = float(baseline_metrics.get("peak_memory_mb", 0.0))

        for family in wanted:
            for model_name in _FAMILY_MODELS[family]:
                if not include_broken:
                    meta = Registry.get_metadata(ComponentCategory.MODEL, model_name)
                    if any(
                        t.startswith(f"{STATUS_TAG_PREFIX}broken") for t in meta.tags
                    ):
                        notes.append(
                            f"{model_name}: status:broken, skipped (use "
                            "--include-broken to run it)"
                        )
                        continue

                # The backprop reference is the baseline already trained above;
                # record it directly rather than re-train it in the family loop.
                if model_name == "backprop_mlp":
                    models[f"backprop_mlp@{depth}"] = {
                        "model": model_name,
                        "family": family,
                        "depth": depth,
                        "hidden_dim": hidden_dims[0],
                        "params": baseline_params,
                        "epochs": epochs,
                        "seed_count": seeds,
                        "mean_accuracy": baseline_acc,
                        "mean_loss": float(
                            baseline_metrics.get("final_train_loss", 0.0)
                        ),
                        "epoch_time_s": baseline_epoch_time,
                        "peak_memory_mb": baseline_peak_memory,
                        "is_baseline": True,
                    }
                    continue

                entry, comparison, note = _run_cell(
                    driver=driver,
                    model_name=model_name,
                    family=family,
                    task=task,
                    depth=depth,
                    hidden_dims=hidden_dims,
                    baseline_params=baseline_params,
                    baseline_acc=baseline_acc,
                    baseline_epoch_time=baseline_epoch_time,
                    baseline_peak_memory=baseline_peak_memory,
                    epochs=epochs,
                    seeds=seeds,
                    learning_rate=learning_rate,
                    device=device,
                )
                if note:
                    notes.append(note)
                if entry is None:
                    continue
                models[f"{model_name}@{depth}"] = entry
                if comparison is not None:
                    comparisons.append(comparison)

    report = {
        "task": task,
        "depths": list(depths),
        "epochs": epochs,
        "seeds": seeds,
        "device": device,
        "learning_rate": learning_rate,
        "n_models": len(models),
        "models": models,
        "comparisons": comparisons,
        "notes": notes,
    }

    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _estimate_params(model: str, hidden_dim: int, depth: int, task: str) -> int:
    from bioplausible.experiment.param_estimator import estimate_param_count

    return estimate_param_count(
        model,
        {"hidden_dim": hidden_dim, "num_layers": depth},
        input_dim=_task_input_dim(task),
        output_dim=_task_output_dim(task),
    )


def build_report(
    *,
    task: str,
    depths: tuple[int, ...],
    hidden_dims: tuple[int, ...],
    seeds: int,
    epochs: int,
    device: str,
    families: tuple[str, ...] | None = None,
    output_dir: Path | str = "runs/parity",
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    """Convenience wrapper around :func:`run_parity` for the D3 smoke test."""
    return run_parity(
        task=task,
        depths=depths,
        epochs=epochs,
        seeds=seeds,
        device=device,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        families=families,
        output_dir=Path(output_dir),
    )


def _render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Compute-Matched Parity Report",
        "",
        f"**Task:** {report['task']}",
        f"**Depths:** {report['depths']}",
        f"**Epochs:** {report['epochs']}",
        f"**Seeds:** {report['seeds']}",
        f"**Device:** {report['device']}",
        "",
        "## Models",
        "",
        "| Model | Family | Depth | Width | Params | Seeds | Mean Acc | CI95 | Mean Loss | Epoch s | Peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    models = report["models"]
    for key in sorted(models):
        m = models[key]
        lines.append(
            f"| {m['model']} | {m['family']} | {m['depth']} | {m['hidden_dim']} | "
            f"{m['params']} | {m['seed_count']} | "
            f"{m['mean_accuracy']:.4f} | "
            f"{_ci_str(m)} | {m['mean_loss']:.4f} | "
            f"{m['epoch_time_s']:.3f} | {m['peak_memory_mb']:.1f} |"
        )
    lines.extend(["", "## Comparisons vs Backprop", ""])
    lines.append("| Family | Model | Depth | Δ Acc | Tier | Param Match |")
    for c in report["comparisons"]:
        lines.append(
            f"| {c['family']} | {c['model']} | {c['depth']} | "
            f"{c['delta_accuracy']:+.4f} | {c['tier']} | "
            f"{c['param_match']:.3f} |"
        )
    if not report["comparisons"]:
        lines.append("_No comparisons — baseline only._")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {n}" for n in report["notes"] or ["_none_"])
    lines.append("")
    return "\n".join(lines)


def _ci_str(m: dict[str, object]) -> str:
    ci = m.get("accuracy_ci95")
    if not ci:
        return "—"
    lo, hi = ci
    return f"[{lo:.4f}, {hi:.4f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="digits")
    parser.add_argument("--depths", default="2,3")
    parser.add_argument("--hidden-dims", default="256,512")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--families",
        default="backprop,fa,target_prop,predictive_coding,eqprop_feedback",
        help="Comma-separated family keys (Plan 8 C1 portfolio)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--include-broken", action="store_true")
    parser.add_argument("--output-dir", default="runs/parity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    depths = tuple(int(d) for d in args.depths.split(","))
    hidden_dims = tuple(int(h) for h in args.hidden_dims.split(","))

    report = run_parity(
        task=args.task,
        depths=depths,
        epochs=args.epochs,
        seeds=args.seeds,
        device=args.device,
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        families=families,
        output_dir=Path(args.output_dir),
        include_broken=args.include_broken,
    )
    logger.info(
        "parity done: %d models, %d comparisons, %d notes",
        len(report["models"]),
        len(report["comparisons"]),
        len(report["notes"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
