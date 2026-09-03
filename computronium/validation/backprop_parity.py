"""Compute-matched parity runner (Plan 8 Track C2).

Compares bio-plausible families against the backprop baseline under three
contracts (Plan 8 §15.4 pre-registration):

1. **Width-matched (Secondary)** — the bio model's width is selected from
   ``hidden_dims`` to land closest to the baseline's param count. Param
   mismatch is reported loudly. This is the legacy §C2 contract.
2. **Capacity-controlled (Tertiary)** — backprop is width-searched up to the
   bio model's param count. If the bio model still wins param-matched, the
   bio claim strengthens; if backprop closes the gap, the bio win was
   capacity-confounded. This is the §15.4 "more worth than any other cell" arm.
3. **Compute-matched (Primary)** — uses the same probes as the width-matched
   arm (same epoch budget, same wall-clock cap, same seeds) but reports
   forward+backward FLOPs so the settling-cost discrepancy (PC/eqprop settle
   FLOPs vs backprop's single forward+backward) is visible. Tiers are
   computed on this contract.

Per §15.4 the tier classification lives on the **Primary** contract; the other
two contracts are reported as additional comparison rows so the reader can
reconcile accuracy wins against compute and capacity.

Each probe runs through :class:`CoreTrainerDriver` — the same training path as
the broad sweep and the experiment campaign layer — so the metrics (final
accuracy, epoch time, peak memory, FLOPs, param count) are directly
comparable with the sweep reports.

Reports: JSON per comparison plus a markdown summary with per-family
confidence intervals (bootstrap), effect sizes (Cohen's d, Cliff's δ),
distribution-free p-value (permutation test) and a parity tier (Plan 8
§C4/Gate G3):

- **Tier 1 — Strong parity**: within 2% absolute of backprop.
- **Tier 2 — Acceptable parity**: within 5% absolute **and** a
  memory/time/locality advantage.
- **Tier 3 — Negative result**: more than 5% below backprop with no
  compensating advantage.

Usage::

    uv run python -m computronium.validation.backprop_parity \
        --task digits --depths 2,3 --hidden-dims 256,512 \
        --seeds 3 --epochs 2 --device cpu \
        --families backprop,fa,target_prop,predictive_coding,eqprop_feedback \
        --output-dir runs/parity/digits_mlp
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = get_logger()

__all__ = [
    "Contract",
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

# Default width ladder for the capacity-controlled backprop search. Sized to
# span the MLP feasible width range across small (digits, input_dim≈256) and
# large (MNIST, input_dim=784) tasks; the search picks the closest non-negative
# param match.
_DEFAULT_WIDTH_LADDER: tuple[int, ...] = (
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    160,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
)


class Contract(StrEnum):
    """Per §15.4 the report carries three independent comparison contracts."""

    WIDTH_MATCHED = "width_matched"
    CAPACITY_CONTROLLED = "capacity_controlled"
    # The compute-matched contract reuses the width-matched probes (same
    # epochs / seeds / wall-clock cap) but reports FLOPs and computes the
    # tier. Keep it as a distinct tag so the reader can pull the primary
    # comparison row in the report without re-inferring from field presence.
    COMPUTE_MATCHED = "compute_matched"


# The families the plan shortlists for compute-matched parity (C1). Each maps
# to registered model names that are prospected for the best config.
_FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "backprop": ("backprop_mlp",),
    "fa": (
        "feedback_alignment",
        "standard_fa",
        "direct_feedback_alignment_eqprop",
        "dfa_deep",
    ),
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
    target_params: int,
    *,
    task: str,
    depth: int,
    target_widths: tuple[int, ...],
) -> tuple[int, int]:
    """Pick the width closest to matching ``target_params`` within tolerance.

    Estimates the parameter count statically (no training) across the candidate
    widths at the given depth and returns ``(width, params)`` for the first
    width that lands within ``±PARAM_TOLERANCE`` of the target — or the width
    whose count is nearest to target when none does (so the disparity, not a
    crash, is reported).

    Raises:
        ValueError: If no registered model matches ``model``.
    """
    from computronium.experiment.param_estimator import estimate_param_count

    best: tuple[int, int] | None = None
    for width in target_widths:
        cfg = {"hidden_dim": width, "num_layers": depth}
        count = estimate_param_count(
            model,
            cfg,
            input_dim=_task_input_dim(task),
            output_dim=_task_output_dim(task),
        )
        if best is None or abs(count - target_params) < abs(best[1] - target_params):
            best = (width, count)
        if abs(count - target_params) < target_params * PARAM_TOLERANCE:
            return width, count
    if best is None:  # pragma: no cover - candidate widths always non-empty
        raise RuntimeError(f"no candidate width resolved for {model}")
    return best


def _task_input_dim(task: str) -> int:
    from computronium.domains.registry import resolve_task

    return int(resolve_task(task).input_dim)


def _task_output_dim(task: str) -> int:
    from computronium.domains.registry import resolve_task

    return int(resolve_task(task).output_dim)


def _run_probe(  # probe call mirrors the driver contract  # ruff: ignore[too-many-arguments]
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
    """Train one probe and return its metrics dict.

    The probe driver is constructed with ``track_flops=True`` so the primary
    (compute-matched) contract can record forward+backward FLOPs — the §15.4
    honest currency for PC/eqprop-style computronium families whose settling
    steps cost more FLOPs than backprop's single forward+backward pass.
    """
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


def backprop_baseline(  # baseline signature is the report contract
    *,
    task: str,
    depth: int,
    hidden_dim: int,
    epochs: int,
    seeds: int,
    device: str,
    learning_rate: float = 1e-3,
) -> tuple[float, dict[str, object], list[dict[str, float]]]:
    """Train the backprop reference.

    Returns ``(mean_acc, last_metrics, probes)`` where ``probes`` is the per-
    seed record list (one entry per seed) used by the §15.4 three-contract
    comparison arms — effect sizes need per-seed accuracies, not just the
    mean. The baseline is always ``backprop_mlp`` at the requested depth/width,
    seeded once per ``seeds`` and averaged. Raises ``RuntimeError`` if no
    seed produces a valid run (the parity report must not silently show a dead
    baseline).
    """
    from computronium.experiment.probe import CoreTrainerDriver

    driver = CoreTrainerDriver(
        num_workers=0,
        batch_size=64,
        track_energy=False,
        track_flops=True,
        track_memory=True,
        record_results=False,
        allow_bptt_fallback=True,
    )
    probes: list[dict[str, float]] = []
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
        probes.append({
            "acc": float(m["final_acc"]),
            "loss": float(m["final_train_loss"]),
            "epoch_time": float(m.get("epoch_time_s", 0.0)),
            "peak_mem": float(m.get("peak_memory_mb", 0.0)),
            "forward_flops": float(m.get("forward_flops", 0.0)),
            "backward_flops": float(m.get("backward_flops", 0.0)),
        })
        metrics = m  # last seed's compute metrics — representative
    if not probes:
        raise RuntimeError("backprop baseline produced no valid seeds")
    return sum(p["acc"] for p in probes) / len(probes), metrics, probes


def _collect_probes(  # probe contract
    driver: object,
    model_name: str,
    task: str,
    *,
    hidden_dim: int,
    depth: int,
    learning_rate: float,
    epochs: int,
    seeds: int,
    device: str,
) -> tuple[list[dict[str, float]], list[str]]:
    """Run ``model_name``/``depth``/``hidden_dim`` across seeds.

    Returns ``(probes, failures)``. Each probe carries the metrics the parity
    contract needs: per-seed accuracy, loss, epoch time, peak memory, and
    forward+backward FLOPs (used for the §15.4 compute-matched contract).
    """
    probes: list[dict[str, float]] = []
    failures: list[str] = []
    for seed in range(seeds):
        try:
            m = _run_probe(
                driver,
                model_name,
                task,
                hidden_dim=hidden_dim,
                depth=depth,
                learning_rate=learning_rate,
                epochs=epochs,
                seed=seed,
                device=device,
            )
        except RuntimeError as exc:
            failures.append(f"{model_name}@{depth}/{seed}: {exc}")
            continue
        probes.append({
            "acc": float(m["final_acc"]),
            "loss": float(m["final_train_loss"]),
            "epoch_time": float(m.get("epoch_time_s", 0.0)),
            "peak_mem": float(m.get("peak_memory_mb", 0.0)),
            "forward_flops": float(m.get("forward_flops", 0.0)),
            "backward_flops": float(m.get("backward_flops", 0.0)),
        })
    return probes, failures


def _aggregate_model_entry(  # report contract
    *,
    model_name: str,
    family: str,
    depth: int,
    width: int,
    params: int,
    epochs: int,
    probes: list[dict[str, float]],
) -> dict[str, object]:
    """Build the per-model JSON/markdown row from raw per-seed probes."""
    from computronium.validation.statistics import bootstrap_percentile_ci

    accs = [p["acc"] for p in probes]
    lo, hi = bootstrap_percentile_ci(accs, seed=0, n_boot=500)
    total_flops = sum(p["forward_flops"] + p["backward_flops"] for p in probes) / len(
        probes
    )
    return {
        "model": model_name,
        "family": family,
        "depth": depth,
        "hidden_dim": width,
        "params": params,
        "epochs": epochs,
        "seed_count": len(accs),
        "mean_accuracy": sum(accs) / len(accs),
        "accuracy_ci95": [lo, hi],
        "mean_loss": sum(p["loss"] for p in probes) / len(probes),
        "epoch_time_s": sum(p["epoch_time"] for p in probes) / len(probes),
        "peak_memory_mb": sum(p["peak_mem"] for p in probes) / len(probes),
        "mean_total_flops": total_flops,
        "is_baseline": False,
    }


def _effect_sizes(
    model_accs: Sequence[float],
    baseline_accs: Sequence[float],
) -> dict[str, float]:
    """Per-comparison effect sizes for the §C2 required fields.

    Args:
        model_accs: Per-seed accuracies of the computronium model.
        baseline_accs: Per-seed accuracies of the backprop baseline (same seeds).

    Returns:
        Dict with ``cohen_d``, ``cliff_delta`` and ``bootstrap_p`` (a
        permutation-test p-value; see ``permutation_test_p`` for the rationale
        behind the absolute-Δ statistic). Fields are ``nan`` when the statistic
        is undefined for the provided cell size (``cohens_d`` requires
        ``≥2`` observations per group) so a 1-seed parity probe can still emit a
        valid JSON row without crashing the report writer.
    """
    from computronium.validation.statistics import (
        cliffs_delta,
        cohens_d,
        permutation_test_p,
    )

    def _safe(stat: Callable[[Sequence[float], Sequence[float]], float]) -> float:
        try:
            return round(stat(list(model_accs), list(baseline_accs)), 4)
        except ValueError, ZeroDivisionError:
            return float("nan")

    def _safe_p() -> float:
        try:
            return round(
                permutation_test_p(
                    list(model_accs), list(baseline_accs), n_perm=2_000, seed=0
                ),
                4,
            )
        except ValueError, ZeroDivisionError:
            return float("nan")

    return {
        "cohen_d": _safe(cohens_d),
        "cliff_delta": _safe(cliffs_delta),
        "bootstrap_p": _safe_p(),
    }


def _make_comparison(  # comparison-record contract  # ruff: ignore[too-many-arguments]
    *,
    contract: Contract,
    model_name: str,
    family: str,
    depth: int,
    model_params: int,
    baseline_params: int,
    model_accs: Sequence[float],
    baseline_accs: Sequence[float],
    model_epoch_time: float,
    baseline_epoch_time: float,
    model_peak_memory: float,
    baseline_peak_memory: float,
) -> dict[str, object]:
    """One comparison row tagged with its §15.4 contract."""
    mean_acc = sum(model_accs) / len(model_accs)
    mean_baseline = sum(baseline_accs) / len(baseline_accs)
    advantage = (
        baseline_epoch_time > 0 and model_epoch_time < baseline_epoch_time * 0.9
    ) or (baseline_peak_memory > 0 and model_peak_memory < baseline_peak_memory * 0.9)
    param_match = (
        abs(model_params - baseline_params) / baseline_params
        if baseline_params
        else 1.0
    )
    record: dict[str, object] = {
        "contract": contract.value,
        "model": model_name,
        "family": family,
        "depth": depth,
        "baseline_params": baseline_params,
        "params": model_params,
        "param_match": round(param_match, 3),
        "delta_accuracy": round(mean_acc - mean_baseline, 4),
        "tier": parity_tier(mean_acc, mean_baseline, advantage=advantage),
        "advantage": advantage,
        "baseline_acc": mean_baseline,
        "model_acc": mean_acc,
        "baseline_epoch_time_s": baseline_epoch_time,
        "model_epoch_time_s": model_epoch_time,
        "baseline_peak_memory_mb": baseline_peak_memory,
        "model_peak_memory_mb": model_peak_memory,
    }
    record.update(_effect_sizes(model_accs, baseline_accs))
    return record


def _width_search_backprop_for_bio_params(
    bio_params: int,
    *,
    task: str,
    depth: int,
    width_ladder: tuple[int, ...],
) -> tuple[int, int]:
    """Width-ladder search for the backprop width closest to ``bio_params``.

    The capacity-controlled contract (§15.4 "Tertiary") widens backprop until
    its param count matches the bio cell. The ladder is the geometric range
    from 16 to 2048; the picked width is the one whose backprop param count is
    closest to (and never below by more than the tolerance) the bio model's.

    Returns:
        ``(width, backprop_params)``. ``backprop_params`` reflects the actual
        constructed architecture so the runner can report the residual
        param-match honestly (it never inherits the bio model's count).
    """
    return _match_width(
        "backprop_mlp",
        bio_params,
        task=task,
        depth=depth,
        target_widths=width_ladder,
    )


def _run_cell(  # cell bundles the three §15.4 contract arms; locals track per-arm probes/notes  # ruff: ignore[too-many-arguments, too-many-locals]
    *,
    driver: object,
    model_name: str,
    family: str,
    task: str,
    depth: int,
    hidden_dims: tuple[int, ...],
    width_ladder: tuple[int, ...],
    baseline_params: int,
    baseline_probes: list[dict[str, float]],
    baseline_acc: float,
    baseline_epoch_time: float,
    baseline_peak_memory: float,
    epochs: int,
    seeds: int,
    learning_rate: float,
    device: str,
) -> tuple[dict[str, object] | None, list[dict[str, object]], str]:
    """Train one (model, depth) cell and return ``(model_entry, comparisons, note)``.

    Produces comparison records under three contracts (Plan 8 §15.4):
    ``width_matched`` (cross-width baseline), ``compute_matched`` (primarily
    reports FLOPs + tier computed on the width-matched probes — same epochs
    / seeds / wall-clock cap), and ``capacity_controlled`` (backprop retrained
    at the bio cell's param budget via a width search).

    Returns an empty ``comparisons`` list when no seed produced a valid run.
    """
    width, params = _match_width(
        model_name,
        baseline_params,
        task=task,
        depth=depth,
        target_widths=hidden_dims,
    )
    probes, failures = _collect_probes(
        driver,
        model_name,
        task,
        hidden_dim=width,
        depth=depth,
        learning_rate=learning_rate,
        epochs=epochs,
        seeds=seeds,
        device=device,
    )
    note = "; ".join(failures)
    if not probes:
        return None, [], note or f"{model_name}@{depth}: no valid seeds"

    entry = _aggregate_model_entry(
        model_name=model_name,
        family=family,
        depth=depth,
        width=width,
        params=params,
        epochs=epochs,
        probes=probes,
    )
    model_accs = [p["acc"] for p in probes]
    baseline_accs = [p["acc"] for p in baseline_probes] or [baseline_acc]
    model_epoch_time = float(entry["epoch_time_s"])
    model_peak_memory = float(entry["peak_memory_mb"])
    comparisons: list[dict[str, object]] = []

    width_record = _make_comparison(
        contract=Contract.WIDTH_MATCHED,
        model_name=model_name,
        family=family,
        depth=depth,
        model_params=params,
        baseline_params=baseline_params,
        model_accs=model_accs,
        baseline_accs=baseline_accs,
        model_epoch_time=model_epoch_time,
        baseline_epoch_time=baseline_epoch_time,
        model_peak_memory=model_peak_memory,
        baseline_peak_memory=baseline_peak_memory,
    )
    if float(width_record["param_match"]) > PARAM_TOLERANCE:
        note += (
            f"{'; ' if note else ''}{model_name}@{depth}[width_matched]: "
            f"param count {params} does not match baseline {baseline_params} "
            f"within {PARAM_TOLERANCE:.0%} (match={width_record['param_match']:.2f}); "
            "comparison may not be compute-matched (Plan 8 §C2)"
        )
    comparisons.append(width_record)

    compute_record = _make_comparison(
        contract=Contract.COMPUTE_MATCHED,
        model_name=model_name,
        family=family,
        depth=depth,
        model_params=params,
        baseline_params=baseline_params,
        model_accs=model_accs,
        baseline_accs=baseline_accs,
        model_epoch_time=model_epoch_time,
        baseline_epoch_time=baseline_epoch_time,
        model_peak_memory=model_peak_memory,
        baseline_peak_memory=baseline_peak_memory,
    )
    compute_record["model_total_flops"] = float(entry["mean_total_flops"])
    compute_record["baseline_total_flops"] = (
        sum(p["forward_flops"] + p["backward_flops"] for p in baseline_probes)
        / len(baseline_probes)
        if baseline_probes
        else 0.0
    )
    # FLOPs advantage: bio model uses fewer total FLOPs by ≥10% → counts as
    # an advantage for §C4 Tier 2 even when wall-clock is comparable. This is
    # the honest currency per §15.4 ("settling steps make FLOPs the honest
    # currency for PC/eqprop"); modelled as an ``flops_advantage`` flag.
    base_flops = float(compute_record["baseline_total_flops"])
    flops_advantage = (
        base_flops > 0 and float(compute_record["model_total_flops"]) < base_flops * 0.9
    )
    compute_record["flops_advantage"] = flops_advantage
    if flops_advantage and not bool(compute_record["advantage"]):
        compute_record["advantage"] = True
        compute_record["tier"] = parity_tier(
            sum(model_accs) / len(model_accs),
            sum(baseline_accs) / len(baseline_accs),
            advantage=True,
        )
    comparisons.append(compute_record)

    # Capacity-controlled: widen backprop to the bio cell's param budget.
    if width_ladder:
        bp_width, bp_params = _width_search_backprop_for_bio_params(
            params,
            task=task,
            depth=depth,
            width_ladder=width_ladder,
        )
        cap_probes, cap_failures = _collect_probes(
            driver,
            "backprop_mlp",
            task,
            hidden_dim=bp_width,
            depth=depth,
            learning_rate=learning_rate,
            epochs=epochs,
            seeds=seeds,
            device=device,
        )
        if cap_failures:
            note += f"{'; ' if note else ''}capacity-controlled backprop: " + "; ".join(
                cap_failures
            )
        if cap_probes:
            cap_accs = [p["acc"] for p in cap_probes]
            cap_model_epoch_time = sum(p["epoch_time"] for p in cap_probes) / len(
                cap_probes
            )
            cap_model_peak_memory = sum(p["peak_mem"] for p in cap_probes) / len(
                cap_probes
            )
            cap_record = _make_comparison(
                contract=Contract.CAPACITY_CONTROLLED,
                model_name=model_name,
                family=family,
                depth=depth,
                model_params=params,
                baseline_params=bp_params,
                model_accs=model_accs,
                baseline_accs=cap_accs,
                model_epoch_time=model_epoch_time,
                baseline_epoch_time=cap_model_epoch_time,
                model_peak_memory=model_peak_memory,
                baseline_peak_memory=cap_model_peak_memory,
            )
            cap_record["baseline_width"] = bp_width
            cap_record["baseline_total_flops"] = sum(
                p["forward_flops"] + p["backward_flops"] for p in cap_probes
            ) / len(cap_probes)
            cap_record["model_total_flops"] = float(entry["mean_total_flops"])
            comparisons.append(cap_record)
        elif cap_failures:
            note += (
                f"{'; ' if note else ''}{model_name}@{depth}"
                "[capacity_controlled]: no valid backprop seeds"
            )

    return entry, comparisons, note


def run_parity(  # campaign signature; per-depth baseline + cells accumulate locals  # ruff: ignore[too-many-arguments, too-many-locals]
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
    width_ladder: tuple[int, ...] = _DEFAULT_WIDTH_LADDER,
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
        width_ladder: Backprop width candidates for the §15.4 capacity-
            controlled contract. The default ladder spans 16→2048 covers both
            the small (digits) and large (MNIST) tasks; pass ``()`` to disable
            the capacity-controlled arm (saves a backprop retrain per cell).

    Returns:
        The full report dict (models, comparisons, provenance).

    Raises:
        ValueError: On an unknown family.
    """
    from computronium.experiment.probe import CoreTrainerDriver

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
        track_flops=True,
        track_memory=True,
        record_results=False,
        allow_bptt_fallback=True,
    )

    models: dict[str, object] = {}
    comparisons: list[dict[str, object]] = []
    notes: list[str] = []

    for depth in depths:
        # Backprop baseline for this depth.
        (
            baseline_acc,
            baseline_metrics,
            baseline_probes,
        ) = backprop_baseline(
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
        baseline_metrics_dict: dict[str, object] = {
            "model": "backprop_mlp",
            "family": "backprop",
            "depth": depth,
            "hidden_dim": hidden_dims[0],
            "params": baseline_params,
            "epochs": epochs,
            "seed_count": seeds,
            "mean_accuracy": baseline_acc,
            "mean_loss": float(baseline_metrics.get("final_train_loss", 0.0)),
            "epoch_time_s": baseline_epoch_time,
            "peak_memory_mb": baseline_peak_memory,
            "mean_total_flops": (
                sum(p["forward_flops"] + p["backward_flops"] for p in baseline_probes)
                / len(baseline_probes)
                if baseline_probes
                else 0.0
            ),
            "is_baseline": True,
        }
        models[f"backprop_mlp@{depth}"] = baseline_metrics_dict

        for family in wanted:
            for model_name in _FAMILY_MODELS[family]:
                # The backprop reference already has its own family port (above);
                # skip it inside other family lists to keep the report single-entry.
                if model_name == "backprop_mlp":
                    continue

                entry, cell_comparisons, note = _run_cell(
                    driver=driver,
                    model_name=model_name,
                    family=family,
                    task=task,
                    depth=depth,
                    hidden_dims=hidden_dims,
                    width_ladder=width_ladder,
                    baseline_params=baseline_params,
                    baseline_probes=baseline_probes,
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
                comparisons.extend(cell_comparisons)

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
    from computronium.experiment.param_estimator import estimate_param_count

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
    """Render the parity report as a Plan 8 §15.4 three-contract summary."""
    lines = [
        "# Compute-Matched Parity Report",
        "",
        f"**Task:** {report['task']}",
        f"**Depths:** {report['depths']}",
        f"**Epochs:** {report['epochs']}",
        f"**Seeds:** {report['seeds']}",
        f"**Device:** {report['device']}",
        "",
        "_Three contracts per Plan 8 §15.4: width_matched (Secondary — same "
        "width ladder as backprop baseline), compute_matched (Primary — same "
        "probes + FLOPs reported, tier computed here), capacity_controlled "
        "(backprop widened to the bio model's param budget — isolates "
        "capacity from rule)._",
        "",
        "## Models",
        "",
        "| Model | Family | Depth | Width | Params | Seeds | Mean Acc | CI95 | "
        "Mean Loss | Epoch s | Peak MB | MFLOPs |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    models = report["models"]
    for key in sorted(models):
        m = models[key]
        m_flops = float(m.get("mean_total_flops", 0.0)) / 1e6
        lines.append(
            f"| {m['model']} | {m['family']} | {m['depth']} | {m['hidden_dim']} | "
            f"{m['params']} | {m['seed_count']} | "
            f"{m['mean_accuracy']:.4f} | "
            f"{_ci_str(m)} | {m['mean_loss']:.4f} | "
            f"{m['epoch_time_s']:.3f} | {m['peak_memory_mb']:.1f} | "
            f"{m_flops:.1f} |"
        )
    lines.extend(["", "## Comparisons vs Backprop", ""])
    lines.append(
        "| Contract | Family | Model | Depth | Δ Acc | Tier | Param Match"
        " | Cohen d | Cliff δ | boot p | Base MFLOPs | Model MFLOPs |"
    )
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for c in report["comparisons"]:
        base_flops_m = float(c.get("baseline_total_flops", 0.0)) / 1e6
        model_flops_m = float(c.get("model_total_flops", 0.0)) / 1e6
        contract = c.get("contract", "width_matched")
        row = (
            f"| {contract} | {c['family']} | {c['model']} | {c['depth']} | "
            f"{c['delta_accuracy']:+.4f} | {c['tier']} | {c['param_match']:.3f} | "
            f"{_fmt_or_dash(c.get('cohen_d'), '{:+.3f}')} | "
            f"{_fmt_or_dash(c.get('cliff_delta'), '{:+.3f}')} | "
            f"{_fmt_or_dash(c.get('bootstrap_p'), '{:.3f}')} | "
            f"{base_flops_m:.1f} | {model_flops_m:.1f} |"
        )
        lines.append(row)
    if not report["comparisons"]:
        lines.append("_No comparisons — baseline only._")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {n}" for n in report["notes"] or ["_none_"])
    lines.append("")
    return "\n".join(lines)


def _fmt_or_dash(value: object, fmt: str) -> str:
    """Format a possibly-missing numeric comparison field (no ``None`` in MD)."""
    if value is None:
        return "—"
    try:
        f = float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return "—"
    if math.isnan(f):  # parity rows drop the stat when seeds < 2
        return "—"
    return fmt.format(f)


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
