"""Apply the R5b-0 implementation-fidelity gate to a commissioned campaign.

Loads a campaign's episode records, runs the fidelity probes over every
coordinate present, and writes ``records/fidelity_manifest.json`` +
``records/fidelity_report.md``: per-axis rollup, quarantined coordinates,
leakage evidence (reported accuracy vs Δθ movement vs free-settle accuracy),
and defect-filtered attribution (pooled + stratified per seed x family).

Per the TODO8 policy, deltas on coordinates failing fidelity are
quarantined — excluded from attribution, listed by identity, never
interpreted. A failed fidelity check is inconclusive, not a refutation.

Usage:
    uv run scripts/fidelity_gate_report.py --campaign-dir autoscientist_campaigns/r51c
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R5b-0 fidelity gate report for a commissioned campaign"
    )
    parser.add_argument("--campaign-dir", default="autoscientist_campaigns/r51c")
    parser.add_argument("--metric", default="task_accuracy")
    return parser.parse_args()


def _load_records(path: Path) -> list[dict]:
    episodes = path / "records" / "episodes.json"
    if not episodes.exists():
        sys.exit(f"{episodes} not found — run the commissioning script first")
    return json.loads(episodes.read_text())


def _leakage_probe(coordinate: str) -> dict[str, float] | None:
    """Reported accuracy vs Δθ movement vs free-settle (target-free) accuracy."""
    from computronium.core.campaign.evaluation import (
        IncompatibleCoordinateError,
        build_coordinate_system,
        episode_batch,
    )
    from computronium.core.pipeline import forward_pass
    from computronium.ontology import SystemState

    try:
        joint = build_coordinate_system(coordinate)
    except IncompatibleCoordinateError:
        # R3.9-fenced pairing: uncomposable, so no leakage numbers exist.
        return None
    x, y = episode_batch(0)
    before = {
        name: tensor.detach().clone() for name, tensor in joint.geometry.params.items()
    }
    metrics = joint.train_step(x, y)
    moved = float(
        sum(
            (joint.geometry.params[name].detach() - before[name]).norm()
            for name in before
        )
    )
    state = SystemState(x=x, y=y)
    state.activations = forward_pass(joint.substrate, joint.geometry, x)
    settled = joint.dynamics.settle(state, joint.geometry, joint.substrate, target=None)
    acts = settled.activations
    out = acts[-1] if isinstance(acts, list) else acts
    free_acc = float((out.argmax(-1) == y).float().mean())
    return {
        "reported_acc": float(metrics["accuracy"]),
        "dtheta": moved,
        "free_settle_acc": free_acc,
    }


def _rollup(manifest: dict) -> list[dict]:
    by_value: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"pass": 0, "fail": 0, "blocked": 0}
    )
    reasons: dict[tuple[str, str], str] = {}
    for verdict in manifest.values():
        for check in verdict.checks:
            key = (check.axis, check.value)
            by_value[key][check.status] += 1
            if check.status == "fail" and key not in reasons:
                reasons[key] = check.detail
    return [
        {
            "axis": axis,
            "value": value,
            **counts,
            "first_failure_reason": reasons.get((axis, value), ""),
        }
        for (axis, value), counts in sorted(by_value.items())
    ]


def _write_manifest(manifest: dict, path: Path) -> None:
    """Serialize the fidelity manifest for downstream tooling."""
    payload = {
        c: {
            "passed": v.passed,
            "failures": list(v.failures),
            "checks": [
                {
                    "axis": k.axis,
                    "value": k.value,
                    "status": k.status,
                    "detail": k.detail,
                }
                for k in v.checks
            ],
        }
        for c, v in manifest.items()
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evidence(records_obj: list, manifest: dict, metric: str) -> tuple:
    """Defect-filtered attribution, stratified attribution, and Pareto."""
    from computronium.core.campaign.fidelity import defect_filtered_attribution

    filtered = defect_filtered_attribution(records_obj, manifest, metric=metric)
    stratified = _stratified_attribution(records_obj, manifest, metric)
    pareto = _defect_filtered_pareto(records_obj, manifest)
    return filtered, stratified, pareto


def main() -> None:
    args = _parse_args()
    from computronium.core.campaign.fidelity import fidelity_manifest

    campaign = Path(args.campaign_dir)
    records = _load_records(campaign)
    coordinates = sorted({r["coordinate"] for r in records})
    print(f"fidelity gate over {len(coordinates)} coordinates ({len(records)} records)")

    manifest = fidelity_manifest(coordinates)
    passing = {c for c, v in manifest.items() if v.passed}
    excluded = sorted(set(coordinates) - passing)

    leakage = {
        c: stats for c in coordinates if (stats := _leakage_probe(c)) is not None
    }
    n_fenced = len(coordinates) - len(leakage)

    records_obj = _records_from_dicts(records)
    evidence = _evidence(records_obj, manifest, args.metric)

    manifest_path = campaign / "records" / "fidelity_manifest.json"
    _write_manifest(manifest, manifest_path)
    print(f"manifest -> {manifest_path}")

    report = _render_report(
        campaign,
        manifest,
        leakage,
        filtered=evidence[0],
        stratified=evidence[1],
        pareto=evidence[2],
        excluded=excluded,
        passing=passing,
        n_fenced=n_fenced,
    )
    report_path = campaign / "records" / "fidelity_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"report -> {report_path}")


def _records_from_dicts(records: list[dict]) -> list:
    from computronium.core.campaign.frontier_record import FrontierRecord

    return [FrontierRecord.from_dict(r) for r in records]


def _stratified_attribution(records_obj: list, manifest: dict, metric: str) -> dict:
    """Per-(seed, family) attribution over passing records: rank stability."""
    from computronium.analysis.counterfactual import attribute_axis_effects

    groups: dict[tuple[int, str], list] = defaultdict(list)
    for record in records_obj:
        verdict = manifest.get(record.coordinate)
        if verdict is not None and verdict.passed:
            groups[record.seed, record.task_name].append(record)
    result = {}
    for (seed, task), group_records in sorted(groups.items()):
        attributions = attribute_axis_effects(group_records, metric=metric)
        result[f"seed={seed}/{task}"] = [
            {"axis": a.axis, "mean_delta": a.mean_delta, "n_pairs": a.n_pairs}
            for a in attributions
        ]
    return result


PARETO_OBJECTIVES: tuple[str, ...] = (
    "task_loss",
    "compute",
    "memory",
    "energy",
    "latency",
    "plastic_state_capacity",
)


@dataclass(frozen=True, slots=True)
class _AttributionRow:
    axis: str
    from_value: str
    to_value: str
    mean_delta: float
    n_pairs: int = 0


def _canonicalized(attributions) -> list:
    """Merge direction-split attributions into canonical signed rows.

    Minimal pairs occur in both record orders (cross-seed/cross-family),
    so one logical axis effect appears as two raw rows; reports (like the
    R5b-D locks) compare transitions canonically.
    """
    from computronium.core.campaign.discovery import _canonical

    grouped: dict[tuple[str, str, str], list] = defaultdict(list)
    for attribution in attributions:
        key = _canonical(
            (attribution.axis, attribution.from_value, attribution.to_value)
        )
        grouped[key].append(attribution)
    rows = []
    for (axis, from_value, to_value), members in grouped.items():
        deltas = [
            -a.mean_delta
            if (a.from_value, a.to_value) != (from_value, to_value)
            else a.mean_delta
            for a in members
        ]
        rows.append(
            _AttributionRow(
                axis=axis,
                from_value=from_value,
                to_value=to_value,
                mean_delta=sum(deltas) / len(deltas),
                n_pairs=sum(a.n_pairs for a in members),
            )
        )
    return sorted(rows, key=lambda r: abs(r.mean_delta), reverse=True)


def _defect_filtered_pareto(records_obj: list, manifest: dict) -> dict:
    """Defect-filtered Pareto frontier over the resource vector (R5b-C).

    The frontier is computed only over records whose coordinates pass
    fidelity, over (task_loss, compute, memory, energy, latency, psi-cap)
    — all minimized except task_loss. Knee ownership = the axes on which a
    frontier coordinate holds a value no other frontier coordinate shares.
    """
    from computronium.core.campaign.pareto import objective_vector, pareto_frontier

    passing = [
        r
        for r in records_obj
        if (v := manifest.get(r.coordinate)) is not None and v.passed
    ]
    if not passing:
        return {"frontier": [], "n_passing": 0, "hypervolume": 0.0}
    vectors = [objective_vector(r, PARETO_OBJECTIVES) for r in passing]
    # Data-derived reference: worst passing value per objective, nudged 5%
    # below the observed span so every frontier point contributes volume.
    reference = tuple(
        lo - 0.05 * ((hi - lo) or 1.0)
        for lo, hi in zip(
            (min(v[i] for v in vectors) for i in range(len(PARETO_OBJECTIVES))),
            (max(v[i] for v in vectors) for i in range(len(PARETO_OBJECTIVES))),
            strict=True,
        )
    )
    frontier = pareto_frontier(
        passing,
        PARETO_OBJECTIVES,
        maximize=(True,) * len(PARETO_OBJECTIVES),
        reference_point=reference,
    )
    by_coordinate: dict[str, list] = defaultdict(list)
    for record in frontier.frontier:
        by_coordinate[record.coordinate].append(record)

    def _stat(rows: list, attr: str) -> float:
        return sum(getattr(r, attr) for r in rows) / len(rows)

    def _res(rows: list, attr: str) -> float:
        return sum(getattr(r.resources, attr) for r in rows) / len(rows)

    axes = ("substrate", "geometry", "dynamics", "plasticity", "credit", "update")
    coordinate_rows = []
    for coordinate, rows in sorted(
        by_coordinate.items(), key=lambda kv: _stat(kv[1], "task_loss")
    ):
        parts = coordinate.split("/")
        owned = [
            axis
            for axis, value in zip(axes, parts, strict=True)
            if sum(1 for c in by_coordinate if c.split("/")[axes.index(axis)] == value)
            == 1
        ]
        coordinate_rows.append({
            "coordinate": coordinate,
            "n": len(rows),
            "task_loss": _stat(rows, "task_loss"),
            "compute": _res(rows, "compute"),
            "memory": _res(rows, "memory"),
            "energy": _res(rows, "energy"),
            "latency_ms": _res(rows, "latency") * 1e3,
            "psi_capacity": _res(rows, "plastic_state_capacity"),
            "owned_axes": owned,
        })
    return {
        "frontier": coordinate_rows,
        "n_passing": len(passing),
        "n_dominated": len(frontier.dominated),
        "hypervolume": frontier.hypervolume,
    }


def _render_report(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - report context bundle
    campaign: Path,
    manifest: dict,
    leakage: dict[str, dict[str, float]],
    filtered,
    stratified: dict,
    excluded: list[str],
    passing: set[str],
    n_fenced: int,
    pareto: dict,
) -> str:
    by_class: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for coordinate, stats in leakage.items():
        parts = coordinate.split("/")
        by_class[parts[2], parts[4]].append(stats)

    rollup = _rollup(manifest)
    lines = [
        f"# R5b-0 fidelity gate report — `{campaign.name}`",
        "",
        f"- Coordinates probed: {len(manifest)} · passing: {len(passing)} · "
        f"quarantined: {len(excluded)} (of which {n_fenced} R3.9-fenced: "
        "uncomposable at composition, no leakage numbers)",
        f"- Records: {filtered.n_records_total} total · "
        f"{filtered.n_records_passing} survive the fidelity filter",
        "",
        "Policy: deltas on failing coordinates are quarantined from "
        "attribution — inconclusive, never a refutation.",
        "",
        "## Per-axis rollup",
        "",
        "| axis | value | pass | fail | blocked | first failure reason |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['axis']} | {r['value']} | {r['pass']} | {r['fail']} "
            f"| {r['blocked']} | {r['first_failure_reason'][:80]} |"
            for r in rollup
        ],
        "",
        "## Quarantined coordinates (excluded from attribution)",
        "",
    ]
    lines += [f"- `{c}` — {'; '.join(manifest[c].failures)}" for c in excluded]
    lines += [
        "",
        "## Leakage evidence (reported accuracy vs Δθ vs target-free settle)",
        "",
        "| dynamics x credit | n | mean reported acc | mean Δθ | "
        "mean free-settle acc |",
        "|---|---|---|---|---|",
        *[
            f"| {d} x {c} | {len(rows)} | "
            f"{sum(r['reported_acc'] for r in rows) / len(rows):.4f} | "
            f"{sum(r['dtheta'] for r in rows) / len(rows):.2e} | "
            f"{sum(r['free_settle_acc'] for r in rows) / len(rows):.4f} |"
            for (d, c), rows in sorted(by_class.items())
        ],
        "",
        "Δθ ≈ 0 with high reported accuracy ⇒ the metric measures the "
        "nudged settle (supervision leakage), not learning. Free-settle "
        "accuracy ≈ chance confirms it.",
        "",
        "## Defect-filtered attribution (pooled over passing coordinates)",
        "",
    ]
    if filtered.attributions:
        lines += [
            "| axis | from → to | mean Δ | pairs |",
            "|---|---|---|---|",
            *[
                f"| {a.axis} | {a.from_value} → {a.to_value} "
                f"| {a.mean_delta:+.4f} | {a.n_pairs} |"
                for a in _canonicalized(filtered.attributions)
            ],
        ]
    else:
        lines += ["No minimal pairs within the passing subspace."]
    lines += ["", "## Stratified attribution (per seed x family)", ""]
    if stratified:
        lines += [
            "| stratum | top axes (|mean Δ|) |",
            "|---|---|",
            *[
                f"| {stratum} | "
                + "; ".join(
                    f"{a['axis']} {a['mean_delta']:+.4f} (n={a['n_pairs']})"
                    for a in axes[:3]
                )
                + " |"
                for stratum, axes in stratified.items()
            ],
        ]
    else:
        lines += ["No passing records — nothing to attribute."]

    lines += [
        "",
        "## Defect-filtered Pareto frontier over the resource vector (R5b-C)",
        "",
        "Objectives: task_loss ↓ · compute ↓ · memory ↓ · energy ↓ · "
        "latency ↓ · ψ-capacity ↓ — all minimized except task_loss, "
        "computed only over records whose coordinates pass fidelity. "
        "Resource axes are the deterministic per-episode accounting wired "
        "by imp-17 (previously latency-only, which collapsed the frontier "
        "to a single loss minimizer); latency is wall-clock and noisy at "
        "smoke scale.",
        "",
    ]
    frontier_rows = pareto["frontier"]
    if not frontier_rows:
        lines += ["No passing records — no frontier."]
    else:
        lines += [
            f"Frontier: {len(frontier_rows)} coordinates "
            f"({pareto['n_dominated']} of {pareto['n_passing']} passing "
            f"records dominated) · hypervolume (data-derived reference): "
            f"{pareto['hypervolume']:.4g}",
            "",
            "Knee ownership: the axes on which a frontier coordinate holds a "
            "value no other frontier coordinate shares — the axis whose trade-"
            "off position that knee is bought with.",
            "",
            "| coordinate | owned axes | n | loss | compute | memory | energy "
            "| latency ms | ψ-cap |",
            "|---|---|---|---|---|---|---|---|---|",
            *[
                f"| `{r['coordinate']}` | {', '.join(r['owned_axes']) or '—'} "
                f"| {r['n']} | {r['task_loss']:.4f} | {r['compute']:.3g} "
                f"| {r['memory']:.4g} | {r['energy']:.4f} "
                f"| {r['latency_ms']:.2f} | {r['psi_capacity']:.0f} |"
                for r in frontier_rows
            ],
        ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
