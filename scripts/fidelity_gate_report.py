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
    from computronium.core.campaign.report import build_discovery_report

    discovery = build_discovery_report(
        records_obj, metric=args.metric, fidelity=manifest
    )

    manifest_path = campaign / "records" / "fidelity_manifest.json"
    _write_manifest(manifest, manifest_path)
    print(f"manifest -> {manifest_path}")

    report = _render_report(
        campaign,
        manifest,
        leakage,
        discovery=discovery,
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


def _render_report(
    campaign: Path,
    manifest: dict,
    leakage: dict[str, dict[str, float]],
    discovery,
    excluded: list[str],
    passing: set[str],
    n_fenced: int,
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
        f"- Records: {discovery.n_records} total · "
        f"{discovery.n_passing_records} survive the fidelity filter",
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
    if discovery.attribution:
        lines += [
            "| axis | from → to | mean Δ | pairs |",
            "|---|---|---|---|",
            *[
                f"| {a.axis} | {a.from_value} → {a.to_value} "
                f"| {a.mean_delta:+.4f} | {a.n_pairs} |"
                for a in discovery.attribution
            ],
        ]
    else:
        lines += ["No minimal pairs within the passing subspace."]
    lines += ["", "## Stratified attribution (per seed x family)", ""]
    if discovery.stratified:
        lines += [
            "| stratum | top axes (|mean Δ|) |",
            "|---|---|",
            *[
                f"| {stratum.stratum} | "
                + "; ".join(
                    f"{a.axis} {a.mean_delta:+.4f} (n={a.n_pairs})"
                    for a in stratum.rows[:3]
                )
                + " |"
                for stratum in discovery.stratified
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
    frontier_rows = discovery.frontier
    if not frontier_rows:
        lines += ["No passing records — no frontier."]
    else:
        lines += [
            f"Frontier: {len(frontier_rows)} coordinates "
            f"({discovery.n_dominated} of {discovery.n_passing_records} passing "
            f"records dominated) · hypervolume (data-derived reference): "
            f"{discovery.hypervolume:.4g}",
            "",
            "Knee ownership: the axes on which a frontier coordinate holds a "
            "value no other frontier coordinate shares — the axis whose trade-"
            "off position that knee is bought with.",
            "",
            "| coordinate | owned axes | n | loss | compute | memory | energy "
            "| latency ms | ψ-cap |",
            "|---|---|---|---|---|---|---|---|---|",
            *[
                f"| `{r.coordinate}` | {', '.join(r.owned_axes) or '—'} "
                f"| {r.records} | {r.values['task_loss']:.4f} "
                f"| {r.values['compute']:.3g} "
                f"| {r.values['memory']:.4g} | {r.values['energy']:.4f} "
                f"| {r.values['latency'] * 1e3:.2f} "
                f"| {r.values['plastic_state_capacity']:.0f} |"
                for r in frontier_rows
            ],
        ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
