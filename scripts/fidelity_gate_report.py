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


def _leakage_probe(coordinate: str) -> dict[str, float]:
    """Reported accuracy vs Δθ movement vs free-settle (target-free) accuracy."""
    from computronium.core.campaign.evaluation import (
        build_coordinate_system,
        episode_batch,
    )
    from computronium.core.pipeline import forward_pass
    from computronium.ontology import SystemState

    joint = build_coordinate_system(coordinate)
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


def main() -> None:
    args = _parse_args()
    from computronium.core.campaign.fidelity import (
        defect_filtered_attribution,
        fidelity_manifest,
    )

    campaign = Path(args.campaign_dir)
    records = _load_records(campaign)
    coordinates = sorted({r["coordinate"] for r in records})
    print(f"fidelity gate over {len(coordinates)} coordinates ({len(records)} records)")

    manifest = fidelity_manifest(coordinates)
    passing = {c for c, v in manifest.items() if v.passed}
    excluded = sorted(set(coordinates) - passing)

    leakage = {c: _leakage_probe(c) for c in coordinates}

    records_obj = _records_from_dicts(records)
    filtered = defect_filtered_attribution(records_obj, manifest, metric=args.metric)
    stratified = _stratified_attribution(records_obj, manifest, args.metric)

    manifest_payload = {
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
    manifest_path = campaign / "records" / "fidelity_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    print(f"manifest -> {manifest_path}")

    report = _render_report(
        campaign, manifest, leakage, filtered, stratified, excluded, passing
    )
    report_path = campaign / "records" / "fidelity_report.md"
    report_path.write_text(report)
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


def _render_report(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - report context bundle
    campaign: Path,
    manifest: dict,
    leakage: dict[str, dict[str, float]],
    filtered,
    stratified: dict,
    excluded: list[str],
    passing: set[str],
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
        f"quarantined: {len(excluded)}",
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
                for a in filtered.attributions
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
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
