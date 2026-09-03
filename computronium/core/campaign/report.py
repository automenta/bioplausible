"""Static discovery report (R5b-F Stage 1) over campaign records.

Renders the stack's evidence — defect-filtered Pareto frontier over the
resource vector, replication gate, counterfactual attribution (pooled +
per-stratum), and the per-iteration episode timeline — as deterministic
JSON plus a self-contained HTML page. When a fidelity manifest is supplied,
the frontier and attribution are computed only over records whose
coordinates pass the R5b-0 gate; quarantined coordinates are reported by
identity, never silently dropped (a failed fidelity check is inconclusive,
never a refutation).

No wall-clock data enters the payload, so identical records render byte
identical artifacts — the property the snapshot tests pin.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from computronium.core.campaign.discovery import (
    CanonicalAttribution,
    canonical_attributions,
)
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.core.campaign.pareto import objective_vector, pareto_frontier
from computronium.core.campaign.replication import (
    DEFAULT_MIN_FAMILIES,
    DEFAULT_MIN_SEEDS,
    ReplicationReport,
    replication_manifest,
    task_family,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Defect-filtered resource vector (R5b-C): all minimized except task_loss.
RESOURCE_OBJECTIVES: tuple[str, ...] = (
    "task_loss",
    "compute",
    "memory",
    "energy",
    "latency",
    "plastic_state_capacity",
)

_AXES: tuple[str, ...] = (
    "substrate",
    "geometry",
    "dynamics",
    "plasticity",
    "credit",
    "update",
)


class FidelityVerdict(Protocol):
    """Minimal shape the report consumes from a fidelity verdict."""

    @property
    def passed(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class StoredFidelityVerdict:
    """Fidelity verdict reconstructed from a saved fidelity manifest."""

    passed: bool


@dataclass(frozen=True, slots=True)
class FidelitySummary:
    """Gate state over the report's coordinates."""

    n_passing: int
    n_quarantined: int
    quarantined: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TimelineStep:
    """Episode aggregate for one (seed, iteration) slot."""

    seed: int
    iteration: int
    episodes: int
    mean_loss: float
    mean_accuracy: float


@dataclass(frozen=True, slots=True)
class FrontierRow:
    """Coordinate-level mean over the resource-vector frontier."""

    coordinate: str
    records: int
    values: dict[str, float]
    owned_axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplicationRow:
    """Replication-gate status for one coordinate."""

    coordinate: str
    seeds: int
    families: int
    replicated: bool
    unmet: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StratifiedAttribution:
    """Canonical attribution rows for one (seed, family) stratum."""

    stratum: str
    rows: tuple[CanonicalAttribution, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    """Deterministic discovery report over campaign episode records."""

    metric: str
    objectives: tuple[str, ...]
    n_records: int
    n_passing_records: int
    n_coordinates: int
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    fidelity: FidelitySummary | None
    timeline: tuple[TimelineStep, ...]
    frontier: tuple[FrontierRow, ...]
    n_frontier_records: int
    n_dominated: int
    hypervolume: float
    replication: tuple[ReplicationRow, ...]
    attribution: tuple[CanonicalAttribution, ...]
    stratified: tuple[StratifiedAttribution, ...]

    def to_dict(self) -> dict:
        """JSON-serializable payload (deterministic ordering)."""
        return {
            "metric": self.metric,
            "objectives": list(self.objectives),
            "n_records": self.n_records,
            "n_passing_records": self.n_passing_records,
            "n_coordinates": self.n_coordinates,
            "seeds": list(self.seeds),
            "families": list(self.families),
            "fidelity": (
                {
                    "n_passing": self.fidelity.n_passing,
                    "n_quarantined": self.fidelity.n_quarantined,
                    "quarantined": list(self.fidelity.quarantined),
                }
                if self.fidelity is not None
                else None
            ),
            "timeline": [
                {
                    "seed": s.seed,
                    "iteration": s.iteration,
                    "episodes": s.episodes,
                    "mean_loss": s.mean_loss,
                    "mean_accuracy": s.mean_accuracy,
                }
                for s in self.timeline
            ],
            "frontier": [
                {
                    "coordinate": r.coordinate,
                    "records": r.records,
                    "values": dict(r.values),
                    "owned_axes": list(r.owned_axes),
                }
                for r in self.frontier
            ],
            "n_frontier_records": self.n_frontier_records,
            "n_dominated": self.n_dominated,
            "hypervolume": self.hypervolume,
            "replication": [
                {
                    "coordinate": r.coordinate,
                    "seeds": r.seeds,
                    "families": r.families,
                    "replicated": r.replicated,
                    "unmet": list(r.unmet),
                }
                for r in self.replication
            ],
            "attribution": [_attribution_dict(a) for a in self.attribution],
            "stratified": [
                {
                    "stratum": s.stratum,
                    "rows": [_attribution_dict(a) for a in s.rows],
                }
                for s in self.stratified
            ],
        }

    def to_json(self) -> str:
        """Deterministic JSON serialization of the payload."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_html(self) -> str:
        """Self-contained static HTML page (inline CSS, no scripts)."""
        return _render_html(self)

    def write(
        self, directory: str | Path, *, stem: str = "discovery_report"
    ) -> tuple[Path, Path]:
        """Write ``<stem>.json`` and ``<stem>.html`` into ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{stem}.json"
        html_path = out / f"{stem}.html"
        json_path.write_text(self.to_json(), encoding="utf-8")
        html_path.write_text(self.to_html(), encoding="utf-8")
        return json_path, html_path


def _attribution_dict(attribution: CanonicalAttribution) -> dict:
    return {
        "axis": attribution.axis,
        "from": attribution.from_value,
        "to": attribution.to_value,
        "mean_delta": attribution.mean_delta,
        "n_pairs": attribution.n_pairs,
    }


def build_discovery_report(
    records: Sequence[FrontierRecord],
    *,
    metric: str = "task_accuracy",
    objectives: tuple[str, ...] = RESOURCE_OBJECTIVES,
    fidelity: Mapping[str, FidelityVerdict] | None = None,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    min_families: int = DEFAULT_MIN_FAMILIES,
) -> DiscoveryReport:
    """Build the discovery report over campaign episode records.

    Args:
        records: Frontier records (all seeds/families of one campaign).
        metric: Attribution metric (``task_accuracy`` by default).
        objectives: Pareto objective names (the resource vector by default).
        fidelity: Optional coordinate → verdict mapping; coordinates without
            a passing verdict are quarantined from the frontier and
            attribution (the timeline and replication gate still cover
            every record). ``None`` disables the gate.
        min_seeds: Replication-gate seed requirement.
        min_families: Replication-gate task-family requirement.

    Returns:
        DiscoveryReport; every section is deterministically ordered.
    """
    coordinates = {r.coordinate for r in records}
    if fidelity is None:
        passing_records = list(records)
        fidelity_summary = None
        n_passing_records = len(records)
    else:
        passing = {c for c, v in fidelity.items() if v.passed}
        passing_records = [r for r in records if r.coordinate in passing]
        quarantined = sorted(coordinates - passing)
        fidelity_summary = FidelitySummary(
            n_passing=len(passing & coordinates),
            n_quarantined=len(quarantined),
            quarantined=tuple(quarantined),
        )
        n_passing_records = len(passing_records)

    frontier, n_frontier_records, n_dominated, hypervolume = _frontier_section(
        passing_records, objectives
    )
    return DiscoveryReport(
        metric=metric,
        objectives=objectives,
        n_records=len(records),
        n_passing_records=n_passing_records,
        n_coordinates=len(coordinates),
        seeds=tuple(sorted({r.seed for r in records})),
        families=tuple(sorted({task_family(r.task_name) for r in records})),
        fidelity=fidelity_summary,
        timeline=_timeline_section(records),
        frontier=frontier,
        n_frontier_records=n_frontier_records,
        n_dominated=n_dominated,
        hypervolume=hypervolume,
        replication=_replication_section(records, min_seeds, min_families),
        attribution=tuple(_pooled_attribution(passing_records, metric)),
        stratified=_stratified_section(passing_records, metric),
    )


def _pooled_attribution(
    records: Sequence[FrontierRecord], metric: str
) -> list[CanonicalAttribution]:
    from computronium.analysis.counterfactual import attribute_axis_effects

    return canonical_attributions(attribute_axis_effects(records, metric=metric))


def _timeline_section(
    records: Sequence[FrontierRecord],
) -> tuple[TimelineStep, ...]:
    """Per-(seed, iteration) episode aggregates — the kill→resume view."""
    grouped: dict[tuple[int, int], list[FrontierRecord]] = {}
    for record in records:
        grouped.setdefault((record.seed, record.episode_index), []).append(record)
    return tuple(
        TimelineStep(
            seed=seed,
            iteration=iteration,
            episodes=len(group),
            mean_loss=sum(r.task_loss for r in group) / len(group),
            mean_accuracy=sum(r.task_accuracy for r in group) / len(group),
        )
        for (seed, iteration), group in sorted(grouped.items())
    )


def _mean(records: Sequence[FrontierRecord], attribute: str) -> float:
    if attribute == "task_loss":
        return sum(r.task_loss for r in records) / len(records)
    return sum(getattr(r.resources, attribute) for r in records) / len(records)


def _frontier_section(
    records: Sequence[FrontierRecord], objectives: tuple[str, ...]
) -> tuple[tuple[FrontierRow, ...], int, int, float]:
    """Defect-filtered Pareto over the resource vector (R5b-C semantics)."""
    if not records:
        return (), 0, 0, 0.0
    vectors = [objective_vector(r, objectives) for r in list(records)]
    spans = [
        (min(v[i] for v in vectors), max(v[i] for v in vectors))
        for i in range(len(objectives))
    ]
    # Data-derived reference 5% below the observed span so every frontier
    # point contributes volume (zero-span objectives are vacuous, unit width).
    reference = tuple(lo - 0.05 * ((hi - lo) or 1.0) for lo, hi in spans)
    frontier = pareto_frontier(
        list(records),
        objectives,
        maximize=(True,) * len(objectives),
        reference_point=reference,
    )
    by_coordinate: dict[str, list[FrontierRecord]] = {}
    for record in frontier.frontier:
        by_coordinate.setdefault(record.coordinate, []).append(record)
    value_counts: dict[int, dict[str, int]] = {
        i: Counter(c.split("/")[i] for c in by_coordinate) for i in range(len(_AXES))
    }

    def _mean_loss(group: list[FrontierRecord]) -> float:
        return sum(r.task_loss for r in group) / len(group)

    rows = []
    for coordinate in sorted(by_coordinate, key=lambda c: _mean_loss(by_coordinate[c])):
        group = by_coordinate[coordinate]
        owned = tuple(
            axis
            for axis, value in zip(_AXES, coordinate.split("/"), strict=True)
            if value_counts[_AXES.index(axis)][value] == 1
        )
        rows.append(
            FrontierRow(
                coordinate=coordinate,
                records=len(group),
                values={objective: _mean(group, objective) for objective in objectives},
                owned_axes=owned,
            )
        )
    return (
        tuple(rows),
        len(frontier.frontier),
        len(frontier.dominated),
        frontier.hypervolume,
    )


def _replication_section(
    records: Sequence[FrontierRecord], min_seeds: int, min_families: int
) -> tuple[ReplicationRow, ...]:
    manifest: dict[str, ReplicationReport] = replication_manifest(
        records, min_seeds=min_seeds, min_families=min_families
    )
    return tuple(
        ReplicationRow(
            coordinate=report.coordinate,
            seeds=len(report.seeds),
            families=len(report.task_families),
            replicated=report.replicated,
            unmet=report.unmet(),
        )
        for report in manifest.values()
    )


def _stratified_section(
    records: Sequence[FrontierRecord], metric: str
) -> tuple[StratifiedAttribution, ...]:
    from computronium.analysis.counterfactual import attribute_axis_effects

    strata: dict[tuple[int, str], list[FrontierRecord]] = {}
    for record in records:
        strata.setdefault((record.seed, record.task_name), []).append(record)
    return tuple(
        StratifiedAttribution(
            stratum=f"seed={seed}/{task}",
            rows=tuple(
                canonical_attributions(attribute_axis_effects(group, metric=metric))
            ),
        )
        for (seed, task), group in sorted(strata.items())
    )


def load_campaign_records(
    campaign_dir: str | Path,
) -> tuple[list[FrontierRecord], dict[str, StoredFidelityVerdict] | None]:
    """Load episode records + the stored fidelity manifest from a campaign.

    Args:
        campaign_dir: Commissioned campaign directory containing
            ``records/episodes.json`` (and optionally
            ``records/fidelity_manifest.json``).

    Returns:
        Records and the fidelity verdict mapping; ``None`` when no manifest
        was saved (every coordinate is then treated as passing).
    """
    records_dir = Path(campaign_dir) / "records"
    episodes_path = records_dir / "episodes.json"
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    records = [
        FrontierRecord.from_dict(entry)
        for entry in json.loads(episodes_path.read_text(encoding="utf-8"))
    ]
    manifest_path = records_dir / "fidelity_manifest.json"
    if not manifest_path.exists():
        return records, None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return records, {
        coordinate: StoredFidelityVerdict(passed=verdict["passed"])
        for coordinate, verdict in raw.items()
    }


# -- HTML rendering -------------------------------------------------------------


_CSS = """\
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 72rem;
  color: #1a1a1a; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: 0.5rem 0; font-size: 0.85rem; }
th, td { border: 1px solid #ccc; padding: 0.2rem 0.6rem; text-align: left; }
th { background: #f4f4f4; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #0a7d28; } .bad { color: #b3261e; }
code { font-size: 0.85em; }
p.note { color: #555; max-width: 90ch; }
"""


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "<table>",
        "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>",
        "<tbody>",
        *(
            ("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
            for row in rows
        ),
        "</tbody>",
        "</table>",
    ]


def _render_html(report: DiscoveryReport) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Discovery report</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<h1>Discovery report</h1>",
        '<section id="summary">',
        "<h2>Summary</h2>",
        *_table(
            ["records", "coordinates", "seeds", "families", "metric"],
            [
                [
                    str(report.n_records),
                    str(report.n_coordinates),
                    ", ".join(map(str, report.seeds)),
                    ", ".join(report.families),
                    escape(report.metric),
                ]
            ],
        ),
    ]
    if report.fidelity is not None:
        quarantined = (
            ", ".join(f"<code>{escape(c)}</code>" for c in report.fidelity.quarantined)
            or "—"
        )
        parts += _table(
            ["fidelity passing", "quarantined"],
            [
                [
                    str(report.fidelity.n_passing),
                    f"{report.fidelity.n_quarantined} ({quarantined})",
                ]
            ],
        )
        parts.append(
            '<p class="note">Quarantined coordinates are excluded from the '
            "frontier and attribution — inconclusive, never a refutation.</p>"
        )
    parts += [
        "</section>",
        '<section id="timeline">',
        "<h2>Timeline (per seed x iteration)</h2>",
        *(
            _table(
                ["seed", "iteration", "episodes", "mean loss", "mean accuracy"],
                [
                    [
                        str(s.seed),
                        str(s.iteration),
                        str(s.episodes),
                        _fmt(s.mean_loss),
                        _fmt(s.mean_accuracy),
                    ]
                    for s in report.timeline
                ],
            )
            if report.timeline
            else ["<p>No episodes.</p>"]
        ),
        "</section>",
        '<section id="frontier">',
        "<h2>Pareto frontier over the resource vector</h2>",
        f"<p>{report.n_frontier_records} frontier records · "
        f"{report.n_dominated} dominated · hypervolume {_fmt(report.hypervolume)}</p>",
        *(
            _table(
                [
                    "coordinate",
                    "owned axes",
                    "episodes",
                    *(f"{o} ↓" for o in report.objectives),
                ],
                [
                    [
                        f"<code>{escape(r.coordinate)}</code>",
                        ", ".join(r.owned_axes) or "—",
                        str(r.records),
                        *(_fmt(r.values[o]) for o in report.objectives),
                    ]
                    for r in report.frontier
                ],
            )
            if report.frontier
            else ["<p>No passing records — no frontier.</p>"]
        ),
        '<p class="note">Objectives minimize except '
        f"{escape(report.metric)}; computed only over fidelity-passing "
        "coordinates. Owned axes: values no other frontier coordinate "
        "shares — the axis whose trade-off position the knee is bought with.</p>",
        "</section>",
        '<section id="replication">',
        "<h2>Replication gate</h2>",
        *(
            _table(
                ["coordinate", "seeds", "families", "replicated", "unmet"],
                [
                    [
                        f"<code>{escape(r.coordinate)}</code>",
                        str(r.seeds),
                        str(r.families),
                        (
                            '<span class="ok">yes</span>'
                            if r.replicated
                            else '<span class="bad">no</span>'
                        ),
                        escape("; ".join(r.unmet)) or "—",
                    ]
                    for r in report.replication
                ],
            )
            if report.replication
            else ["<p>No records.</p>"]
        ),
        "</section>",
        '<section id="attribution">',
        "<h2>Counterfactual attribution (pooled, canonical)</h2>",
        *(
            _table(
                ["axis", "from → to", "mean Δ", "pairs"],
                [
                    [
                        escape(a.axis),
                        f"{escape(a.from_value)} → {escape(a.to_value)}",
                        _fmt(a.mean_delta),
                        str(a.n_pairs),
                    ]
                    for a in report.attribution
                ],
            )
            if report.attribution
            else ["<p>No minimal pairs within the passing subspace.</p>"]
        ),
    ]
    if report.stratified:
        parts += ["<h2>Stratified attribution (per seed x family)</h2>"]
        for stratum in report.stratified:
            parts += [f"<h3>{escape(stratum.stratum)}</h3>"]
            parts += (
                _table(
                    ["axis", "from → to", "mean Δ", "pairs"],
                    [
                        [
                            escape(a.axis),
                            f"{escape(a.from_value)} → {escape(a.to_value)}",
                            _fmt(a.mean_delta),
                            str(a.n_pairs),
                        ]
                        for a in stratum.rows
                    ],
                )
                if stratum.rows
                else ["<p>No pairs.</p>"]
            )
    parts += ["</section>", "</body>", "</html>", ""]
    return "\n".join(parts)
