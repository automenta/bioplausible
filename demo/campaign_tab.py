"""Live Campaign tab (R5b-F Stage 2) for the NiceGUI demo.

Renders the R5b-F Stage 1 :class:`DiscoveryReport` — summary, per-(seed,
iteration) timeline, defect-filtered Pareto frontier over the resource
vector, replication gate, and canonical counterfactual attribution — from
commissioned artifacts (``records/episodes.json`` + stored fidelity
manifest, which carries the R5b-0 quarantine) or, for in-progress runs,
from a live ``CampaignStack`` over the directory's SQLite DB. A timer
re-renders when the source changes on disk, so a campaign running in
another terminal appears here as it progresses.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from computronium.core.campaign.report import (
    DiscoveryReport,
    build_discovery_report,
    load_campaign_records,
)

CAMPAIGNS_ROOT = Path(__file__).resolve().parents[1] / "autoscientist_campaigns"

POLL_SECONDS = 3.0

_ATTRIBUTION_ROWS = 8


def _episodes_path(directory: Path) -> Path:
    return directory / "records" / "episodes.json"


def _db_path(directory: Path) -> Path:
    return directory / "campaign.db"


def discover_campaigns(root: Path = CAMPAIGNS_ROOT) -> list[str]:
    """Campaign directories under ``root`` (commissioned or live)."""
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
        and (_episodes_path(entry).exists() or _db_path(entry).exists())
    )


def _signature(directory: Path) -> tuple[int, int] | None:
    """On-disk change signature (mtime, size) of a campaign's record store."""
    for path in (_episodes_path(directory), _db_path(directory)):
        if path.exists():
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)
    return None


def load_report(source: str, root: Path = CAMPAIGNS_ROOT) -> DiscoveryReport:
    """Build the discovery report for one campaign source.

    Commissioned artifacts take precedence — only they carry the stored
    fidelity manifest; the live stack covers campaigns without artifacts.
    """
    directory = root / source
    if _episodes_path(directory).exists():
        records, fidelity = load_campaign_records(directory)
        return build_discovery_report(records, fidelity=fidelity)
    from computronium.core.campaign.stack import CampaignStack

    return CampaignStack(directory).discovery_report()


def _stat_card(label: str, value: str) -> None:
    with ui.card().classes("items-center py-2"):
        ui.label(value).classes("text-lg font-bold")
        ui.label(label).classes("text-xs text-grey")


def _timeline_chart(report: DiscoveryReport) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for seed in report.seeds:
        steps = [s for s in report.timeline if s.seed == seed]
        fig.add_trace(
            go.Scatter(
                x=[s.iteration for s in steps],
                y=[s.mean_accuracy for s in steps],
                mode="lines+markers",
                name=f"seed {seed}",
            )
        )
    fig.update_layout(
        height=280,
        margin={"l": 40, "r": 20, "t": 20, "b": 30},
        xaxis_title="iteration",
        yaxis_title="mean accuracy",
    )
    ui.plotly(fig)


def _section_table(
    rows: list[dict], title: str, *, pagination: int | None = None
) -> None:
    if not rows:
        ui.label(f"{title}: none.").classes("text-grey")
        return
    ui.label(title).classes("text-bold")
    ui.table(rows=rows, pagination=pagination)


def render_report(report: DiscoveryReport) -> None:
    """Render every discovery-report section into the current container."""
    with ui.row().classes("w-full flex-wrap"):
        _stat_card("records", str(report.n_records))
        _stat_card("coordinates", str(report.n_coordinates))
        _stat_card("seeds", ", ".join(map(str, report.seeds)) or "—")
        _stat_card("families", ", ".join(report.families) or "—")
        _stat_card("frontier", str(len(report.frontier)))
        _stat_card("hypervolume", f"{report.hypervolume:.2f}")
        if report.fidelity is not None:
            _stat_card(
                "fidelity gate",
                f"{report.fidelity.n_passing} pass / "
                f"{report.fidelity.n_quarantined} quarantined",
            )
    if report.fidelity is not None and report.fidelity.quarantined:
        ui.label(
            "Quarantined (inconclusive, excluded from frontier/attribution): "
            + ", ".join(report.fidelity.quarantined)
        ).classes("text-grey text-xs")
    _timeline_chart(report)
    _section_table(
        [
            {
                "coordinate": row.coordinate,
                "owned axes": ", ".join(row.owned_axes) or "—",
                "episodes": row.records,
                **{
                    objective: round(row.values[objective], 4)
                    for objective in report.objectives
                },
            }
            for row in report.frontier
        ],
        "Pareto frontier over the resource vector",
        pagination=10,
    )
    unreplicated = [row for row in report.replication if not row.replicated]
    ui.label(
        f"Replication gate: {len(report.replication) - len(unreplicated)}"
        f"/{len(report.replication)} replicated"
    ).classes("text-bold")
    _section_table(
        [
            {
                "coordinate": row.coordinate,
                "seeds": row.seeds,
                "families": row.families,
                "unmet": "; ".join(row.unmet),
            }
            for row in unreplicated
        ],
        "Unreplicated coordinates",
    )
    _section_table(
        [
            {
                "axis": attribution.axis,
                "from → to": f"{attribution.from_value} → {attribution.to_value}",
                "mean Δ": round(attribution.mean_delta, 5),
                "pairs": attribution.n_pairs,
            }
            for attribution in report.attribution[:_ATTRIBUTION_ROWS]
        ],
        f"Counterfactual attribution (pooled, canonical; top {_ATTRIBUTION_ROWS})",
    )


def build_campaign_tab(container: ui.column, root: Path = CAMPAIGNS_ROOT) -> None:
    """Build the live campaign view inside ``container`` (Campaign mode)."""
    sources = discover_campaigns(root)
    with container:
        if not sources:
            ui.label(f"No campaigns found under {root}").classes("text-grey")
            return
        report_box = ui.column().classes("w-full")
        status = ui.label("").classes("text-grey text-xs")
        rendered: list[tuple[int, int] | None] = [None]

        def refresh() -> None:
            report_box.clear()
            rendered[0] = None
            with report_box:
                try:
                    report = load_report(str(source_select.value), root)
                except Exception as error:
                    ui.label(f"Failed to load campaign: {error}").classes("text-red")
                    return
                render_report(report)
            rendered[0] = _signature(root / str(source_select.value))
            status.set_text(
                f"{source_select.value}: rendered {report.n_records} records"
            )

        def poll() -> None:
            stamp = _signature(root / str(source_select.value))
            if stamp is not None and stamp != rendered[0]:
                refresh()

        source_select = ui.select(
            sources, value=sources[0], label="Campaign", on_change=refresh
        ).classes("w-96")
        ui.button("Refresh", on_click=refresh)

        refresh()
        ui.timer(POLL_SECONDS, poll)
