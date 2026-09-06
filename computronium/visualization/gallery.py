"""The Gallery: figures rendered from the demo suite's own deterministic
run records at HEAD (TODO10 R10.1).

Every figure is what a demo test shows, drawn: fixed seeds, CPU, current
code. Every demo declares its figure spec inside its run record
(``data["figure"]``) and one generic renderer (``_fig_declared``) turns
the spec into a styled figure — the producer owns the presentation
(``_demo_api.py``). Nothing frozen,
nothing to re-verify — the figure lock (R10.1.4) regenerates each figure and
compares data-layer checksums so the gallery cannot silently drift from what
the code actually demonstrates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.visualization._style import save

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from matplotlib.figure import Figure

SCOPE_LABEL = "live demo scale (HEAD, CPU, fixed seeds)"
RECORDS_DIRNAME = "run_records"


@dataclass(frozen=True, slots=True)
class FigureMeta:
    """Provenance and scope of one gallery figure."""

    capability_id: str
    capability_name: str
    demo_test: str
    provenance: dict[str, str]
    scope_label: str
    data_sha256: str
    figure_png: str


def _records(records_dir: Path) -> Iterator[dict]:
    for path in sorted(records_dir.glob("*.json")):
        record: dict = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = path
        yield record


def canonicalize_floats(value: object, *, digits: int = 6) -> object:
    """Round every float in a JSON-like structure to ``digits`` decimals.

    The gallery lock must detect a demo CHANGING WHAT IT DEMONSTRATES,
    not multithreaded CPU reduction order — measured record drift sits at
    the 1e-7 level run-to-run while semantic changes move at 1e-3+.
    Hash the canonical projection, not the raw bytes."""
    if isinstance(value, float):
        rounded = round(value, digits)
        return 0.0 if rounded == 0 else rounded  # normalize -0.0
    if isinstance(value, dict):
        return {k: canonicalize_floats(v, digits=digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize_floats(v, digits=digits) for v in value]
    return value


def _sha256_data(record: dict) -> str:
    payload = json.dumps(
        canonicalize_floats(record["data"]), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _fig_declared(record: dict) -> Figure:
    """Generic renderer for records declaring ``data["figure"]`` — the
    common demo API. Every demo declares its panels in the record; the
    producer owns the presentation."""
    from computronium.visualization._demo_api import declared_figure

    return declared_figure(record)


@dataclass(frozen=True, slots=True)
class DemoSpec:
    """One gallery demo: capability id and its figure factory (R1.3)."""

    capability_id: str
    factory: Callable[[dict], Figure]


DEMOS: dict[str, DemoSpec] = {
    "compose_6axis": DemoSpec("D1", _fig_declared),
    "swap_credit": DemoSpec("D2", _fig_declared),
    "swap_plasticity": DemoSpec("D3", _fig_declared),
    "memory_budget": DemoSpec("D4", _fig_declared),
    "substrate_swap": DemoSpec("D6", _fig_declared),
    "spike_settle": DemoSpec("D7", _fig_declared),
    "z3_frozen_theta": DemoSpec("D5", _fig_declared),
    "geometry_swap": DemoSpec("D8", _fig_declared),
    "graph_geometry_swap": DemoSpec("D9", _fig_declared),
    "attention_geometry_swap": DemoSpec("D10", _fig_declared),
    "spatial_lattice_geometry_swap": DemoSpec("D11", _fig_declared),
    "epc_fast_settle": DemoSpec("D12", _fig_declared),
    "failure_manifesto": DemoSpec("F1", _fig_declared),
    "spiking_plateau": DemoSpec("F2", _fig_declared),
    "uaxis_muon_swap": DemoSpec("D13", _fig_declared),
    "jpc_faithful_depth": DemoSpec("D14", _fig_declared),
    "uaxis_depth_frontier": DemoSpec("D15", _fig_declared),
    "uaxis_coverage": DemoSpec("D16", _fig_declared),
    "paxis_pareto": DemoSpec("F3", _fig_declared),
    "update_ladder": DemoSpec("D18", _fig_declared),
    "credit_channel_map": DemoSpec("F4", _fig_declared),
}


def render_gallery(records_dir: Path, out_dir: Path) -> list[FigureMeta]:
    """Render one figure per demo run record and write the manifest.

    A record whose demo test no longer exists produces no figure — no
    orphaned claims (R10.3.2). Returns the rendered figures' metadata.
    """
    metas: list[FigureMeta] = []
    for record in _records(records_dir):
        capability_name = record["capability_name"]
        factory = DEMOS.get(capability_name)
        if factory is None or not Path(record["demo_test"]).exists():
            continue
        fig = factory.factory(record)
        png = out_dir / f"{record['capability'].lower()}_{capability_name}.png"
        save(fig, png)
        plt_close(fig)
        metas.append(
            FigureMeta(
                capability_id=record["capability"],
                capability_name=capability_name,
                demo_test=record["demo_test"],
                provenance=record["provenance"],
                scope_label=SCOPE_LABEL,
                data_sha256=_sha256_data(record),
                figure_png=png.name,
            )
        )
    manifest = {"figures": [asdict(m) for m in metas]}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metas


def plt_close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


__all__ = ["FigureMeta", "render_gallery"]
