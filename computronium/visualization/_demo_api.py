"""The common demo API: figures are DECLARED IN THE RECORD.

A demo test emits its measured data and — in the same dict — a
``figure`` spec describing how the gallery should render it. One generic
renderer turns the spec into a styled figure, so:

- labeling, chance lines, value labels, palettes, and layout are defined
  ONCE (consistency by construction — every demo declares its figure);
- a new demo gets a gallery figure for free by declaring panels next to
  the data it measured (the producer owns the presentation);
- the record stays the single JSON artifact — spec and data re-render
  together, and the gallery lock checksums both.

Spec schema (plain JSON, embedded under ``"figure"`` in the record data):

    {
      "title": "D14 — depth 20 ...",   # figure title (suptitle)
      "figsize": [7.5, 4.5],            # optional
      "layout": [2, 2],                 # optional grid, default 1 x n
      "panels": [
        {                               # COMMON panel keys
          "type": ...,                  # bars | lines | scatter | heatmap
          "title": "...",               # per-axes title (multi-panel figs)
          "chance": 0.1, "chance_label": "chance (0.1)",
          "legend_loc": "upper left",   # optional
        },
        ...
      ]
    }

Panel types:

- ``bars`` — grouped vertical or horizontal bars, one color per series:
    {"type": "bars", "groups": {"group": {"series": v, ...}, ...},
     "group_order": [...], "series_order": [...],
     "series_labels": {"train": "train"}, "series_colors": {"train": "#..."},
     "horizontal": false, "ylabel": "accuracy", "xlabel": "...",
     "ylim": [0, 1], "fmt": ".2f",
     "yerr": {"group": {"series": err, ...}}}   # symmetric error bars
- ``lines`` — one marker-line per series over a shared x:
    {"type": "lines", "series": {"arm": [y, ...], ...},
     "x": [...], "xticklabels": [...], "xlabel": "layer", "ylabel": "...",
     "log_y": false, "symlog_thresh": 1e-4, "annotate": true, "fmt": ".2f",
     "vline": {"x": 8, "label": "depth boundary"},
     "bands": {"arm": {"low": [...], "high": [...]}}}  # mean±spread fill
- ``scatter`` — points per series; the Pareto/frontier panel:
    {"type": "scatter", "series": {"arm": {"x": [...], "y": [...]}},
     "connect": true, "xlabel": "latency", "ylabel": "stability",
     "point_labels": {"arm": ["a(0,0)", ...]}, "fmt": ".2f"}
- ``heatmap`` — annotated grid (matrices, tile x dynamics tables):
    {"type": "heatmap", "grid": [[...], ...], "row_labels": [...],
     "col_labels": [...], "cmap": "viridis", "annotate": true,
     "fmt": ".2f", "vmin": 0.0, "vmax": 1.0, "colorbar": false}

Rendered by :func:`figure_from_spec`; embedded records render through
``gallery._fig_declared``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from computronium.visualization._style import (
    COLOR_ARM,
    COLOR_CONTRAST,
    COLOR_FEASIBLE,
    COLOR_WALLED,
    apply_style,
    chance_line,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

SERIES_PALETTE = (COLOR_ARM, COLOR_CONTRAST, COLOR_FEASIBLE, COLOR_WALLED)


@dataclass(frozen=True, slots=True)
class _Panel:
    """Common panel options."""

    title: str | None = None
    chance: float | None = None
    chance_label: str = "chance"
    legend_loc: str | None = None


@dataclass(frozen=True, slots=True)
class BarPanel(_Panel):
    """Grouped bars: one group per arm, one bar per series."""

    groups: dict[str, dict[str, float]] = field(default_factory=dict)
    group_order: tuple[str, ...] | None = None
    series_order: tuple[str, ...] | None = None
    series_labels: dict[str, str] | None = None
    series_colors: dict[str, str] | None = None
    horizontal: bool = False
    ylabel: str = "accuracy"
    xlabel: str = ""
    ylim: tuple[float, float] | None = None
    xlim: tuple[float, float] | None = None
    yerr: dict[str, dict[str, float]] | None = None
    fmt: str = ".2f"


@dataclass(frozen=True, slots=True)
class LinePanel(_Panel):
    """One marker-line per series over a shared x."""

    series: dict[str, list[float]] = field(default_factory=dict)
    x: list[float] | None = None
    xticklabels: tuple[str, ...] | None = None
    xlabel: str = ""
    ylabel: str = ""
    log_y: bool = False
    symlog_thresh: float | None = None
    vline: dict[str, float | str] | None = None
    bands: dict[str, dict[str, list[float]]] | None = None
    annotate: bool = False
    fmt: str = ".2f"


@dataclass(frozen=True, slots=True)
class ScatterPanel(_Panel):
    """Points per series; the Pareto-frontier / trade-off panel."""

    series: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    connect: bool = False
    xlabel: str = ""
    ylabel: str = ""
    point_labels: dict[str, list[str]] | None = None
    fmt: str = ".2f"


@dataclass(frozen=True, slots=True)
class HeatmapPanel(_Panel):
    """Annotated value grid (matrices, capability tables)."""

    grid: list[list[float]] = field(default_factory=list)
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    annotate: bool = True
    colorbar: bool = False
    fmt: str = ".2f"


def figure_from_spec(spec: dict) -> Figure:
    """Render a declared figure spec (the common demo API)."""
    import matplotlib.pyplot as plt

    panels = [_panel_from_dict(p) for p in spec["panels"]]
    rows, cols = spec.get("layout", [1, len(panels)])
    figsize = tuple(spec.get("figsize", (6.5 * cols, 4.5 * rows)))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(panels) :]:
        ax.axis("off")
    for ax, panel in zip(flat, panels, strict=False):
        match panel:
            case BarPanel():
                _render_bars(ax, panel)
            case LinePanel():
                _render_lines(ax, panel)
            case ScatterPanel():
                _render_scatter(ax, panel)
            case HeatmapPanel():
                _render_heatmap(fig, ax, panel)
            case GraphPanel():
                _render_graph(fig, ax, panel)
        if panel.title:
            ax.set_title(panel.title, fontsize=10)
    fig.suptitle(spec["title"], fontsize=11)
    apply_style(fig)
    return fig


def _panel_from_dict(d: dict) -> BarPanel | LinePanel | ScatterPanel | HeatmapPanel:
    try:
        panel_type = d["type"]
    except KeyError as err:
        raise ValueError(_missing_type_msg(d)) from err
    common = {
        "title": d.get("title"),
        "chance": d.get("chance"),
        "chance_label": d.get("chance_label", "chance"),
        "legend_loc": d.get("legend_loc"),
    }
    match panel_type:
        case "bars":
            return BarPanel(
                groups=d["groups"],
                group_order=_tuple(d.get("group_order")),
                series_order=_tuple(d.get("series_order")),
                series_labels=d.get("series_labels"),
                series_colors=d.get("series_colors"),
                horizontal=d.get("horizontal", False),
                ylabel=d.get("ylabel", "accuracy"),
                xlabel=d.get("xlabel", ""),
                ylim=_tuple(d.get("ylim")),
                xlim=_tuple(d.get("xlim")),
                yerr=d.get("yerr"),
                fmt=d.get("fmt", ".2f"),
                **common,
            )
        case "lines":
            return LinePanel(
                series=d["series"],
                x=d.get("x"),
                xticklabels=_tuple(d.get("xticklabels")),
                xlabel=d.get("xlabel", ""),
                ylabel=d.get("ylabel", ""),
                log_y=d.get("log_y", False),
                symlog_thresh=d.get("symlog_thresh"),
                vline=d.get("vline"),
                bands=d.get("bands"),
                annotate=d.get("annotate", False),
                fmt=d.get("fmt", ".2f"),
                **common,
            )
        case "scatter":
            return ScatterPanel(
                series=d["series"],
                connect=d.get("connect", False),
                xlabel=d.get("xlabel", ""),
                ylabel=d.get("ylabel", ""),
                point_labels=d.get("point_labels"),
                fmt=d.get("fmt", ".2f"),
                **common,
            )
        case "heatmap":
            return HeatmapPanel(
                grid=d["grid"],
                row_labels=d.get("row_labels"),
                col_labels=d.get("col_labels"),
                cmap=d.get("cmap", "viridis"),
                vmin=d.get("vmin"),
                vmax=d.get("vmax"),
                annotate=d.get("annotate", True),
                colorbar=d.get("colorbar", False),
                fmt=d.get("fmt", ".2f"),
                **common,
            )
        case "graph":
            return GraphPanel(
                edges=d["edges"],
                n_nodes=d.get("n_nodes", 0),
                node_labels=d.get("node_labels"),
                node_values=d.get("node_values"),
                node_sizes=d.get("node_sizes"),
                edge_weights=d.get("edge_weights"),
                layout=d.get("layout", "layered"),
                root=d.get("root", 0),
                cmap=d.get("cmap", "viridis"),
                show_labels=d.get("show_labels", True),
                directed=d.get("directed", True),
                **common,
            )
        case _:
            raise ValueError(_unknown_spec_msg(panel_type))


def _unknown_spec_msg(panel_type: object) -> str:
    return (
        f"Unknown figure panel type: {panel_type!r} — expected one of "
        f"{sorted(PANEL_TYPES)}"
    )


def _tuple(value) -> tuple | None:
    return tuple(value) if value is not None else None


def _series_color(panel: BarPanel, series: str, idx: int) -> str:
    override = panel.series_colors or {}
    return override.get(series, SERIES_PALETTE[idx % len(SERIES_PALETTE)])


def _render_bars(ax: Axes, panel: BarPanel) -> None:
    group_order = panel.group_order or tuple(panel.groups)
    series_order = panel.series_order or tuple(next(iter(panel.groups.values())))
    width = 0.8 / max(len(series_order), 1)
    yerr = panel.yerr or {}

    for s_idx, series in enumerate(series_order):
        values = [panel.groups[g].get(series, 0.0) for g in group_order]
        errors = [yerr.get(g, {}).get(series, 0.0) for g in group_order]
        offsets = [i + s_idx * width - 0.4 + width / 2 for i in range(len(group_order))]
        _draw_bar_series(
            ax, panel, series, s_idx, group_order, offsets, values, errors, width
        )

    if panel.chance is not None:
        chance_line(ax, panel.chance, panel.chance_label)
    if panel.horizontal:
        if panel.xlabel:
            ax.set_xlabel(panel.xlabel)
    elif panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    if len(series_order) > 1:
        ax.legend(loc=panel.legend_loc or "best")


def _draw_bar_series(
    ax: Axes,
    panel: BarPanel,
    series: str,
    s_idx: int,
    group_order: tuple[str, ...],
    offsets: list[float],
    values: list[float],
    errors: list[float],
    width: float,
) -> None:
    color = _series_color(panel, series, s_idx)
    label = (panel.series_labels or {}).get(series, series)
    if panel.horizontal:
        ax.barh(offsets, values, width, color=color, label=label, xerr=errors)
        for y, v in zip(offsets, values, strict=True):
            ax.text(v, y, f"{v:{panel.fmt}}", va="center", fontsize=8)
        ax.set_yticks(range(len(group_order)), group_order)
        if panel.xlim:
            ax.set_xlim(*panel.xlim)
    else:
        ax.bar(offsets, values, width, color=color, label=label, yerr=errors)
        for x, v in zip(offsets, values, strict=True):
            ax.text(x, v, f"{v:{panel.fmt}}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(group_order)), group_order)
        if panel.ylim:
            ax.set_ylim(*panel.ylim)


def _render_lines(ax: Axes, panel: LinePanel) -> None:
    bands = panel.bands or {}
    for s_idx, (name, ys) in enumerate(panel.series.items()):
        _draw_line_series(ax, panel, name, ys, s_idx, bands.get(name))
    if panel.chance is not None:
        chance_line(ax, panel.chance, panel.chance_label)
    if panel.vline is not None:
        ax.axvline(
            float(panel.vline["x"]),
            color="grey",
            linestyle="--",
            linewidth=1,
        )
        ax.text(
            float(panel.vline["x"]),
            ax.get_ylim()[1],
            f" {panel.vline.get('label', '')}",
            fontsize=8,
            color="grey",
            va="top",
        )
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    if panel.log_y:
        ax.set_yscale("log")
    elif panel.symlog_thresh is not None:
        ax.set_yscale("symlog", linthresh=panel.symlog_thresh)
    if panel.xticklabels is not None:
        ax.set_xticks(range(len(panel.xticklabels)), panel.xticklabels)
    if len(panel.series) > 1:
        ax.legend(loc=panel.legend_loc or "best")


def _draw_line_series(
    ax: Axes,
    panel: LinePanel,
    name: str,
    ys: list[float],
    s_idx: int,
    band: dict[str, list[float]] | None,
) -> None:
    xs = panel.x if panel.x is not None else list(range(len(ys)))
    color = SERIES_PALETTE[s_idx % len(SERIES_PALETTE)]
    if band:
        ax.fill_between(
            xs,
            band["low"],
            band["high"],
            color=color,
            alpha=0.2,
            linewidth=0,
        )
    ax.plot(xs, ys, marker="o", color=color, label=name)
    if panel.annotate:
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(
                f"{y:{panel.fmt}}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
            )


def _render_scatter(ax: Axes, panel: ScatterPanel) -> None:
    for s_idx, (name, xy) in enumerate(panel.series.items()):
        color = SERIES_PALETTE[s_idx % len(SERIES_PALETTE)]
        ax.scatter(xy["x"], xy["y"], color=color, label=name, zorder=2)
        if panel.connect:
            ax.plot(xy["x"], xy["y"], color=color, linewidth=1, alpha=0.6)
        point_labels = (panel.point_labels or {}).get(name)
        if point_labels:
            for x, y, label in zip(xy["x"], xy["y"], point_labels, strict=True):
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=7,
                )
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    if len(panel.series) > 1:
        ax.legend(loc=panel.legend_loc or "best")


def _render_heatmap(fig: Figure, ax: Axes, panel: HeatmapPanel) -> None:
    import numpy as np

    data = np.array(panel.grid, dtype=float)
    image = ax.imshow(
        data,
        cmap=panel.cmap,
        vmin=panel.vmin,
        vmax=panel.vmax,
        aspect="auto",
    )
    if panel.row_labels:
        ax.set_yticks(range(len(panel.row_labels)), panel.row_labels)
    if panel.col_labels:
        ax.set_xticks(range(len(panel.col_labels)), panel.col_labels)
    if panel.annotate:
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:{panel.fmt}}", ha="center", va="center", fontsize=8)
    if panel.colorbar:
        fig.colorbar(image, ax=ax)


def declared_figure(record: dict) -> Figure:
    """Render the figure declared inside a run record's data."""
    spec = record["data"].get("figure")
    if spec is None:
        raise KeyError(_undeclared_msg(record))
    return figure_from_spec(spec)


def _undeclared_msg(record: dict) -> str:
    return (
        f"record {record.get('capability_name')!r} declares no figure spec "
        "— either declare one under data['figure'] or register a bespoke factory"
    )


def _missing_type_msg(d: dict) -> str:
    return f"Panel spec missing 'type': {d!r}"


# ============================================================
# Structural panels: graphs and trees
# ============================================================


@dataclass(frozen=True, slots=True)
class GraphPanel(_Panel):
    """Nodes + edges from an adjacency structure (GraphGeometry,
    TileMesh, attention patterns, depth-metric layouts).

    ``layout``: "layered" (y by longest-path depth from sources — the
    R11.3.13 depth notion), "tree" (hierarchical from ``root``: level =
    distance from root, x by leaf order), "circle", "spring"
    (deterministic force-directed). ``node_values`` colors nodes by a
    metric (e.g. per-node depth, activation norm); ``edge_weights``
    scale edge line widths."""

    edges: list[list[int]] = field(default_factory=list)
    n_nodes: int = 0
    node_labels: list[str] | None = None
    node_values: list[float] | None = None
    node_sizes: list[float] | None = None
    edge_weights: list[float] | None = None
    layout: str = "layered"
    root: int = 0
    cmap: str = "viridis"
    show_labels: bool = True
    directed: bool = True


def graph_panel(
    edges: list[list[int]],
    n_nodes: int | None = None,
    *,
    layout: str = "layered",
    **kwargs,
) -> dict:
    """Ergonomic builder: declare a graph/network panel."""
    n = (
        n_nodes
        if n_nodes is not None
        else (1 + max((max(e) for e in edges), default=0))
    )
    return {"type": "graph", "edges": edges, "n_nodes": n, "layout": layout, **kwargs}


def tree_panel(edges: list[list[int]], root: int = 0, **kwargs) -> dict:
    """Ergonomic builder: a hierarchy is a graph with a tree layout."""
    return graph_panel(edges, layout="tree", root=root, **kwargs)


def _graph_positions(panel: GraphPanel) -> list[tuple[float, float]]:
    import numpy as np

    n = max(panel.n_nodes, 1 + max((v for e in panel.edges for v in e), default=0))
    if panel.layout == "circle":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return list(zip(np.cos(angles), np.sin(angles)))
    if panel.layout == "spring":
        return _spring_positions(panel.edges, n)
    if panel.layout == "tree":
        return _tree_positions(panel.edges, n, panel.root)
    if panel.layout == "layered":
        return _layered_positions(panel.edges, n)
    raise ValueError(_unknown_layout_msg(panel.layout))


def _adjacency(edges: list[list[int]], n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def _depths(edges: list[list[int]], n: int, sources: list[int]) -> list[int]:
    """Longest-path depth from sources (BFS relaxation, DAG-safe; cycles
    saturate at n). Same notion as R11.3.13's LongestPathDepth."""
    adj = _adjacency(edges, n)
    depth = [-1] * n
    frontier = [(s, 0) for s in sources]
    for s, _ in frontier:
        depth[s] = 0
    while frontier:
        nxt = []
        for u, d in frontier:
            for v in adj[u]:
                if depth[v] < d + 1:
                    depth[v] = d + 1
                    nxt.append((v, depth[v]))
        frontier = nxt
    return [max(d, 0) for d in depth]


def _layered_positions(edges: list[list[int]], n: int) -> list[tuple[float, float]]:
    depth = _depths(
        edges, n, sources=list(range(n)) if not edges else _sources(edges, n)
    )
    levels: dict[int, list[int]] = {}
    for node, d in enumerate(depth):
        levels.setdefault(d, []).append(node)
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * n
    for d, nodes in levels.items():
        span = max(len(nodes) - 1, 1)
        for k, node in enumerate(nodes):
            positions[node] = (k / span if len(nodes) > 1 else 0.5, -float(d))
    return positions


def _sources(edges: list[list[int]], n: int) -> list[int]:
    has_in = {v for _, v in edges}
    return [u for u in range(n) if u not in has_in] or [0]


def _tree_positions(
    edges: list[list[int]], n: int, root: int
) -> list[tuple[float, float]]:
    adj = _adjacency(edges, n)
    depth = _depths(edges, n, [root])
    # in-order leaf counting: x = (leaf_rank / total_leaves), internal
    # nodes sit at the mean of their children (falls back to level
    # ordering when the "tree" is a DAG with shared children).
    leaf_rank: dict[int, float] = {}
    counter = [0.0]
    total = [1.0]

    def count(node: int, seen: set[int]) -> tuple[bool, float]:
        if node in seen:
            return (False, 0.0)
        seen.add(node)
        children = [c for c in adj[node] if c not in seen]
        if not children:
            leaf_rank[node] = counter[0]
            counter[0] += 1.0
            return (True, leaf_rank[node])
        vals = [count(c, seen)[1] for c in children]
        leaf_rank[node] = sum(vals) / len(vals)
        return (True, leaf_rank[node])

    seen: set[int] = set()
    for node in range(n):
        if depth[node] >= 0 and node not in seen:
            count(node, seen)
    total = max(counter[0], 1.0)
    return [
        (leaf_rank.get(i, i / max(n, 1)) / total, -float(d))
        for i, d in enumerate(depth)
    ]


def _spring_positions(
    edges: list[list[int]], n: int, iterations: int = 200
) -> list[tuple[float, float]]:
    """Deterministic Fruchterman-Reingold-lite: seeded ring init, repulse
    all pairs, attract along edges."""
    import numpy as np

    rng = np.random.default_rng(0)
    pos = np.array([
        (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)
    ]) * (1 + (rng.random(n) * 0.1)[:, None])
    k = 1.0 / max(n**0.5, 1.0)
    for _ in range(iterations):
        delta = np.zeros((n, 2))
        for i in range(n):
            diff = pos[i] - pos
            dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-6)
            delta[i] += (diff / dist[:, None] * (dist**2 / k)[:, None]).sum(0)
        for u, v in edges:
            diff = pos[u] - pos[v]
            dist = max(float(np.linalg.norm(diff)), 1e-6)
            force = (dist - k) * diff / dist
            delta[u] -= force
            delta[v] += force
        pos += 0.01 * delta / (np.abs(delta).max() + 1e-12)
        pos -= pos.mean(0)
    return list(map(tuple, pos))


def _render_graph(fig: Figure, ax: Axes, panel: GraphPanel) -> None:
    from matplotlib.lines import Line2D

    positions = _graph_positions(panel)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    if panel.edge_weights:
        max_w = max(panel.edge_weights)
        for (u, v), w in zip(panel.edges, panel.edge_weights, strict=True):
            ax.add_line(
                Line2D(
                    [xs[u], xs[v]],
                    [ys[u], ys[v]],
                    linewidth=0.5 + 2.5 * w / max(max_w, 1e-9),
                    color="grey",
                    zorder=1,
                )
            )
    else:
        for u, v in panel.edges:
            ax.add_line(Line2D([xs[u], xs[v]], [ys[u], ys[v]], color="grey", zorder=1))
    sizes = panel.node_sizes or [200.0] * panel.n_nodes
    if panel.node_values is not None:
        scatter = ax.scatter(
            xs,
            ys,
            s=sizes,
            c=panel.node_values,
            cmap=panel.cmap,
            zorder=2,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.7)
    else:
        ax.scatter(xs, ys, s=sizes, color=COLOR_ARM, zorder=2)
    if panel.show_labels and panel.node_labels:
        for x, y, label in zip(xs, ys, panel.node_labels, strict=True):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
    if panel.directed:
        for u, v in panel.edges:
            ax.annotate(
                "",
                xy=(xs[v], ys[v]),
                xytext=(xs[u], ys[u]),
                arrowprops={"arrowstyle": "-|>", "color": "grey", "lw": 0.5},
                zorder=1,
            )
    ax.set_axis_off()


def _unknown_layout_msg(layout: str) -> str:
    return (
        f"Unknown graph layout: {layout!r} — expected one of "
        "['layered', 'tree', 'circle', 'spring']"
    )


PANEL_TYPES: dict[str, type] = {
    "bars": BarPanel,
    "lines": LinePanel,
    "scatter": ScatterPanel,
    "heatmap": HeatmapPanel,
    "graph": GraphPanel,
}


# ============================================================
# Ergonomic builders — the supported way to author specs from a demo
# ============================================================


def figure_spec(title: str, *panels: dict, figsize: list[float] | None = None) -> dict:
    """Assemble a record ``figure`` spec from panel builders."""
    spec: dict = {"title": title, "panels": list(panels)}
    if figsize is not None:
        spec["figsize"] = figsize
    return spec


def bars_panel(
    groups: dict[str, dict[str, float]],
    *,
    chance: float | None = None,
    chance_label: str = "chance",
    ylabel: str = "accuracy",
    yerr: dict[str, dict[str, float]] | None = None,
    **kwargs,
) -> dict:
    spec = {
        "type": "bars",
        "groups": groups,
        "chance": chance,
        "chance_label": chance_label,
        "ylabel": ylabel,
        **kwargs,
    }
    if yerr is not None:
        spec["yerr"] = yerr
    return spec


def lines_panel(
    series: dict[str, list[float]],
    *,
    chance: float | None = None,
    xlabel: str = "",
    ylabel: str = "",
    bands: dict[str, dict[str, list[float]]] | None = None,
    **kwargs,
) -> dict:
    spec = {
        "type": "lines",
        "series": series,
        "chance": chance,
        "xlabel": xlabel,
        "ylabel": ylabel,
        **kwargs,
    }
    if bands is not None:
        spec["bands"] = bands
    return spec


def scatter_panel(
    series: dict[str, dict[str, list[float]]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    **kwargs,
) -> dict:
    return {
        "type": "scatter",
        "series": series,
        "xlabel": xlabel,
        "ylabel": ylabel,
        **kwargs,
    }


def heatmap_panel(grid: list[list[float]], **kwargs) -> dict:
    return {"type": "heatmap", "grid": grid, **kwargs}
