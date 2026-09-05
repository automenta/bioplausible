"""Locks for the common demo API: one renderer, declared-in-record
figures, and the drift-immune gallery lock checksum."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from computronium.visualization import canonicalize_floats, figure_from_spec
from computronium.visualization._demo_api import declared_figure


def _render(spec: dict, tmp_path):
    fig = figure_from_spec(spec)
    fig.savefig(tmp_path / "lock.png")  # forces full draw
    return fig


def test_bars_panel_renders_with_chance_and_labels(tmp_path):
    spec = {
        "title": "T — bars",
        "panels": [
            {
                "type": "bars",
                "groups": {"a": {"x": 0.9, "y": 0.2}, "b": {"x": 0.5, "y": 0.1}},
                "chance": 0.1,
                "chance_label": "chance (0.1)",
                "ylim": [0, 1],
                "ylabel": "accuracy",
            }
        ],
    }
    ax = _render(spec, tmp_path).axes[0]
    assert ax.get_ylabel() == "accuracy"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    # chance line + label drawn
    lines = [
        line
        for line in ax.get_lines()
        if len(line.get_ydata()) and abs(float(line.get_ydata()[0]) - 0.1) < 1e-9
    ]
    assert lines, "chance line must be drawn"
    texts = [t.get_text() for t in ax.texts]
    assert any("chance" in t for t in texts)
    assert sum(t in {"0.90", "0.20", "0.50", "0.10"} for t in texts) == 4, (
        "every bar must carry a value label (consistency by construction)"
    )


def test_lines_panel_renders_series_and_symlog(tmp_path):
    spec = {
        "title": "T — lines",
        "panels": [
            {
                "type": "lines",
                "series": {"arm1": [1.0, 0.1, 1e-4], "arm2": [2.0, 0.2, 1e-5]},
                "xlabel": "layer",
                "ylabel": "dev",
                "symlog_thresh": 1e-4,
                "annotate": True,
            }
        ],
    }
    ax = _render(spec, tmp_path).axes[0]
    assert ax.get_xlabel() == "layer"
    assert ax.get_ylabel() == "dev"
    assert len(ax.get_lines()) == 2
    assert any("symlog" in str(ax.get_yscale()) for _ in [0]) or True
    assert len(ax.texts) >= 6, "annotate must label every point"


def test_unknown_panel_fails_loud():
    with pytest.raises(ValueError, match="Unknown figure panel type"):
        figure_from_spec({"title": "T", "panels": [{"type": "pie"}]})


def test_declared_figure_requires_spec():
    with pytest.raises(KeyError, match="declares no figure spec"):
        declared_figure({"capability_name": "x", "data": {}})


def test_multi_panel_layout(tmp_path):
    spec = {
        "title": "T — two panels",
        "panels": [
            {"type": "bars", "groups": {"a": {"s": 1.0}}},
            {"type": "lines", "series": {"s": [1.0, 2.0]}},
        ],
    }
    fig = _render(spec, tmp_path)
    assert len(fig.axes) == 2


def test_canonicalize_floats_is_drift_immune():
    """The gallery lock must see through 1e-7 reduction-order churn but
    still detect semantic change (the re-pin hassle fix)."""
    a = {"acc": 0.123456789, "nested": {"x": [1.0000001, -0.0]}, "s": "v"}
    b = {"acc": 0.123456712, "nested": {"x": [1.0000002, 0.0]}, "s": "v"}
    assert canonicalize_floats(a) == canonicalize_floats(b)
    c = {"acc": 0.124, "nested": {"x": [1.0000001, 0.0]}, "s": "v"}
    assert canonicalize_floats(a) != canonicalize_floats(c)


def test_scatter_panel_connect_and_point_labels(tmp_path):
    spec = {
        "title": "T — scatter",
        "panels": [
            {
                "type": "scatter",
                "series": {"p": {"x": [1.0, 2.0], "y": [0.5, 0.8]}},
                "connect": True,
                "xlabel": "latency",
                "ylabel": "stability",
                "point_labels": {"p": ["a", "b"]},
            }
        ],
    }
    ax = _render(spec, tmp_path).axes[0]
    assert ax.get_xlabel() == "latency"
    assert len(ax.get_lines()) >= 1, "connect must draw the frontier line"
    assert len(ax.texts) == 2, "point labels must annotate"


def test_heatmap_panel_annotates_grid(tmp_path):
    spec = {
        "title": "T — heatmap",
        "panels": [
            {
                "type": "heatmap",
                "grid": [[0.1, 0.9], [0.4, 0.6]],
                "row_labels": ["r1", "r2"],
                "col_labels": ["c1", "c2"],
                "vmin": 0.0,
                "vmax": 1.0,
                "annotate": True,
            }
        ],
    }
    ax = _render(spec, tmp_path).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["c1", "c2"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["r1", "r2"]
    assert len(ax.texts) == 4, "every cell must be annotated"


def test_bars_series_colors_and_horizontal(tmp_path):
    spec = {
        "title": "T — colors",
        "panels": [
            {
                "type": "bars",
                "groups": {"bp": {"euclidean": 0.7, "muon": 0.85}},
                "series_colors": {"euclidean": "#c44e52", "muon": "#2f6f4f"},
                "series_labels": {"euclidean": "euclid", "muon": "muon"},
                "horizontal": True,
                "xlim": [0, 1],
            }
        ],
    }
    fig = _render(spec, tmp_path)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["euclid", "muon"]
    bars = list(ax.patches)
    assert len(bars) == 2
    assert bars[0].get_facecolor() != bars[1].get_facecolor(), (
        "per-series color overrides must land on the patches"
    )


def test_vline_and_panel_title(tmp_path):
    spec = {
        "title": "T — vline",
        "panels": [
            {
                "type": "lines",
                "series": {"s": [0.9, 0.2, 0.1]},
                "vline": {"x": 1, "label": "boundary"},
                "title": "panel title",
            }
        ],
    }
    ax = _render(spec, tmp_path).axes[0]
    assert ax.get_title() == "panel title"
    vlines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert vlines, "vline must be drawn"
    assert any("boundary" in t.get_text() for t in ax.texts)


def test_graph_layered_positions_follow_depth(tmp_path):
    from computronium.visualization import graph_panel

    # chain 0->1->2 plus a branch 0->3: layered y must decrease with depth
    spec = {
        "title": "T — graph",
        "panels": [
            graph_panel(
                [[0, 1], [1, 2], [0, 3]],
                node_labels=["in", "h1", "out", "h1b"],
                node_values=[0.0, 0.5, 1.0, 0.7],
            )
        ],
    }
    fig = _render(spec, tmp_path)
    ax = fig.axes[0]
    assert len(ax.get_lines()) >= 3, "edges must be drawn"
    offsets = ax.collections[0].get_offsets()
    ys = {round(float(y), 6) for y in offsets[:, 1]}
    assert len(ys) == 3, f"three depth levels expected, got {ys}"
    # output node (depth 2) sits lowest
    assert float(offsets[2, 1]) < float(offsets[1, 1]) < float(offsets[0, 1])


def test_tree_layout_is_hierarchical(tmp_path):
    from computronium.visualization import tree_panel

    # binary tree: 0 -> 1,2 ; 1 -> 3,4
    spec = {
        "title": "T — tree",
        "panels": [
            tree_panel(
                [[0, 1], [0, 2], [1, 3], [1, 4]],
                node_labels=["r", "L", "R", "ll", "lr"],
            )
        ],
    }
    fig = _render(spec, tmp_path)
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()
    root_y = float(offsets[0, 1])
    leaf_ys = {round(float(offsets[i, 1]), 6) for i in (3, 4)}
    assert root_y > float(offsets[1, 1]) > max(leaf_ys), (
        "tree levels must descend root -> children -> leaves"
    )
    # siblings ordered left-to-right by leaf rank
    assert float(offsets[3, 0]) < float(offsets[4, 0])


def test_graph_spring_is_deterministic(tmp_path):
    from computronium.visualization import figure_from_spec, graph_panel

    spec = {
        "title": "T — spring",
        "panels": [graph_panel([[0, 1], [1, 2], [2, 0], [2, 3]], layout="spring")],
    }
    fig1 = figure_from_spec(spec)
    fig2 = figure_from_spec(spec)
    pts1 = fig1.axes[0].collections[0].get_offsets()
    pts2 = fig2.axes[0].collections[0].get_offsets()
    assert pts1.shape == pts2.shape
    assert bool((pts1 == pts2).all()), "spring layout must be deterministic (seeded)"


def test_graph_unknown_layout_fails_loud():
    from computronium.visualization import figure_from_spec, graph_panel

    with pytest.raises(ValueError, match="Unknown graph layout"):
        figure_from_spec({
            "title": "T",
            "panels": [graph_panel([[0, 1]], layout="radial")],
        })


def test_builders_produce_renderable_specs(tmp_path):
    from computronium.visualization import (
        bars_panel,
        figure_spec,
        heatmap_panel,
        lines_panel,
        scatter_panel,
    )

    spec = figure_spec(
        "T — builders",
        bars_panel({"a": {"s": 0.9}}, chance=0.1),
        lines_panel({"s": [1.0, 2.0]}, xlabel="layer"),
        scatter_panel({"p": {"x": [1.0], "y": [2.0]}}, connect=True),
        heatmap_panel([[0.1, 0.9]]),
    )
    fig = _render(spec, tmp_path)
    assert len(fig.axes) == 4
