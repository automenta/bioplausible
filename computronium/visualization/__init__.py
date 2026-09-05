"""Public API for the computronium visualization package (TODO10 R10.1).

The common demo API (2026-09-05): demo records DECLARE their figure
(``data["figure"]`` spec) and one generic renderer produces the gallery
figure — consistent labeling, chance lines, palettes, and value labels
by construction. ``figure_from_spec`` renders a spec standalone.
"""

from computronium.visualization._demo_api import (
    figure_from_spec,
    figure_spec,
    bars_panel,
    lines_panel,
    scatter_panel,
    heatmap_panel,
    graph_panel,
    tree_panel,
)
from computronium.visualization.gallery import (
    DEMOS,
    DemoSpec,
    FigureMeta,
    canonicalize_floats,
    render_gallery,
)

__all__ = [
    "DEMOS",
    "DemoSpec",
    "FigureMeta",
    "bars_panel",
    "canonicalize_floats",
    "figure_from_spec",
    "figure_spec",
    "graph_panel",
    "heatmap_panel",
    "lines_panel",
    "render_gallery",
    "scatter_panel",
    "tree_panel",
]
