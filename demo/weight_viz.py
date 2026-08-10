"""Animated weight-matrix visualization (Sprint 3.5).

Two layers mirror the rest of the demo:

1. **Pure transforms** (browser-free and unit-testable) that read a panel's
   ``weight_history`` (list of per-layer matrices captured per step by
   :class:`runner._WeightProbe`) and produce ready-to-plot data: which layers
   were captured, a 0..1 heatmap grid for one snapshot, and an A-minus-B diff
   between two panels' histories.

2. A thin NiceGUI widget (:class:`WeightMatrixAnimator`) that renders the
   current snapshot as a Plotly heatmap with a play/pause toggle and a scrub
   slider, optionally showing the Config A - Config B diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runner import DemoPanel


def weight_layers(panel: DemoPanel) -> list[str]:
    """Layer names captured in a panel's weight history (stable order)."""
    return list(panel.weight_history.keys())


@dataclass
class MatrixFrame:
    """A single normalized heatmap frame for one layer."""

    layer: str
    frame: int
    rows: int
    cols: int
    # Flat list of 0..1 intensity values (row-major) safe for JSON/browser.
    values: list[float] = field(default_factory=list)

    def as_grid(self, rows: int, cols: int) -> list[list[float]]:
        """Reshape ``values`` into a ``rows x cols`` list-of-lists for Plotly."""
        vals = self.values[: rows * cols]
        pad = rows * cols - len(vals)
        if pad:
            vals = vals + [0.0] * pad
        return [vals[r * cols : (r + 1) * cols] for r in range(rows)]


def _normalize(tensor) -> tuple[list[float], int, int]:
    """Min-max normalize a weight matrix to 0..1; return (values, rows, cols)."""

    m = tensor.float()
    rows, cols = m.shape
    lo = float(m.min())
    hi = float(m.max())
    span = (hi - lo) or 1.0
    vals = ((m - lo) / span).view(-1).tolist()
    return vals, rows, cols


def matrix_frame(panel: DemoPanel, layer: str, frame: int) -> MatrixFrame | None:
    """Return the normalized heatmap for one (layer, frame) of a panel."""
    snaps = panel.weight_history.get(layer)
    if not snaps:
        return None
    frame = max(0, min(frame, len(snaps) - 1))
    vals, rows, cols = _normalize(snaps[frame])
    return MatrixFrame(layer=layer, frame=frame, rows=rows, cols=cols, values=vals)


def align_length(a: DemoPanel, b: DemoPanel, layer: str) -> int:
    """Number of available diff frames across both panels for ``layer``."""
    return min(
        len(a.weight_history.get(layer, [])), len(b.weight_history.get(layer, []))
    )


def diff_frame(
    a: DemoPanel, b: DemoPanel, layer: str, frame: int
) -> MatrixFrame | None:
    """A-plus / B-minus heatmap for a shared (layer, frame).

    Animating the difference highlights which weights diverge between the two
    configs (core recruitment story of the demo).
    """
    sa = a.weight_history.get(layer)
    sb = b.weight_history.get(layer)
    if not sa or not sb:
        return None
    n = min(len(sa), len(sb))
    if n == 0:
        return None
    frame = max(0, min(frame, n - 1))
    vals, rows, cols = _normalize(sa[frame] - sb[frame])
    return MatrixFrame(layer=layer, frame=frame, rows=rows, cols=cols, values=vals)


class WeightMatrixAnimator:
    """NiceGUI widget: animated heatmap + play/pause + scrub slider.

    Renders into an optional ``container`` context. ``diff=True`` shows the
    Config A minus Config B matrix for a chosen layer. Frame index is driven by
    a slider so the animation is scrub-safe even with a dense weight history.
    """

    def __init__(
        self,
        panel_a: DemoPanel,
        panel_b: DemoPanel | None = None,
        layer: str | None = None,
        diff: bool = False,
        fps: int = 8,
    ) -> None:
        self.panel_a = panel_a
        self.panel_b = panel_b
        self.diff = diff
        self.fps = max(1, fps)
        layers = weight_layers(panel_a)
        self.layer = layer or (layers[0] if layers else "")
        self.slider = None
        self.fig = None
        self._timer = None
        self._playing = False

    def _frames(self) -> int:
        if self.diff and self.panel_b is not None:
            return max(align_length(self.panel_a, self.panel_b, self.layer), 1)
        snaps = self.panel_a.weight_history.get(self.layer, [])
        return max(len(snaps), 1)

    def _heatmap(self, from_, to):
        import plotly.graph_objects as go

        if self.diff and self.panel_b is not None:
            frame = diff_frame(self.panel_a, self.panel_b, self.layer, from_)
        else:
            frame = matrix_frame(self.panel_a, self.layer, from_)
        if frame is None:
            return to
        grid = frame.as_grid(frame.rows, frame.cols)
        to.data = [go.Heatmap(z=grid, colorscale="RdBu", zmid=0.5)]
        to.update_layout(
            title=f"{self.layer} · frame {frame.frame + 1}/{self._frames()}",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return to

    def render(self, container=None) -> WeightMatrixAnimator:
        """Build the Plotly heatmap + play/scrub controls (idempotent)."""
        from nicegui import ui

        ctx = container or ui

        with ctx:
            import plotly.graph_objects as go

            self.fig = ui.plotly(
                go.Figure(go.Heatmap(z=[[0]], colorscale="RdBu", zmid=0.5))
            )
            with ui.row():
                play = ui.button("▶", on_click=lambda: self._toggle(play))
                self.slider = ui.slider(
                    min=0,
                    max=self._frames() - 1,
                    step=1,
                    value=0,
                    on_change=lambda e: self._apply(int(e.value)),
                )
            self._heatmap(0, self.fig)
        return self

    def _apply(self, frame: int) -> None:
        if self.fig is not None:
            self._heatmap(frame, self.fig)
            if self.slider is not None:
                self.slider.value = frame

    def _toggle(self, btn) -> None:
        self._playing = not self._playing
        btn.text = "⏸" if self._playing else "▶"
        if self._playing:
            self._step()

    def _step(self) -> None:
        if not self._playing or self.slider is None:
            return
        n = self._frames()
        frame = (int(self.slider.value) + 1) % n
        self._apply(frame)
        import asyncio

        loop = asyncio.get_running_loop()
        loop.call_later(1 / self.fps, self._step)
