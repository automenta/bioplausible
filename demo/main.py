"""Bioplausible NiceGUI demo entry point (Sprint 3).

Two-panel side-by-side config comparison with live streaming charts. The
training runs in worker threads via :mod:`runner`, emitting telemetry through
the Sprint 3.4 ``ExecutionCallback`` protocol — the UI is a pure consumer.

Run::

    uv run python main.py        # or: uv run uvicorn main:app
"""

from __future__ import annotations

import asyncio
import logging

from compat import apply_compat_shims

apply_compat_shims()  # must run before `import nicegui`

from nicegui import ui  # noqa: E402

from charts import loss_series, parity_gap  # noqa: E402
from runner import DemoPanel, default_trainer_config, run_headless  # noqa: E402
from tasks import build_tasks  # noqa: E402

logger = logging.getLogger(__name__)

DEMO_MODELS = [
    "backprop_mlp",
    "eqprop_mlp",
    "equitile",
    "forward_forward",
    "pepita",
    "feedback_alignment",
]


def _fresh_panel(model: str, task: str, epochs: int, lr: float) -> DemoPanel:
    cfg = default_trainer_config(model=model, task=task, epochs=epochs, lr=lr)
    return DemoPanel(trainer_config=cfg, epochs=epochs)


class DemoUi:
    """Holds both config panels + the shared task/epoch controls."""

    def __init__(self) -> None:
        self.panel_a: DemoPanel | None = None
        self.panel_b: DemoPanel | None = None
        self.loss_fig: ui.elements.plotly_element.Plotly | None = None
        self.acc_fig: ui.elements.plotly_element.Plotly | None = None
        self.gap_label: ui.label | None = None
        self.run_btn: ui.button | None = None

    def _make_chart(self) -> "ui.elements.plotly_element.Plotly":
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Scatter(name="A", mode="lines"),
                go.Scatter(name="B", mode="lines"),
            ]
        )
        fig.update_layout(height=260, margin=dict(l=40, r=20, t=20, b=30))
        return ui.plotly(fig)


def create_page(demo: DemoUi) -> None:
    """Compose the demo page (called once at startup)."""
    ui.dark_mode().enable()

    # --- Controls row ---
    with ui.row():
        task_sel = ui.select(
            [t.name for t in build_tasks()], value="mnist", label="Task"
        )
        model_a = ui.select(DEMO_MODELS, value="equitile", label="Config A")
        model_b = ui.select(DEMO_MODELS, value="backprop_mlp", label="Config B")
        epochs = ui.number(value=5, min=1, max=50, label="Epochs")
        lr = ui.number(value=0.001, format="%.4f", label="Learning Rate")

    # --- Charts ---
    with ui.row():
        demo.loss_fig = demo._make_chart()
        demo.acc_fig = demo._make_chart()

    demo.gap_label = ui.label("Parity gap: —")

    # --- Train button ---
    async def train() -> None:
        demo.panel_a = _fresh_panel(
            model_a.value, task_sel.value, int(epochs.value), float(lr.value)
        )
        demo.panel_b = _fresh_panel(
            model_b.value, task_sel.value, int(epochs.value), float(lr.value)
        )
        demo.run_btn.disable()

        async def train_one(panel: DemoPanel) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, run_headless, panel)

        await asyncio.gather(train_one(demo.panel_a), train_one(demo.panel_b))

        _refresh_charts(demo)
        gap = parity_gap(demo.panel_a, demo.panel_b)
        text = f"Parity gap: {gap} pp" if gap is not None else "Parity gap: —"
        demo.gap_label.set_text(text)
        demo.run_btn.enable()

    demo.run_btn = ui.button("Run", on_click=train)


def _refresh_charts(demo: DemoUi) -> None:
    """Reset and redraw chart figures from the finished panels."""
    if (
        demo.loss_fig is not None
        and demo.panel_a is not None
        and demo.panel_b is not None
    ):
        a = loss_series(demo.panel_a)
        b = loss_series(demo.panel_b)
        demo.loss_fig.update_traces(
            x=[a.x, b.x], y=[a.y, b.y], selector={}
        )


def build_ui() -> DemoUi:
    """Instantiate and compose the UI, returning the state object."""
    demo = DemoUi()
    create_page(demo)
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_ui()
    ui.run(title="Bioplausible Demo", port=8080, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
