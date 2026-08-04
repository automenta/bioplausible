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
from collections.abc import Callable

from compat import apply_compat_shims

apply_compat_shims()  # must run before `import nicegui`

from charts import (  # ruff: ignore[module-import-not-at-top-of-file]
    loss_series,
    parity_explanation,
    parity_gap,
)
from nicegui import ui  # ruff: ignore[module-import-not-at-top-of-file]
from persistence import (  # ruff: ignore[module-import-not-at-top-of-file]
    config_to_url,
    export_run_csv,
    export_run_png,
    load_config,
    save_config,
)
from renderer import render_group  # ruff: ignore[module-import-not-at-top-of-file]
from runner import (  # ruff: ignore[module-import-not-at-top-of-file]
    TRAINABLE_MODELS,
    DemoPanel,
    default_trainer_config,
    model_metadata,
    prepare_trainer_config,
    run_headless,
)
from tasks import build_tasks  # ruff: ignore[module-import-not-at-top-of-file]
from widgets import build_widget_tree  # ruff: ignore[module-import-not-at-top-of-file]

logger = logging.getLogger(__name__)

DEMO_MODELS = list(TRAINABLE_MODELS)


def _fresh_panel(model: str, task: str, epochs: int, lr: float) -> DemoPanel:
    cfg = default_trainer_config(model=model, task=task, epochs=epochs, lr=lr)
    return DemoPanel(trainer_config=cfg, epochs=epochs)


def _cooked_panel(
    prev: DemoPanel | None, model: str, task: str, epochs: int, lr: float
) -> DemoPanel:
    """Build the panel to train, preserving live widget-tree knob edits."""
    cfg = prepare_trainer_config(
        prev.trainer_config if prev is not None else None,
        model,
        task,
        int(epochs),
        float(lr),
    )
    return DemoPanel(trainer_config=cfg, epochs=int(epochs))


class DemoUi:
    """Holds both config panels + the shared task/epoch controls."""

    def __init__(self) -> None:
        self.panel_a: DemoPanel | None = None
        self.panel_b: DemoPanel | None = None
        self.loss_fig: ui.elements.plotly_element.Plotly | None = None
        self.acc_fig: ui.elements.plotly_element.Plotly | None = None
        self.gap_label: ui.label | None = None
        self.run_btn: ui.button | None = None
        self.meta_a: ui.label | None = None
        self.meta_b: ui.label | None = None
        self.weight_box: ui.column | None = None

    def _make_chart(self) -> ui.elements.plotly_element.Plotly:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Scatter(name="A", mode="lines"),
                go.Scatter(name="B", mode="lines"),
            ]
        )
        fig.update_layout(height=260, margin=dict(l=40, r=20, t=20, b=30))
        return ui.plotly(fig)


def _meta_text(name: str) -> str:
    """One-line Sprint 2.5 metadata summary for a model (or '—' if unknown)."""
    m = model_metadata(name)
    if not m:
        return "—"
    return (
        f"bio {m['bio_plausibility_score']} · locality {m['locality_level']} · "
        f"family {m['family']}"
    )


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

    # --- Live editable config panels (Sprint 3.2 widget tree) ---
    demo.panel_a = _fresh_panel(model_a.value, task_sel.value, 5, 0.001)
    demo.panel_b = _fresh_panel(model_b.value, task_sel.value, 5, 0.001)

    demo.meta_a = ui.label("")
    demo.meta_b = ui.label("")
    demo.meta_a.set_text(_meta_text(model_a.value))
    demo.meta_b.set_text(_meta_text(model_b.value))
    model_a.on("change", lambda: demo.meta_a.set_text(_meta_text(model_a.value)))
    model_b.on("change", lambda: demo.meta_b.set_text(_meta_text(model_b.value)))

    def sync_a() -> None:
        """Rebind quick-set controls (epochs/lr) onto panel A's live config."""
        demo.panel_a.trainer_config.epochs = int(epochs.value)
        demo.panel_a.trainer_config.optimizer_kwargs["lr"] = float(lr.value)

    def sync_b() -> None:
        demo.panel_b.trainer_config.epochs = int(epochs.value)
        demo.panel_b.trainer_config.optimizer_kwargs["lr"] = float(lr.value)

    with ui.row():
        with ui.column():
            ui.label("Config A")
            render_group(
                build_widget_tree(demo.panel_a.trainer_config, "Config A"),
                demo.panel_a.trainer_config,
                lambda cfg: None,  # renderer mutates config in place
                ui.column(),
            )
        with ui.column():
            ui.label("Config B")
            render_group(
                build_widget_tree(demo.panel_b.trainer_config, "Config B"),
                demo.panel_b.trainer_config,
                lambda cfg: None,
                ui.column(),
            )

    # --- Charts ---
    with ui.row():
        demo.loss_fig = demo._make_chart()
        demo.acc_fig = demo._make_chart()

    demo.gap_label = ui.label("Parity gap: —")

    # --- Animated weight matrices (Sprint 3.5) ---
    demo.weight_box = ui.column()
    _refresh_weight_viz(demo)

    # --- Train button ---
    async def train() -> None:
        # Use the current selectors, but reuse each panel's live (widget-mutated)
        # config when the model/task are unchanged so Sprint 3.2 knob edits feed
        # the run; only a model/task change rebuilds from defaults.
        demo.panel_a = _cooked_panel(
            demo.panel_a,
            model_a.value,
            task_sel.value,
            int(epochs.value),
            float(lr.value),
        )
        demo.panel_b = _cooked_panel(
            demo.panel_b,
            model_b.value,
            task_sel.value,
            int(epochs.value),
            float(lr.value),
        )
        sync_a()
        sync_b()
        demo.run_btn.disable()

        async def train_one(panel: DemoPanel) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, run_headless, panel)

        await asyncio.gather(train_one(demo.panel_a), train_one(demo.panel_b))

        _refresh_charts(demo)
        _refresh_weight_viz(demo)
        errs = [
            f"{p.trainer_config.model}: {p.error}"
            for p in (demo.panel_a, demo.panel_b)
            if p.error
        ]
        if errs:
            demo.gap_label.set_text("Error: " + " | ".join(errs))
            demo.run_btn.enable()
            return
        gap = parity_gap(demo.panel_a, demo.panel_b)
        if gap is None:
            text = "Parity gap: —"
        else:
            note = parity_explanation(demo.panel_a, demo.panel_b, gap)
            text = f"Parity gap (B−A): {gap} pp{note}"
        demo.gap_label.set_text(text)
        demo.run_btn.enable()

    demo.run_btn = ui.button("Run", on_click=train)

    # --- Persistence controls (Sprint 3.6) ---
    export_info = ui.label("").classes("text-grey text-xs")
    with ui.row():
        ui.button("Save Config A", on_click=_save_cfg("a", demo, export_info))
        ui.button("Save Config B", on_click=_save_cfg("b", demo, export_info))
        ui.button("Load Config A", on_click=_load_cfg("a", demo, export_info))
        ui.button(
            "Copy Share URL A",
            on_click=_copy_share("a", demo, export_info),
        )
    with ui.row():
        ui.button("Export Run (CSV+PNG)", on_click=_export_run(demo, export_info))


def _save_cfg(side: str, demo: DemoUi, status) -> Callable[[], None]:
    from pathlib import Path

    def _do() -> None:
        panel = demo.panel_a if side == "a" else demo.panel_b
        if panel is None:
            return
        path = Path(f"/tmp/bioplausible-{side}.json")
        save_config(panel.trainer_config, path)
        ui.download(path)
        status.set_text(f"Saved Config {side.upper()} to {path}")

    return _do


def _load_cfg(side: str, demo: DemoUi, status) -> Callable[[], None]:
    from pathlib import Path

    def _do() -> None:
        panel = demo.panel_a if side == "a" else demo.panel_b
        if panel is None:
            return
        path = Path(f"/tmp/bioplausible-{side}.json")
        if not path.exists():
            status.set_text(f"No saved Config {side.upper()} yet")
            return
        panel.trainer_config = load_config(path)
        status.set_text(f"Loaded Config {side.upper()}: {panel.trainer_config.model}")

    return _do


def _copy_share(side: str, demo: DemoUi, status) -> Callable[[], None]:
    def _do() -> None:
        panel = demo.panel_a if side == "a" else demo.panel_b
        if panel is None:
            return
        ui.clipboard().write(config_to_url(panel.trainer_config))
        status.set_text(f"Config {side.upper()} share URL copied to clipboard")

    return _do


def _export_run(demo: DemoUi, status) -> Callable[[], None]:
    from pathlib import Path

    def _do() -> None:
        a, b = demo.panel_a, demo.panel_b
        if a is None or b is None:
            return
        status.set_text("Exporting run CSV + PNG…")
        for label, panel in (("A", a), ("B", b)):
            csv_path = Path(f"/tmp/bioplausible-run-{label}.csv")
            png_path = Path(f"/tmp/bioplausible-run-{label}.png")
            export_run_csv(
                panel.losses,
                panel.accuracies,
                csv_path,
                header={
                    "model": panel.trainer_config.model,
                    "task": panel.trainer_config.task,
                },
            )
            export_run_png(
                panel.losses,
                panel.accuracies,
                png_path,
                title=f"Bioplausible {label} — {panel.trainer_config.model}",
            )
            ui.download(csv_path)
            ui.download(png_path)
        status.set_text("Exported CSV + PNG for both configs")

    return _do


def _refresh_charts(demo: DemoUi) -> None:
    """Reset and redraw chart figures from the finished panels."""
    if (
        demo.loss_fig is not None
        and demo.panel_a is not None
        and demo.panel_b is not None
    ):
        a = loss_series(demo.panel_a)
        b = loss_series(demo.panel_b)
        demo.loss_fig.update_traces(x=[a.x, b.x], y=[a.y, b.y], selector={})


def _refresh_weight_viz(demo: DemoUi) -> None:
    """(Re)build the animated weight-matrix widget from the current panels."""
    from weight_viz import WeightMatrixAnimator

    if demo.weight_box is None or demo.panel_a is None or demo.panel_b is None:
        return
    demo.weight_box.clear()
    if not demo.panel_a.weight_history:
        with demo.weight_box:
            ui.label("Run training to inspect weight evolution").classes("text-grey")
        return
    with demo.weight_box:
        ui.label("Weight evolution (Config A − Config B)").classes("text-bold")
        WeightMatrixAnimator(demo.panel_a, demo.panel_b, diff=True).render(
            container=None
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
