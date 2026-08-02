"""Live chart data transforms (Sprint 3.4).

Pure, plotly-independent helpers that turn DemoPanel telemetry into ready-to-
plot series. Keep plotting (NiceGUI `Plotly` element) in main/charts_ui; this
module holds the testable math (rolling windows, dual-axis series, gaps)."""

from __future__ import annotations

from dataclasses import dataclass, field

from runner import DemoPanel, model_metadata


@dataclass
class ChartSeries:
    """A named trace; y can be ``None`` (NaN) where data is absent."""

    name: str
    x: list[int] = field(default_factory=list)
    y: list[float | None] = field(default_factory=list)


def rolling_mean(values: list[float], window: int = 20) -> list[float | None]:
    """Moving average with ``None`` until the window fills (clean spike-free Y)."""
    if window <= 1 or not values:
        return list(values)
    out: list[float | None] = [None] * (window - 1)
    for i in range(window - 1, len(values)):
        out.append(sum(values[i - window + 1 : i + 1]) / window)
    return out


def loss_series(panel: DemoPanel, window: int = 20) -> ChartSeries:
    """Per-step loss, optionally smoothed, x = global step index."""
    values = panel.losses
    return ChartSeries(
        name="loss",
        x=list(range(len(values))),
        y=rolling_mean(values, window),
    )


def accuracy_series(panel: DemoPanel) -> ChartSeries:
    """Per-epoch accuracy, x = epoch index."""
    values = panel.accuracies
    return ChartSeries(name="accuracy", x=list(range(len(values))), y=list(values))


def energy_series(panel: DemoPanel) -> ChartSeries:
    """Per-settling-step energy trace (EqProp/EP only)."""
    values = panel.energies
    return ChartSeries(name="energy", x=list(range(len(values))), y=list(values))


def parity_gap(panel_a: DemoPanel, panel_b: DemoPanel) -> float | None:
    """Final accuracy gap (percentage points) between two panels, if both done."""
    if not (panel_a.finished and panel_b.finished):
        return None
    if not panel_a.accuracies or not panel_b.accuracies:
        return None
    return round((panel_b.accuracies[-1] - panel_a.accuracies[-1]) * 100, 3)


def parity_explanation(panel_a: DemoPanel, panel_b: DemoPanel, gap: float) -> str:
    """Qualifier when a wide gap traces to a backward-free family (Sprint 3.7).

    Equilibrium/forward-only rules that need no backward pass get an inline
    'gap expected' note when they trail the other config by more than their
    documented ``parity_threshold`` (absolute accuracy gap, read from the
    hyperparam YAMLs via registry ``extra``; default 5 pp). This stops the demo
    from reading a known bio trade-off as a plain failure.
    """
    no_bwd = [
        meta
        for cfg in (panel_a.trainer_config, panel_b.trainer_config)
        for meta in (model_metadata(cfg.model),)
        if meta.get("requires_backward") is False
    ]
    if no_bwd and abs(gap) >= 100 * min(
        m.get("parity_threshold", 0.05) for m in no_bwd
    ):
        families = ", ".join(
            sorted({m.get("family") for m in no_bwd if m.get("family")})
        )
        return f" (gap expected: {families} is backward-free)"
    return ""
