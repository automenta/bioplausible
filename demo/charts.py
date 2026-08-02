"""Live chart data transforms (Sprint 3.4).

Pure, plotly-independent helpers that turn DemoPanel telemetry into ready-to-
plot series. Keep plotting (NiceGUI `Plotly` element) in main/charts_ui; this
module holds the testable math (rolling windows, dual-axis series, gaps)."""

from __future__ import annotations

from dataclasses import dataclass, field

from runner import DemoPanel


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
