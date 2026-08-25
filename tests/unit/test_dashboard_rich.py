"""Rich-backed execution dashboard fallback tests."""

from __future__ import annotations

import inspect
from io import StringIO

from computronium.execution.dashboard._pure import Dashboard as PureDashboard
from computronium.execution.dashboard._rich import Dashboard as RichDashboard


def _public_api(cls: type) -> frozenset[str]:
    return frozenset(
        name
        for name, _ in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    )


def test_rich_and_pure_dashboards_share_public_api() -> None:
    """Both backends expose an identical public surface (no API divergence)."""
    assert _public_api(RichDashboard) == _public_api(PureDashboard)


def test_rich_dashboard_renders_full_lifecycle() -> None:
    """The Rich fallback tracks trials and mirrors shared dashboard behavior."""
    stream = StringIO()
    dashboard = RichDashboard(stream=stream)

    dashboard.start()
    dashboard.set_trial("42", "tile_pc", "digits", "SMOKE", {"lr": 0.01})
    dashboard.update_progress(1, 3, {"loss": 0.125, "accuracy": 0.875})
    dashboard.complete_trial("completed", {"accuracy": 0.875})
    dashboard.live.refresh()
    dashboard.stop()

    # State assertions (robust, independent of Live render timing)
    assert dashboard.recent_trials[-1]["accuracy"] == 0.875
    assert dashboard.best_model is not None
    assert "New SOTA: 87.5% (tile_pc)" in dashboard.status_log

    # Smoke: final render contains best model and SOTA log
    output = stream.getvalue()
    assert "Best: tile_pc 87.5%" in output
    assert "New SOTA: 87.5% (tile_pc)" in output
