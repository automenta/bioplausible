"""Dependency-free execution dashboard tests."""

from __future__ import annotations

from io import StringIO
from logging import Logger

from computronium.execution.dashboard import BRAILLE_FRAMES, PURE, Dashboard


def test_dashboard_exports_standard_logger() -> None:
    """The public dashboard logger remains available without Rich."""
    from computronium.execution.dashboard import logger

    assert isinstance(logger, Logger)


def test_pure_dashboard_uses_hermes_braille_frames() -> None:
    """The default dashboard reuses Hermes' one-cell Unicode spinner."""
    assert PURE is True
    assert BRAILLE_FRAMES == ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def test_pure_dashboard_strips_terminal_control_sequences() -> None:
    """Event text cannot inject terminal control sequences into the live line."""
    stream = StringIO()
    dashboard = Dashboard(stream=stream)
    sequence = "\x1b]8;;https://example.test\x1b\\"

    dashboard.start()
    dashboard.set_trial("42", f"tile{sequence}", "digits", "SMOKE", {})
    dashboard.set_system_status(f"ready{sequence}")
    dashboard.stop()

    assert sequence not in stream.getvalue()


def test_pure_dashboard_surfaces_new_sota_event() -> None:
    """A newly best completed trial remains visible in the terminal dashboard."""
    stream = StringIO()
    dashboard = Dashboard(stream=stream)

    dashboard.start()
    dashboard.set_trial("42", "tile_pc", "digits", "SMOKE", {})
    dashboard.complete_trial("completed", {"accuracy": 0.875})
    dashboard.stop()

    assert "New SOTA: 87.5% (tile_pc)" in stream.getvalue()


def test_pure_dashboard_renders_trial_progress_and_best_result() -> None:
    """Lifecycle events produce a concise dependency-free terminal dashboard."""
    stream = StringIO()
    dashboard = Dashboard(stream=stream)

    dashboard.start()
    dashboard.set_trial("42", "tile_pc", "digits", "SMOKE", {"lr": 0.01})
    dashboard.update_progress(1, 3, {"loss": 0.125, "accuracy": 0.875})
    dashboard.complete_trial("completed", {"accuracy": 0.875})
    dashboard.stop()

    output = stream.getvalue()
    assert "⠋" in output
    assert "tile_pc/digits" in output
    assert "epoch 1/3" in output
    assert "loss 0.1250" in output
    assert "acc 87.5%" in output
    assert "best tile_pc 87.5%" in output
    assert output.endswith("\n")
