"""Rich-backed alternative execution dashboard."""

from __future__ import annotations

from typing import TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from computronium.execution.dashboard._base import BaseDashboard
from computronium.execution.dashboard._utils import (
    format_number,
    sanitize_display_text,
)


class Dashboard(BaseDashboard):
    """Render execution lifecycle events through Rich when ``PURE`` is disabled."""

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__()
        self.console = Console(file=stream, stderr=True)
        self.live = Live(console=self.console, refresh_per_second=4)

    def start(self) -> None:
        """Start Rich live rendering."""
        with self._lock:
            self.live.start()
        self.update()

    def stop(self) -> None:
        """Stop Rich live rendering."""
        with self._lock:
            self.live.stop()

    def update(self) -> None:
        """Render the latest experiment state."""
        with self._lock:
            renderable = self._render()
        self.live.update(renderable)

    def _render(self) -> Group:
        trial = self.current_trial_info
        summary = "idle"
        if trial is not None:
            metrics = trial["metrics"]
            summary = (
                f"{sanitize_display_text(trial['model'])}/"
                f"{sanitize_display_text(trial['task'])} · "
                f"epoch {trial['epoch']}/{trial['total_epochs']} · "
                f"loss {format_number(metrics, 'loss'):.4f} · "
                f"acc {format_number(metrics, 'accuracy'):.1%}"
            )
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify="right")
        table.add_row(Text(summary), Text(sanitize_display_text(self.system_status)))
        logs = "\n".join(self.status_log[-5:]) or self.insight_text
        best = "—"
        if self.best_model is not None:
            best = (
                f"{sanitize_display_text(self.best_model['model'])} "
                f"{format_number(self.best_model, 'accuracy'):.1%}"
            )
        return Group(
            Panel(table, title="AutoScientist"),
            Panel(Text(sanitize_display_text(logs)), title=Text(f"Best: {best}")),
        )
