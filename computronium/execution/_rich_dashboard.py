"""Rich-backed alternative execution dashboard."""

from __future__ import annotations

from threading import RLock
from typing import cast

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Dashboard:
    """Render execution lifecycle events through Rich when ``PURE`` is disabled."""

    def __init__(self) -> None:
        self.console = Console(stderr=True)
        self.live = Live(console=self.console, refresh_per_second=4)
        self._lock = RLock()
        self.status_log: list[str] = []
        self.recent_trials: list[dict[str, object]] = []
        self.current_trial_info: dict[str, object] = {}
        self.best_model: dict[str, object] | None = None
        self.insight_text = "Initializing analysis modules..."
        self.system_status = "Initializing"

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

    def log(self, message: str, style: str = "") -> None:
        """Record an execution message."""
        del style
        with self._lock:
            self.status_log.append(message)
        self.update()

    def set_trial(
        self,
        trial_id: str,
        model: str,
        task: str,
        tier: str,
        params: dict[str, object],
    ) -> None:
        """Set the experiment currently being executed."""
        with self._lock:
            self.current_trial_info = {
                "id": trial_id,
                "model": model,
                "task": task,
                "tier": tier,
                "params": params,
                "epoch": 0,
                "total_epochs": 0,
                "metrics": {},
            }
        self.update()

    def update_progress(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None:
        """Update progress and metrics for the current experiment."""
        with self._lock:
            self.current_trial_info.update(
                epoch=epoch, total_epochs=total_epochs, metrics=metrics
            )
        self.update()

    def complete_trial(self, status: str, metrics: dict[str, object]) -> None:
        """Add the current experiment to history and update the best result."""
        with self._lock:
            if not self.current_trial_info:
                return
            accuracy = _number(metrics, "accuracy")
            trial: dict[str, object] = {
                "id": self.current_trial_info["id"],
                "model": self.current_trial_info["model"],
                "task": self.current_trial_info["task"],
                "accuracy": accuracy,
                "metrics": metrics,
                "status": status,
            }
            self.recent_trials.append(trial)
            if status == "completed" and (
                self.best_model is None
                or accuracy > _number(self.best_model, "accuracy")
            ):
                self.best_model = trial
            self.current_trial_info = {}
        self.update()

    def set_insight(self, text: str) -> None:
        """Update the current scientific insight."""
        with self._lock:
            self.insight_text = text
        self.update()

    def set_system_status(self, status: str, style: str = "white") -> None:
        """Update the system status."""
        del style
        with self._lock:
            self.system_status = status
        self.update()

    def _render(self) -> Group:
        trial = self.current_trial_info
        summary = "idle"
        if trial:
            metrics = cast("dict[str, object]", trial["metrics"])
            summary = (
                f"{_display_text(trial['model'])}/{_display_text(trial['task'])} · "
                f"epoch {trial['epoch']}/{trial['total_epochs']} · "
                f"loss {_number(metrics, 'loss'):.4f} · "
                f"acc {_number(metrics, 'accuracy'):.1%}"
            )
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify="right")
        table.add_row(Text(summary), Text(_display_text(self.system_status)))
        logs = "\n".join(self.status_log[-5:]) or self.insight_text
        best = "—"
        if self.best_model is not None:
            best = (
                f"{_display_text(self.best_model['model'])} "
                f"{_number(self.best_model, 'accuracy'):.1%}"
            )
        return Group(
            Panel(table, title="AutoScientist"),
            Panel(Text(_display_text(logs)), title=Text(f"Best: {best}")),
        )


def _number(values: dict[str, object], name: str) -> float:
    value = values.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _display_text(value: object) -> str:
    return "".join(char if char.isprintable() else "?" for char in str(value))
