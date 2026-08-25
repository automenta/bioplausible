"""Shared dashboard state management and backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock

from computronium.execution.dashboard._utils import (
    ActiveTrial,
    CompletedTrial,
    format_number,
    sanitize_display_text,
)

__all__ = ["BaseDashboard"]


class BaseDashboard(ABC):
    """Single source of truth for execution-dashboard state transitions.

    Backends subclass and implement :meth:`start`, :meth:`stop`, and
    :meth:`update`; every lifecycle event handler lives here so renderers
    cannot drift apart.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self.status_log: list[str] = []
        self.recent_trials: list[CompletedTrial] = []
        self.current_trial_info: ActiveTrial | None = None
        self.best_model: CompletedTrial | None = None
        self.insight_text = "Initializing analysis modules..."
        self.system_status = "Initializing"

    @abstractmethod
    def start(self) -> None:
        """Start emitting live dashboard updates."""

    @abstractmethod
    def stop(self) -> None:
        """Stop live updates."""

    @abstractmethod
    def update(self) -> None:
        """Render the current execution state."""

    def log(self, message: str, style: str = "") -> None:
        """Record an execution message and refresh the live display."""
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
            current = self.current_trial_info
            if current is not None:
                current["epoch"] = epoch
                current["total_epochs"] = total_epochs
                current["metrics"] = metrics
        self.update()

    def complete_trial(self, status: str, metrics: dict[str, object]) -> None:
        """Add the current experiment to history and track the best result."""
        with self._lock:
            current = self.current_trial_info
            if current is None:
                return
            accuracy = format_number(metrics, "accuracy")
            trial: CompletedTrial = {
                "id": current["id"],
                "model": current["model"],
                "task": current["task"],
                "accuracy": accuracy,
                "metrics": metrics,
                "status": status,
            }
            self.recent_trials.append(trial)
            if status == "completed" and (
                self.best_model is None
                or accuracy > format_number(self.best_model, "accuracy")
            ):
                self.best_model = trial
                self.status_log.append(
                    f"New SOTA: {accuracy:.1%} "
                    f"({sanitize_display_text(trial['model'])})"
                )
            self.current_trial_info = None
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
