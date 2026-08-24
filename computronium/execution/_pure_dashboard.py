"""Standard-library terminal dashboard with Hermes-compatible Braille feedback."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Mapping
from threading import RLock
from typing import Final, TextIO, cast

BRAILLE_FRAMES: Final = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class Dashboard:
    """Render execution lifecycle events as one live terminal status line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._lock = RLock()
        self._active = False
        self._frame = 0
        self.status_log: list[str] = []
        self.recent_trials: list[dict[str, object]] = []
        self.current_trial_info: dict[str, object] = {}
        self.best_model: dict[str, object] | None = None
        self.insight_text = "Initializing analysis modules..."
        self.system_status = "Initializing"

    def start(self) -> None:
        """Start emitting live dashboard updates."""
        with self._lock:
            self._active = True
        self.update()

    def stop(self) -> None:
        """Stop live updates and terminate the status line."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stream.write("\n")
            self._stream.flush()

    def update(self) -> None:
        """Render the current execution state when the dashboard is active."""
        with self._lock:
            if not self._active:
                return
            frame = BRAILLE_FRAMES[self._frame]
            self._frame = (self._frame + 1) % len(BRAILLE_FRAMES)
            width = shutil.get_terminal_size(fallback=(100, 24)).columns
            line = self._snapshot(frame)
            self._stream.write(f"\r\x1b[2K{line[: max(width - 1, 1)]}")
            self._stream.flush()

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
            self.current_trial_info["epoch"] = epoch
            self.current_trial_info["total_epochs"] = total_epochs
            self.current_trial_info["metrics"] = metrics
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
                self.status_log.append(
                    f"New SOTA: {accuracy:.1%} ({_display_text(trial['model'])})"
                )
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

    def _snapshot(self, frame: str) -> str:
        parts = [frame, "AutoScientist"]
        if self.current_trial_info:
            parts.extend(_trial_parts(self.current_trial_info))
        else:
            parts.append("idle")
        if self.best_model is not None:
            parts.append(
                f"best {_display_text(self.best_model['model'])} "
                f"{_number(self.best_model, 'accuracy'):.1%}"
            )
        parts.append(_display_text(self.system_status))
        parts.append(
            _display_text(self.status_log[-1] if self.status_log else self.insight_text)
        )
        return " | ".join(parts)


def _trial_parts(trial: Mapping[str, object]) -> list[str]:
    model = trial.get("model", "N/A")
    task = trial.get("task", "N/A")
    parts = [f"{_display_text(model)}/{_display_text(task)}"]
    epoch = _number(trial, "epoch")
    total_epochs = _number(trial, "total_epochs")
    if total_epochs:
        parts.append(f"epoch {epoch:.0f}/{total_epochs:.0f}")
    metrics = trial.get("metrics")
    if isinstance(metrics, Mapping):
        values = cast("Mapping[str, object]", metrics)
        loss = _number(values, "loss")
        accuracy = _number(values, "accuracy")
        if "loss" in values:
            parts.append(f"loss {loss:.4f}")
        if "accuracy" in values:
            parts.append(f"acc {accuracy:.1%}")
    return parts


def _number(values: Mapping[str, object], name: str) -> float:
    value = values.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _display_text(value: object) -> str:
    return "".join(char if char.isprintable() else "?" for char in str(value))
