"""Standard-library terminal dashboard with Hermes-compatible Braille feedback."""

from __future__ import annotations

import shutil
import sys
from typing import Final, TextIO

from computronium.execution.dashboard._base import BaseDashboard
from computronium.execution.dashboard._utils import (
    format_number,
    format_trial_line,
    sanitize_display_text,
)

BRAILLE_FRAMES: Final = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class Dashboard(BaseDashboard):
    """Render execution lifecycle events as one live terminal status line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__()
        self._stream = stream if stream is not None else sys.stderr
        self._active = False
        self._frame = 0

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

    def _snapshot(self, frame: str) -> str:
        parts = [frame, "AutoScientist"]
        if self.current_trial_info:
            parts.extend(format_trial_line(self.current_trial_info))
        else:
            parts.append("idle")
        if self.best_model is not None:
            best = self.best_model
            parts.append(
                f"best {sanitize_display_text(best['model'])} "
                f"{format_number(best, 'accuracy'):.1%}"
            )
        parts.append(sanitize_display_text(self.system_status))
        parts.append(
            sanitize_display_text(
                self.status_log[-1] if self.status_log else self.insight_text
            )
        )
        return " | ".join(parts)
