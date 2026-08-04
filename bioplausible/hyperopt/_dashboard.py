"""Phase 1.5 live run log — colorful, emoji-rich, append-only.

A plain scrolling console log (no full-screen Layout/Live). Every trial start and
completion is printed immediately and **never overwritten**, so nothing is hidden.
Rich ``Console`` is used purely for color/emoji styling; there is no background
polling thread. Events are emitted synchronously by the runner on the main thread.

Key design choice vs. the old TUI: a full-screen Layout table truncated cells
(``0.2422M`` -> ``0…``) and only showed a static param estimate. This log instead
prints the **real** per-trial values (accuracy, param count, epoch time) read
directly from each completed Optuna trial.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

# ─── Level → (emoji, rich style) ─────────────────────────────────────────────
_LEVELS = {
    "START": ("▶️", "bold bright_blue"),
    "COMPLETE": ("✅", "bold bright_green"),
    "FAIL": ("💥", "bold bright_red"),
    "PRUNED": ("✂️", "bright_red"),
    "WARNING": ("⚠️", "bold yellow"),
    "ERROR": ("❌", "bold bright_red"),
    "EMIT": ("📄", "dim bright_cyan"),
    "INFO": ("ℹ️", "bright_white"),
    "DONE": ("🎉", "bold bright_magenta"),
}


class Dashboard:
    """Append-only colorful emoji log that prints every event immediately."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def add_log(
        self,
        level: str,
        message: str,
        family: str | None = None,
        model: str | None = None,
    ) -> None:
        """Print one colored, emoji-prefixed log line (never buffered/erased)."""
        emoji, style = _LEVELS.get(level.upper(), _LEVELS["INFO"])
        text = Text()
        text.append(f"{emoji} ", style=style)
        if family:
            tag = f"{family}/{model}" if model else f"{family}"
            text.append(f"[{tag}] ", style="bold cyan")
        text.append(message, style=style)
        self.console.print(text)


def create_dashboard(console: Console | None = None) -> Dashboard:
    """Create a new live run-log dashboard instance."""
    return Dashboard(console=console)
