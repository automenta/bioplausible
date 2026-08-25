"""Formatting primitives and display-state vocabulary shared by dashboards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

__all__ = [
    "ActiveTrial",
    "CompletedTrial",
    "format_number",
    "format_trial_line",
    "sanitize_display_text",
]


class ActiveTrial(TypedDict):
    """Experiment currently being executed, set atomically by ``set_trial``."""

    id: str
    model: str
    task: str
    tier: str
    params: dict[str, object]
    epoch: int
    total_epochs: int
    metrics: Mapping[str, object]


class CompletedTrial(TypedDict):
    """Historical record appended to history when a trial completes."""

    id: str
    model: str
    task: str
    accuracy: float
    metrics: Mapping[str, object]
    status: str


def format_number(values: Mapping[str, object], name: str) -> float:
    """Extract a numeric value, falling back to ``0.0`` for non-numeric input."""
    value = values.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def sanitize_display_text(value: object) -> str:
    """Replace non-printable characters so text cannot inject terminal escapes."""
    return "".join(char if char.isprintable() else "?" for char in str(value))


def format_trial_line(trial: Mapping[str, object]) -> list[str]:
    """Summarize a trial as compact single-line parts for terminal rendering."""
    parts = [
        f"{sanitize_display_text(trial.get('model', 'N/A'))}/"
        f"{sanitize_display_text(trial.get('task', 'N/A'))}"
    ]
    epoch = format_number(trial, "epoch")
    total_epochs = format_number(trial, "total_epochs")
    if total_epochs:
        parts.append(f"epoch {epoch:.0f}/{total_epochs:.0f}")
    metrics = trial.get("metrics")
    if isinstance(metrics, Mapping):
        values = cast("Mapping[str, object]", metrics)
        if "loss" in values:
            parts.append(f"loss {format_number(values, 'loss'):.4f}")
        if "accuracy" in values:
            parts.append(f"acc {format_number(values, 'accuracy'):.1%}")
    return parts
