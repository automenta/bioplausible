"""Experiment persistence (Sprint 3.6).

Pure helpers to (de)serialize a TrainerConfig to/from JSON so a saved demo
config reloads identically and runs can be exported. The NiceGUI layer calls
these; the logic is browser-free and unit-tested.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from bioplausible.core.trainer import TrainerConfig


def _scrub(value: Any) -> Any:
    """Convert dataclasses/objects into plain JSON-safe structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _scrub(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def config_to_dict(config: TrainerConfig) -> dict[str, Any]:
    """Serialize a TrainerConfig to a JSON-safe dict."""
    return _scrub(config)  # type: ignore[return-value]


def save_config(config: TrainerConfig, path: str | Path) -> Path:
    """Write a config to ``path`` as formatted JSON; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, default=str))
    return path


def load_config(path: str | Path) -> TrainerConfig:
    """Reconstruct a TrainerConfig from a previously saved JSON file."""
    data = json.loads(Path(path).read_text())
    return TrainerConfig.from_dict(data)


def export_summary(
    losses: list[float],
    accuracies: list[float],
    model: str,
    task: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a minimal run-export payload (charts/CSV-ready)."""
    return {
        "model": model,
        "task": task,
        "seed": seed,
        "final_accuracy": accuracies[-1] if accuracies else None,
        "final_loss": losses[-1] if losses else None,
        "n_steps": len(losses),
        "n_epochs": len(accuracies),
    }
