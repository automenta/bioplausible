"""Experiment persistence (Sprint 3.6).

Pure helpers to (de)serialize a TrainerConfig to/from JSON so a saved demo
config reloads identically and runs can be exported. The NiceGUI layer calls
these; the logic is browser-free and unit-tested.
"""

from __future__ import annotations

import base64
import csv
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


def export_run_csv(
    losses: list[float],
    accuracies: list[float],
    path: str | Path,
    header: dict[str, Any] | None = None,
) -> Path:
    """Write a per-epoch run trace to CSV (header + epoch/step columns).

    Returns the written path. Pure and browser-free so it is unit-testable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(losses), len(accuracies))
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(["#", *header.keys()])
            writer.writerow(["#", *header.values()])
        writer.writerow(["step", "loss", "accuracy"])
        for i in range(n):
            loss = losses[i] if i < len(losses) else ""
            acc = accuracies[i] if i < len(accuracies) else ""
            writer.writerow([i, loss, acc])
    return path


_URL_PREFIX = "bioplausible://"


def config_to_url(config: TrainerConfig) -> str:
    """Encode a TrainerConfig into a compact shareable URL.

    ``bioplausible://`` + urlsafe-base64(JSON) so a saved demo comparison can
    be pasted into a chat/doc and rehydrated with :func:`config_from_url`.
    Only the selector-relevant knobs are encoded so the URL stays short.
    """
    knobs = {
        "model": config.model,
        "task": config.task,
        "epochs": config.epochs,
        "optimizer_kwargs": {
            k: config.optimizer_kwargs[k]
            for k in ("lr",)
            if k in config.optimizer_kwargs
        },
        "model_kwargs": {
            k: config.model_kwargs[k]
            for k in ("hidden_dim",)
            if k in config.model_kwargs
        },
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(knobs, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return _URL_PREFIX + payload


def config_from_url(url: str) -> TrainerConfig:
    """Rehydrate a TrainerConfig from a :func:`config_to_url` URL."""
    if not url.startswith(_URL_PREFIX):
        raise ValueError("not a bioplausible share URL")
    raw = url[len(_URL_PREFIX) :]
    knobs = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
    return TrainerConfig.from_dict(knobs)


def export_run_png(
    losses: list[float],
    accuracies: list[float],
    path: str | Path,
    title: str = "Bioplausible run",
) -> Path:
    """Render a loss/accuracy trace to a PNG (Agg backend, headless-safe).

    Uses matplotlib's non-interactive Agg backend so this is callable from a
    worker thread without a display; returns the written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = list(range(len(losses)))
    ax.plot(steps, losses, label="loss", color="tab:blue")
    ax.set_xlabel("step")
    ax.set_ylabel("loss", color="tab:blue")
    ax.set_title(title)
    if accuracies:
        ax2 = ax.twinx()
        ax2.plot(
            list(range(len(accuracies))),
            accuracies,
            label="accuracy",
            color="tab:orange",
        )
        ax2.set_ylabel("accuracy", color="tab:orange")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
