"""Typed, single-file JSONL experiment logging (FIX2a §4.3).

Every stage of a trial is appended as one self-describing JSON object to a
single ``.jsonl`` file — no duplicate parameters, no parse hacks. Consumers
(e.g. ``jq '.config' metrics.jsonl``) read a strict newline-delimited stream.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

__all__ = [
    "Epoch",
    "ExperimentLogger",
    "GateOutcome",
    "TrialEnd",
    "TrialStart",
]


@dataclass(frozen=True, slots=True)
class TrialStart:
    """Event emitted immediately before a trial begins training."""

    trial_id: int
    model: str
    task: str
    arm: str
    config: dict[str, object]
    param_count: int
    seed: int
    kind: str = "trial_start"


@dataclass(frozen=True, slots=True)
class TrialEnd:
    """Event emitted when a trial finishes (success or failure)."""

    trial_id: int
    status: str
    metrics: dict[str, float]
    wall_time_s: float
    kind: str = "trial_end"


@dataclass(frozen=True, slots=True)
class Epoch:
    """Per-epoch progress event."""

    trial_id: int
    epoch: int
    metrics: dict[str, float]
    kind: str = "epoch"


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """A staircase gate (TIER 0 / 0.5) verdict for one model x task."""

    tier: str
    model: str
    task: str
    passed: bool
    reason: str
    metrics: dict[str, float]
    kind: str = "gate"


class ExperimentLogger:
    """Append-only JSONL logger bound to a single artifact file.

    The file handle is opened once and flushed after every event so that a
    crashed process never loses more than the in-flight event.
    """

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    @staticmethod
    def _encode(event: object) -> dict[str, object]:
        if isinstance(event, (TrialStart, TrialEnd, Epoch, GateOutcome)):
            return asdict(event)
        raise TypeError(f"Unsupported log event: {event!r}")  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API

    def log(self, event: object) -> None:
        """Serialize ``event`` and append one JSON line, then flush."""
        payload = self._encode(event)
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
