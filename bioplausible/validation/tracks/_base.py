"""Shared boilerplate for validation tracks: banner logging and result assembly.

Collapses the per-track header (three logging lines + timing anchor) and the
``TrackResult`` construction (including elapsed-time computation) into two
helpers; each ``track_*`` keeps only its bespoke scoring/evidence.
"""

from __future__ import annotations

import time

from bioplausible.core.logging import get_logger

from ..notebook import TrackResult

logger = get_logger()

__all__ = [
    "build_track_result",
    "track_header",
]


def track_header(track_id: int, name: str, width: int = 60) -> float:
    """Log the standard track banner and return the elapsed-time anchor."""
    logger.info("\n%s", "=" * width)
    logger.info("TRACK %s: %s", track_id, name)
    logger.info("%s", "=" * width)
    return time.time()


def build_track_result(  # ruff: ignore[too-many-arguments] (mirrors TrackResult fields)
    *,
    track_id: int,
    name: str,
    start: float,
    status: str,
    score: float,
    metrics: dict,
    evidence: str,
    improvements: list[str] | None = None,
    evidence_level: str = "smoke",
    limitations: list[str] | None = None,
) -> TrackResult:
    """Assemble a ``TrackResult`` with elapsed time since ``track_header``."""
    return TrackResult(
        track_id=track_id,
        name=name,
        status=status,
        score=score,
        metrics=metrics,
        evidence=evidence,
        time_seconds=time.time() - start,
        improvements=improvements if improvements is not None else [],
        evidence_level=evidence_level,
        limitations=limitations if limitations is not None else [],
    )
