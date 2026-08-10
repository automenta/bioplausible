"""Canonical metrics dataclasses shared across the bioplausible codebase.

These provide a small, frozen+slots hierarchy so that domain-specific metric
containers (training/epoch/trial) share a common base shape (epoch, step,
extra) instead of each redefining it. ``BaseMetrics`` is the common ancestor;
``EpochMetrics`` is the canonical epoch-level container reused by benchmark
runners.

``TrialMetrics`` (with Pareto dominance logic) remains canonical in
``bioplausible.hyperopt.metrics`` and imports ``BaseMetrics`` here for its
shared shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BaseMetrics", "EpochMetrics"]


@dataclass(frozen=True, slots=True)
class BaseMetrics:
    """Common shape shared by training/epoch/trial metric containers.

    Attributes:
        epoch: Epoch index (1-based when set by a runner).
        step: Global update step.
        extra: Bucket for domain-specific metrics not covered by base fields.
    """

    epoch: int = 0
    step: int = 0
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpochMetrics(BaseMetrics):
    """Metrics for a single epoch.

    Attributes:
        epoch: Epoch index (1-based).
        train_loss: Mean training loss for the epoch.
        train_acc: Training accuracy for the epoch.
        val_loss: Mean validation loss for the epoch.
        val_acc: Validation accuracy for the epoch.
        epoch_time: Wall-clock seconds the epoch took.
    """

    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    epoch_time: float = 0.0
