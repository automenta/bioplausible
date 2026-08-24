"""Canonical metrics dataclasses shared across the computronium codebase.

These provide a small, frozen+slots hierarchy so that domain-specific metric
containers (training/epoch/trial) share a common base shape (epoch, step,
extra) instead of each redefining it. ``BaseMetrics`` is the common ancestor;
``EpochMetrics`` is the canonical epoch-level container reused by benchmark
runners.

``TrialMetrics`` (with Pareto dominance logic) remains canonical in
``computronium.hyperopt.metrics`` and imports ``BaseMetrics`` here for its
shared shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["BaseMetrics", "EpochMetrics"]


@dataclass(frozen=True, slots=True)
class BaseMetrics:
    """Common shape shared by training/epoch/trial metric containers.

    Attributes:
        epoch: Epoch index (1-based when set by a runner).
        step: Global update step.
        extra: Bucket for domain-specific metrics not covered by base fields.

    The ``to_dict`` method serializes to a plain ``dict``, omitting ``None``
    values so the result round-trips cleanly through ``TrainingMetrics(**m)``
    reconstruction (fields absent from the dict fall back to their defaults).
    """

    epoch: int = 0
    step: int = 0
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dict, omitting ``None`` values.

        Uses :func:`dataclasses.asdict` for a deep copy of nested
        containers, then strips ``None`` entries so the result is
        JSON-serialisable and losslessly reconstructable via the
        class constructor (absent keys use defaults).
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


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
