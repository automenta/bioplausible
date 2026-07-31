"""Shared dataclasses for MEP benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time: float


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Results for a single optimizer."""

    name: str
    metrics: list[EpochMetrics]
    total_time: float
    best_val_acc: float
    final_train_acc: float
