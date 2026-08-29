"""Configuration for SystemTrainer.

This module contains only the SystemTrainerConfig dataclass.
Protocols are in protocol.py, serialization utilities are in spec.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from torch import Tensor

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class SystemTrainerConfig:
    """Configuration for SystemTrainer.

    Attributes:
        max_epochs: Number of training epochs
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        device: Target device ("auto", "cpu", "cuda", "mps")
        grad_clip: Gradient clipping norm (applied in ParameterUpdate if supported)
        track_energy: Track energy metrics during training
        track_flops: Track FLOPs during training
        track_memory: Track memory usage
        log_every_n_steps: Logging frequency
        seed: Random seed
        deterministic: Use deterministic algorithms
    """

    max_epochs: int = 10
    batch_size: int = 64
    val_batch_size: int | None = None
    device: str = "auto"
    grad_clip: float | None = 1.0
    track_energy: bool = True
    track_flops: bool = True
    track_memory: bool = True
    log_every_n_steps: int = 10
    seed: int = 42
    deterministic: bool = False


class _DataProvider(Protocol):
    """Protocol for data providers (DataLoader, Task, etc.)."""

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def __len__(self) -> int: ...


__all__ = [
    "SystemTrainerConfig",
    "_DataProvider",
]
