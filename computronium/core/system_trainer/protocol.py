"""Protocols for SystemTrainer composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from computronium.core.joint.transition import PlasticityPrimitive
from computronium.ontology import (
    CreditAssignment,
    Geometry,
    ParameterUpdate,
    StateDynamics,
    Substrate,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from torch import Tensor

    from computronium.state import SystemContext


class JointSystem[
    TS: Substrate,
    TG: Geometry,
    TD: StateDynamics,
    TP: PlasticityPrimitive,
    TC: CreditAssignment,
    TU: ParameterUpdate,
](Protocol):
    """Protocol for 6-D joint systems."""

    substrate: TS
    geometry: TG
    dynamics: TD
    plasticity: TP
    credit: TC
    update: TU

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]: ...
    def forward(self, x: Tensor) -> Tensor: ...

    @property
    def device(self) -> torch.device: ...
    @property
    def context(self) -> SystemContext: ...
    def to_spec(self) -> dict[str, object]: ...
    @classmethod
    def from_spec(cls, spec: dict) -> JointSystem: ...


TS = TypeVar("TS", bound=Substrate)
TG = TypeVar("TG", bound=Geometry)
TD = TypeVar("TD", bound=StateDynamics)
TC = TypeVar("TC", bound=CreditAssignment)
TU = TypeVar("TU", bound=ParameterUpdate)


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
    "TC",
    "TD",
    "TG",
    "TS",
    "TU",
    "JointSystem",
    "SystemTrainerConfig",
    "_DataProvider",
]
