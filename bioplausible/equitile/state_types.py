"""EquiTile state type definitions (TypedDicts)."""

from typing import Any, TypedDict

from .config import EquiTileConfig


class EquiTileTrainingState(TypedDict, total=False):
    step_count: int
    error_ema: dict[int, float]
    warmup_steps: int
    total_steps: int


class EquiTileStateDict(TypedDict, total=False):
    model_state_dict: dict[str, Any]
    task_type: str
    config: EquiTileConfig
    training: EquiTileTrainingState
    optim_io: dict[str, Any] | None
    optim_importance: dict[str, Any] | None
    optim_full: dict[str, Any] | None
    lr_scheduler: dict[str, Any] | None
    lr_scheduler_type: str | None
    metadata: dict[str, Any] | None
