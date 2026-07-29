"""EquiTile state type definitions (TypedDicts)."""

from typing import TypedDict

from bioplausible.equitile.core.config import EquiTileConfig


class EquiTileTrainingState(TypedDict, total=False):
    step_count: int
    error_ema: dict[int, float]
    warmup_steps: int
    total_steps: int


class EquiTileStateDict(TypedDict, total=False):
    model_state_dict: dict[str, object]
    task_type: str
    config: EquiTileConfig
    training: EquiTileTrainingState
    optim_io: dict[str, object] | None
    optim_importance: dict[str, object] | None
    optim_full: dict[str, object] | None
    lr_scheduler: dict[str, object] | None
    lr_scheduler_type: str | None
    metadata: dict[str, object] | None
