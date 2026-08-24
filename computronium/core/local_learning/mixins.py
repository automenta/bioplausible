"""Multi-optimizer and scheduler management for local-learning models.

Generic weight/importance/full optimizer split. Moved from
``equitile.training.optimizer_mixin`` during generification; consumer models
(EquiTile and friends) provide a config object exposing ``learning_rate``,
``importance_lr``, and ``mode`` under ``self.equitile_config``.
"""

from typing import Protocol

import torch
from torch import nn

from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer

__all__ = [
    "LocalLearningConfigProtocol",
    "MultiOptimizerMixin",
]


class LocalLearningConfigProtocol(Protocol):
    """Minimal config surface the optimizer mixin depends on.

    Keeps ``core/*`` free of the ``equitile`` package at type-check time: any
    consumer (EquiTile or a new zoo algorithm) supplies a config exposing
    these three fields.
    """

    learning_rate: float
    importance_lr: float
    mode: str


class MultiOptimizerMixin:
    """Mixin for tile-based local-learning optimizer and scheduler management."""

    # Type hints for attributes expected from the consumer model
    W_in: nn.Linear
    W_out: nn.Linear
    tile_importance: nn.Parameter
    edge_importance: nn.Parameter
    equitile_config: LocalLearningConfigProtocol
    _optim_io: torch.optim.Optimizer
    _optim_importance: torch.optim.Optimizer
    _optim_full: torch.optim.Optimizer | None
    _lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None
    _lr_scheduler_type: str | None
    _step_count: int
    _warmup_steps: int
    _warmup_start_lr: float
    _total_steps: int

    def extra_importance_params(self) -> list[nn.Parameter] | None:
        """Extra parameters for the importance optimizer (e.g. per-tile LR scale)."""
        return None

    def importance_params(self) -> list[nn.Parameter]:
        """Importance-group parameters, including optional subclass extras."""
        extra = self.extra_importance_params()
        return [self.tile_importance, self.edge_importance] + (extra or [])

    def _setup_optimizers(self) -> None:
        """Initialize optimizers explicitly."""
        self._optim_io = create_optimizer(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            OptimizerConfig(name="adam", lr=self.equitile_config.learning_rate),
        )

        self._optim_importance = create_optimizer(
            self.importance_params(),
            OptimizerConfig(name="adam", lr=self.equitile_config.importance_lr),
        )

        if self.equitile_config.mode in ("backprop", "ep"):
            self._optim_full = create_optimizer(
                self,
                OptimizerConfig(name="adam", lr=self.equitile_config.learning_rate),
            )

    def reset_optimizers(self) -> None:
        """Reset optimizers (e.g. after topology change)."""
        self._setup_optimizers()
        if self._lr_scheduler is not None:
            self.configure_lr_scheduler(
                scheduler_type=self._lr_scheduler_type,
                total_steps=self._total_steps,
                warmup_steps=self._warmup_steps,
            )

    def configure_lr_scheduler(
        self,
        scheduler_type: str = "cosine",
        total_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        warmup_steps: int = 100,
    ):
        """Configure learning rate scheduler."""
        self._lr_scheduler_type = scheduler_type

        if scheduler_type == "cosine":
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optim_io,
                T_max=total_steps - warmup_steps,
                eta_min=self.equitile_config.learning_rate * min_lr_ratio,
            )
        elif scheduler_type == "step":
            self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self._optim_io,
                step_size=total_steps // 5,
                gamma=0.5,
            )
        elif scheduler_type == "linear":
            self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self._optim_io,
                start_factor=1.0,
                end_factor=min_lr_ratio,
                total_iters=total_steps - warmup_steps,
            )

        self._warmup_steps = warmup_steps
        self._warmup_start_lr = self.equitile_config.learning_rate * 0.1
        self._total_steps = total_steps

    def step_lr_scheduler(self):
        """Step the learning rate scheduler."""
        if self._lr_scheduler is None:
            return

        if hasattr(self, "_warmup_steps") and self._step_count < self._warmup_steps:
            warmup_progress = self._step_count / self._warmup_steps
            current_lr = (
                self._warmup_start_lr
                + (self.equitile_config.learning_rate - self._warmup_start_lr)
                * warmup_progress
            )

            for param_group in self._optim_io.param_groups:
                param_group["lr"] = current_lr
        else:
            self._lr_scheduler.step()

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self._optim_io.param_groups:
            return param_group["lr"]
        return self.equitile_config.learning_rate
