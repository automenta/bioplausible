"""
Training utilities for the TaskProtocol interface.

Moved from ``hyperopt/tasks.py`` during Phase 3.1 task hierarchy merge.
"""

import logging
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from bioplausible.domains.base import DomainType

__all__ = [
    "TaskProtocol",
    "_TaskTrainer",
    "_resolve_task_loss",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class TaskProtocol(Protocol):
    """Structural interface for experiment tasks.

    All task classes should satisfy this protocol.  Type annotations should
    use ``TaskProtocol`` instead of ``BaseTask`` to allow duck-typed task
    implementations.
    """

    name: str
    device: str
    quick_mode: bool

    @property
    def input_dim(self) -> int | None: ...

    @property
    def output_dim(self) -> int: ...

    @property
    def task_type(self) -> str: ...

    def setup(self) -> None: ...

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def create_trainer(self, model: nn.Module, **kwargs) -> object: ...

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> dict[str, float]: ...


def _resolve_task_loss(task: TaskProtocol) -> nn.Module:
    """Pick a torch loss module matching the task's output geometry.

    Regression tasks (``task_type == "tabular"`` with ``output_dim == 1``
    — e.g. California Housing) emit float ``[B, 1]`` targets and must use
    MSELoss; everything else (vision/lm/discrete-tabular) treats the
    target as a class index and uses CrossEntropyLoss.
    """
    if task.task_type == DomainType.TABULAR and task.output_dim == 1:
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


class _TaskTrainer:
    """Lightweight task-protocol trainer.

    Thin wrapper around ``CoreTrainer`` that delegates training to
    ``CoreTrainer.from_task()``.  The wrapper exists to preserve the
    ``train_*``-prefixed metric shape and inline validation behaviour
    expected by ``hyperopt`` callers.
    """

    def __init__(
        self,
        model: nn.Module,
        task: TaskProtocol,
        device: str = "cpu",
        optimizer=None,
        epochs: int = 1,
        batches_per_epoch: int = 1,
        grad_clip: float | None = None,
        use_compile: bool = False,
        track_energy: bool = False,
        ablation_tags: dict | None = None,
        output_dir: str = "",
        **kwargs,
    ):
        from bioplausible.core.trainer import CoreTrainer

        self._trainer = CoreTrainer.from_task(
            model=model,
            task=task,
            device=device,
            optimizer=optimizer,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            grad_clip=grad_clip,
            use_compile=use_compile,
            track_energy=track_energy,
            ablation_tags=ablation_tags or {},
            output_dir=output_dir,
            batch_size=kwargs.pop("batch_size", 32),
        )
        self.model = model
        self.task = task
        self.epochs = epochs

    def train_epoch(self) -> dict[str, float]:
        """Run one epoch of training and return aggregated metrics."""
        import time

        epoch_t0 = time.time()
        raw = self._trainer.train_epoch()

        metrics: dict[str, float] = {}
        for k, v in raw.items():
            if k in ("loss", "accuracy"):
                metrics[f"train_{k}"] = v
            elif k == "samples_seen":
                continue
            else:
                metrics[k] = v
        metrics["loss"] = metrics.get("train_loss", 0.0)
        metrics["accuracy"] = metrics.get("train_accuracy", 0.0)

        metrics["val_loss"] = float("nan")
        metrics["val_accuracy"] = float("nan")
        try:
            val_raw = self._trainer._validate(1)
            metrics["val_loss"] = val_raw.get("val_loss", float("nan"))
            metrics["val_accuracy"] = val_raw.get("val_accuracy", float("nan"))
            if "val_perplexity" in val_raw:
                metrics["val_perplexity"] = val_raw["val_perplexity"]
        except (NotImplementedError, RuntimeError) as e:
            logger.warning("Validation skipped for %s: %s", self.task.name, e)

        metrics["time"] = time.time() - epoch_t0
        return metrics
