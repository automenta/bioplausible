"""
Training utilities for the TaskProtocol interface.

Moved from ``hyperopt/tasks.py`` during Phase 3.1 task hierarchy merge.
"""

import contextlib
import time
from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn

from computronium.core.logging import get_logger
from computronium.core.losses import compute_loss
from computronium.domains.base import DomainType
from computronium.execution._guards import SafetyConfig, SafetyWrapper

__all__ = [
    "TaskProtocol",
    "_TaskTrainer",
    "_resolve_task_loss",
]

logger = get_logger()


class _TrackerProtocol(Protocol):
    """Protocol for experiment trackers."""

    def log_metrics(self, metrics: dict[str, float]) -> None: ...


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


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Classification accuracy; 0.0 for non-index targets (regression)."""
    if y.dtype not in (torch.long, torch.int, torch.int32, torch.int64):  # ruff: ignore[literal-membership]
        return 0.0
    preds = logits[:, -1, :] if logits.dim() == 3 else logits
    return (preds.argmax(-1) == y).float().mean().item()


class _TaskTrainer:
    """Lightweight task-protocol trainer for plain ``nn.Module`` models.

    Runs the canonical forward/loss/backward/step loop over task batches with
    inline validation, preserving the ``train_*``-prefixed metric shape
    expected by hyperopt callers.

    Supports:
    - Learning rate schedulers (via ``scheduler_type``/``scheduler_kwargs``)
    - Experiment tracking (via ``tracker``)
    - Numerical safety (via ``safety_config``)
    - Energy tracking placeholder (via ``track_energy``; no-op for plain modules)
    """

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        model: nn.Module,
        task: TaskProtocol,
        device: str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 1,
        batches_per_epoch: int = 1,
        grad_clip: float | None = None,
        use_compile: bool = False,
        track_energy: bool = False,
        ablation_tags: dict | None = None,
        output_dir: str = "",
        tracker: _TrackerProtocol | None = None,
        safety_config: SafetyConfig | None = None,
        scheduler_type: str | None = None,
        scheduler_kwargs: dict | None = None,
        **kwargs,
    ):
        self.model: nn.Module = cast("nn.Module", model)
        self.task = task
        self.device = device
        self.epochs = epochs
        self.batches_per_epoch = int(kwargs.pop("steps", batches_per_epoch))
        self.episodes_per_epoch = self.batches_per_epoch
        self.batch_size = int(kwargs.pop("batch_size", 32))
        self.eval_batches = kwargs.pop("eval_batches", None)
        self.grad_clip = grad_clip
        self.track_energy = track_energy
        self.ablation_tags = ablation_tags or {}
        self.output_dir = output_dir
        self._loss = _resolve_task_loss(task)
        self.tracker: _TrackerProtocol | None = tracker
        self.safety_config = safety_config or SafetyConfig()
        self.safety_wrapper = SafetyWrapper(self.safety_config)
        if use_compile:
            self.model = cast("nn.Module", torch.compile(self.model))
        lr = float(kwargs.pop("lr", 1e-3))
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if scheduler_type:
            self._create_scheduler(scheduler_type, scheduler_kwargs or {})

    def _create_scheduler(
        self, scheduler_type: str, scheduler_kwargs: dict
    ) -> None:
        """Create learning rate scheduler from type and kwargs."""
        scheduler_type_lower = scheduler_type.lower()
        if scheduler_type_lower == "cosine":
            t_max = scheduler_kwargs.get("t_max", self.epochs)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, **{
                    k: v for k, v in scheduler_kwargs.items() if k != "t_max"
                }
            )
        elif scheduler_type_lower == "step":
            step_size = scheduler_kwargs.get("step_size", 10)
            gamma = scheduler_kwargs.get("gamma", 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type_lower == "linear":
            start_factor = scheduler_kwargs.get("start_factor", 1.0)
            end_factor = scheduler_kwargs.get("end_factor", 0.01)
            total_iters = scheduler_kwargs.get("total_iters", self.epochs)
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters,
            )
        elif scheduler_type_lower == "cosine_warmup":
            from torch.optim.lr_scheduler import SequentialLR
            warmup_iters = scheduler_kwargs.get("warmup_iters", 5)
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=scheduler_kwargs.get("warmup_start_factor", 0.01),
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            t_max = scheduler_kwargs.get("t_max", max(1, self.epochs - warmup_iters))
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_iters],
            )
        else:
            logger.warning("Unknown scheduler type: %s", scheduler_type)

    def _step_scheduler(self) -> None:
        """Step the learning rate scheduler if present."""
        if self.scheduler is not None:
            self.scheduler.step()

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to tracker if available."""
        if self.tracker is not None and hasattr(self.tracker, "log_metrics"):
            self.tracker.log_metrics(metrics)

    def _safe_step(self, loss: torch.Tensor) -> tuple[bool, dict[str, object]]:
        """Execute a safe backward/step with gradient clipping and NaN checks."""
        return self.safety_wrapper.safe_backward_and_step(
            loss, self.optimizer, self.model, self.grad_clip
        )

    def _run_batches(self, split: str, no_grad: bool = False) -> dict[str, float]:
        total_loss = 0.0
        total_acc = 0.0
        total_energy = 0.0
        batches = self.eval_batches if (no_grad and self.eval_batches) else 1
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            for _ in range(int(batches)):
                x, y = self.task.get_batch(split=split, batch_size=self.batch_size)
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = compute_loss(self._loss, logits, y)
                if not no_grad:
                    success, step_info = self._safe_step(loss)
                    if not success:
                        logger.warning(
                            "Step failed: %s, failure %d/%d",
                            step_info.get("error"),
                            self.safety_wrapper.consecutive_failures,
                            self.safety_config.max_nan_retries,
                        )
                        if self.safety_wrapper.should_abort():
                            self.safety_wrapper.handle_failure(self.optimizer)
                            raise RuntimeError(
                                f"Training aborted after {self.safety_config.max_nan_retries} consecutive failures"
                            )
                total_loss += loss.item()
                total_acc += _accuracy(logits, y)
        n = int(batches)
        result = {"loss": total_loss / n, "accuracy": total_acc / n}
        if self.track_energy:
            result["energy"] = total_energy / n
        return result

    def train_epoch(self) -> dict[str, float]:
        """Run one epoch of training and return aggregated metrics."""
        epoch_t0 = time.time()
        self.model.train()
        train_metrics = self._run_batches("train")
        metrics: dict[str, float] = {f"train_{k}": v for k, v in train_metrics.items()}
        metrics |= train_metrics

        metrics["val_loss"] = float("nan")
        metrics["val_accuracy"] = float("nan")
        try:
            val_metrics = self._run_batches("val", no_grad=True)
            metrics["val_loss"] = val_metrics["loss"]
            metrics["val_accuracy"] = val_metrics["accuracy"]
        except (NotImplementedError, RuntimeError, KeyError, ValueError) as e:
            logger.warning("Validation skipped for %s: %s", self.task.name, e)

        # Step scheduler at epoch boundary
        self._step_scheduler()

        metrics["time"] = time.time() - epoch_t0
        metrics["samples_seen"] = float(self.batches_per_epoch * self.batch_size)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        # Log to tracker
        self._log_metrics(metrics)

        return metrics
