"""UI-agnostic training telemetry callbacks (Sprint 3.4).

Defines the ``ExecutionCallback`` protocol consumed by the NiceGUI demo
(Sprint 3) for live chart streaming. The protocol lives in its own
lightweight module (no torch/optuna imports) so that ``core.trainer`` can
consume it without pulling in the execution engine's heavy dependency
tree — preserving the Sprint 0.5 module-boundary goal.

Also provides a PyTorch Lightning compatible callback for logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bioplausible.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BaseExecutionCallback",
    "ExecutionCallback",
    "LightningExecutionCallback",
    "logger",
]

logger = get_logger()


@runtime_checkable
class _LightningLoggable(Protocol):
    """Structural view of ``pl.LightningModule.log`` (kept import-lightweight)."""

    def log(self, name: str, value: object, **kwargs: object) -> None: ...


@runtime_checkable
class ExecutionCallback(Protocol):
    """Telemetry hooks fired during a training run.

    Implementations receive scalar metrics only — never tensors, model
    references, or the trainer itself — so the engine stays UI-agnostic
    and callbacks cannot mutate training state. All hooks are optional:
    implement only the subset a consumer cares about.

    Hooks are invoked best-effort: a raising callback is logged and
    swallowed so a misbehaving UI listener can never break training.
    """

    def on_epoch_end(self, epoch: int, metrics: object) -> None:
        """Fired once per epoch after metrics are finalized.

        Args:
            epoch: 0-based epoch index just completed.
            metrics: ``TrainingMetrics`` value object for that epoch.
        """

    def on_step_end(
        self, step: int, loss: float, grad_norms: Mapping[str, float]
    ) -> None:
        """Fired after every training step.

        Args:
            step: 1-based global training step counter.
            loss: Scalar loss of the completed step (NaN if unavailable).
            grad_norms: Per-parameter ``L2`` gradient norms keyed by
                parameter name; empty when gradients are not materialized
                (e.g. kernel-mode models).
        """

    def on_settling_step(self, step: int, energy: float) -> None:
        """Fired when the model emits equilibrium/settling energy telemetry.

        Args:
            step: 1-based global training step counter.
            energy: Scalar energy proxy (e.g. EquiTile energy trajectory).
        """


class BaseExecutionCallback:
    """Convenience base: all hooks default to no-ops.

    Subclass and override only the hooks you need, e.g.::

        class ChartCallback(BaseExecutionCallback):
            def on_epoch_end(self, epoch, metrics):
                self.chart.append(metrics.train_loss)
    """

    def on_epoch_end(self, epoch: int, metrics: object) -> None:
        """No-op default. Override to consume epoch telemetry."""

    def on_step_end(
        self, step: int, loss: float, grad_norms: Mapping[str, float]
    ) -> None:
        """No-op default. Override to consume step telemetry."""

    def on_settling_step(self, step: int, energy: float) -> None:
        """No-op default. Override to consume settling energy telemetry."""


class LightningExecutionCallback(BaseExecutionCallback):
    """ExecutionCallback that logs to PyTorch Lightning's logging interface.

    This allows CoreTrainer to be used with PL's logging ecosystem by
    implementing the ExecutionCallback protocol and forwarding metrics
    to a PL module's ``log`` method.
    """

    def __init__(self, lightning_module: _LightningLoggable) -> None:
        """
        Args:
            lightning_module: A PyTorch Lightning module instance with a
                ``log`` method (e.g., ``pl.LightningModule``).
        """
        self.lightning_module = lightning_module

    def on_epoch_end(self, epoch: int, metrics: object) -> None:
        """Log epoch metrics to PL."""
        del epoch  # epoch index is implicit in PL's step counter
        for key, value in metrics.__dict__.items():
            if not key.startswith("_") and isinstance(value, (int, float)):
                self.lightning_module.log(f"epoch_{key}", value, on_epoch=True)

    def on_step_end(
        self, step: int, loss: float, grad_norms: Mapping[str, float]
    ) -> None:
        """Log step metrics to PL."""
        del step  # PL maintains its own global step counter
        self.lightning_module.log("train_loss", loss, on_step=True, prog_bar=True)
        for name, norm in grad_norms.items():
            self.lightning_module.log(f"grad_norm/{name}", norm, on_step=True)
