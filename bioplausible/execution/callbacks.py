"""UI-agnostic training telemetry callbacks (Sprint 3.4).

Defines the ``ExecutionCallback`` protocol consumed by the NiceGUI demo
(Sprint 3) for live chart streaming. The protocol lives in its own
lightweight module (no torch/optuna imports) so that ``core.trainer`` can
consume it without pulling in the execution engine's heavy dependency
tree — preserving the Sprint 0.5 module-boundary goal.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

__all__ = ["BaseExecutionCallback", "ExecutionCallback", "logger"]

logger = logging.getLogger(__name__)


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
