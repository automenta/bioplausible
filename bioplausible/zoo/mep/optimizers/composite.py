"""
Composite Optimizer: Strategy pattern-based optimizer (MEP edition).

Subclasses :class:`~bioplausible.core.optimization.optimizer.StrategyOptimizer`
(REFACTOR.md §7); the MEP layer supplies the EP-specific energy function and
re-exports the generic loop. See ``core/optimization`` for the base class.
"""

from collections.abc import Iterable

from torch import nn

from bioplausible.core.optimization import StrategyOptimizer

from .energy import EnergyFunction

__all__ = [
    "CompositeOptimizer",
]


class CompositeOptimizer(StrategyOptimizer):
    """
    Composable MEP optimizer built from strategy components.

    Example usage:
        optimizer = CompositeOptimizer(
            model.parameters(),
            gradient=EPGradient(beta=0.5, settle_steps=20),
            update=MuonUpdate(ns_steps=5),
            constraint=SpectralConstraint(gamma=0.95),
            feedback=ErrorFeedback(beta=0.9),
            lr=0.02,
            model=model,
        )

    Attributes:
        model: The model being optimized (for EP).
        gradient: Strategy for computing gradients.
        update: Strategy for transforming gradients.
        constraint: Strategy for enforcing constraints.
        feedback: Strategy for error accumulation.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        gradient,
        update,
        constraint=None,
        feedback=None,
        lr: float = 0.02,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        model: nn.Module | None = None,
        max_grad_norm: float = 10.0,
    ):
        """
        Initialize composite optimizer.

        Args:
            params: Iterable of parameters to optimize.
            gradient: Strategy for computing gradients.
            update: Strategy for transforming gradients to updates.
            constraint: Strategy for enforcing constraints (default: none).
            feedback: Strategy for error feedback (default: none).
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: Weight-decay coefficient.
            model: Model instance (required for EP gradient strategies).
            max_grad_norm: Maximum gradient norm for clipping.
        """
        loss_type = getattr(gradient, "loss_type", "mse")
        softmax_temperature = getattr(gradient, "softmax_temperature", 1.0)
        self._energy_fn = EnergyFunction(
            loss_type=loss_type, softmax_temperature=softmax_temperature
        )
        # Energy function is shared with the base energy-based gradient path.
        super().__init__(
            params,
            gradient=gradient,
            update=update,
            constraint=constraint,
            feedback=feedback,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            model=model,
            max_grad_norm=max_grad_norm,
            energy_fn=self._energy_fn,
        )
