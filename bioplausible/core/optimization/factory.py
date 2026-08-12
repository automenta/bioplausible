"""Strategy optimizer factory (REFACTOR.md §7).

Builds ``StrategyOptimizer`` instances from frozen configs. The default
registry covers the generic strategies; the MEP package augments the
registry with equilibrium-propagation strategies (no core dependency).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from .config import StrategyConfig, StrategyOptimizerConfig
from .optimizer import StrategyOptimizer
from .strategies import (
    BackpropGradient,
    ErrorFeedback,
    HebbianGradient,
    MuonUpdate,
    NoConstraint,
    NoFeedback,
    PlainUpdate,
    SpectralConstraint,
    TargetPropGradient,
)

__all__ = ["StrategyRegistry", "create_strategy_optimizer"]

StrategyFactory = Callable[[StrategyConfig], Any]

#: name -> constructor mapping; populate a copy before delegating to the
#: generic factory for MEP-specific strategies.
StrategyRegistry: dict[str, StrategyFactory] = {
    "backprop": lambda c: BackpropGradient(**c.kwargs),
    "target_prop": lambda c: TargetPropGradient(**c.kwargs),
    "hebbian": lambda c: HebbianGradient(**c.kwargs),
    "plain": lambda c: PlainUpdate(**c.kwargs),
    "muon": lambda c: MuonUpdate(**c.kwargs),
    "none": lambda _c: NoConstraint(),
    "no_constraint": lambda _c: NoConstraint(),
    "spectral": lambda c: SpectralConstraint(**c.kwargs),
    "no_feedback": lambda _c: NoFeedback(),
    "error_feedback": lambda c: ErrorFeedback(**c.kwargs),
}


def _resolve(
    spec: StrategyConfig, registry: dict[str, StrategyFactory], kind: str
) -> Any:
    factory = registry.get(spec.name)
    if factory is None:
        raise ValueError(
            f"Unknown {kind} strategy: {spec.name!r}. Available: {sorted(registry)}"
        )
    return factory(spec)


def create_strategy_optimizer(
    config: StrategyOptimizerConfig,
    model: nn.Module | None = None,
    registry: dict[str, StrategyFactory] | None = None,
    energy_fn: Callable | None = None,
) -> StrategyOptimizer:
    """Build a ``StrategyOptimizer`` from a frozen config.

    Args:
        config: Strategy permutation + hyperparameters.
        model: Model to optimize (required by energy-based gradients).
        registry: Optional augmented strategy registry (MEP hook).
        energy_fn: Energy / loss callable for energy-based gradients.

    Returns:
        A configured ``StrategyOptimizer``.
    """
    reg = {**StrategyRegistry, **(registry or {})}
    gradient = _resolve(config.gradient, reg, "gradient")
    update = _resolve(config.update, reg, "update")
    constraint = (
        _resolve(config.constraint, reg, "constraint")
        if config.constraint is not None
        else None
    )
    feedback = (
        _resolve(config.feedback, reg, "feedback")
        if config.feedback is not None
        else None
    )
    return StrategyOptimizer(
        model.parameters() if model is not None else [],
        gradient=gradient,
        update=update,
        constraint=constraint,
        feedback=feedback,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        model=model,
        energy_fn=energy_fn,
    )
