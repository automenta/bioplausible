"""Composable optimizer configuration (REFACTOR.md §7).

Frozen runtime config describing a full strategy permutation: gradient /
update / constraint / feedback selected by name with their hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["StrategyConfig", "StrategyOptimizerConfig"]


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Named strategy instantiation.

    Attributes:
        name: Strategy class name (e.g. ``"backprop"``, ``"muon"``,
            ``"spectral"``, ``"error_feedback"``). MEP-added strategies are
            resolved by name with no core dependency.
        **kwargs: Constructor arguments for the strategy.
    """

    name: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategyOptimizerConfig:
    """Composable optimizer configuration.

    Attributes:
        gradient: Gradient-computation strategy (e.g. "backprop").
        update: Update-transformation strategy (e.g. "plain").
        constraint: Optional parameter-constraint strategy.
        feedback: Optional error-feedback strategy.
        lr: Learning rate.
        momentum: Momentum factor (0 disables momentum).
        weight_decay: Weight-decay coefficient.
        max_grad_norm: Gradient-norm clipping bound.
    """

    gradient: StrategyConfig = field(
        default_factory=lambda: StrategyConfig(name="backprop")
    )
    update: StrategyConfig = field(default_factory=lambda: StrategyConfig(name="plain"))
    constraint: StrategyConfig | None = None
    feedback: StrategyConfig | None = None
    lr: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 0.0005
    max_grad_norm: float = 10.0