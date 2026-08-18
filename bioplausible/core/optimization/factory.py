"""Strategy optimizer factory (REFACTOR.md §7).

Builds ``StrategyOptimizer`` instances from frozen configs. The default
registry covers the generic strategies; the MEP package augments the
registry with equilibrium-propagation strategies (no core dependency).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .config import StrategyConfig, StrategyOptimizerConfig
from .optimizer import StrategyOptimizer
from .strategies import (
    BackpropGradient,
    ConstraintStrategy,
    ErrorFeedback,
    FeedbackStrategy,
    GradientStrategy,
    HebbianGradient,
    MuonUpdate,
    NoConstraint,
    NoFeedback,
    PCGradient,
    PlainUpdate,
    SpectralConstraint,
    TargetPropGradient,
    UpdateStrategy,
)

__all__ = ["StrategyRegistry", "create_strategy_optimizer"]

GradientFactory = Callable[[StrategyConfig], GradientStrategy]
UpdateFactory = Callable[[StrategyConfig], UpdateStrategy]
ConstraintFactory = Callable[[StrategyConfig], ConstraintStrategy]
FeedbackFactory = Callable[[StrategyConfig], FeedbackStrategy]
StrategyFactory = GradientFactory | UpdateFactory | ConstraintFactory | FeedbackFactory

#: name -> constructor mapping; populate a copy before delegating to the
#: generic factory for MEP-specific strategies.
StrategyRegistry: dict[str, StrategyFactory] = {
    "backprop": lambda c: BackpropGradient(**c.kwargs),
    "target_prop": lambda c: TargetPropGradient(**c.kwargs),
    "hebbian": lambda c: HebbianGradient(**c.kwargs),
    "pc": lambda c: PCGradient(**c.kwargs),
    "plain": lambda c: PlainUpdate(**c.kwargs),
    "muon": lambda c: MuonUpdate(**c.kwargs),
    "none": lambda _c: NoConstraint(),
    "no_constraint": lambda _c: NoConstraint(),
    "spectral": lambda c: SpectralConstraint(**c.kwargs),
    "no_feedback": lambda _c: NoFeedback(),
    "error_feedback": lambda c: ErrorFeedback(**c.kwargs),
}


def make_strategy_optimizer(
    *,
    model: nn.Module,
    gradient: str,
    update: str,
    constraint: str | None = None,
    feedback: str | None = None,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    energy_fn: Callable | None = None,
    gradient_kwargs: dict | None = None,
    update_kwargs: dict | None = None,
    constraint_kwargs: dict | None = None,
    feedback_kwargs: dict | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Build a StrategyOptimizer from named strategy components.

    Generic permutation builder — any registered gradient × update ×
    constraint × feedback combination. Mirrors the MEP preset pattern
    without hardcoding EP-specific defaults.

    Args:
        model: Model to optimize.
        gradient: Gradient strategy name ("backprop", "target_prop", "hebbian", "pc", ...).
        update: Update strategy name ("plain", "muon", ...).
        constraint: Optional constraint strategy name ("spectral", "none", ...).
        feedback: Optional feedback strategy name ("error_feedback", "none", ...).
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay.
        max_grad_norm: Gradient clipping norm.
        energy_fn: Energy/loss callable for energy-based gradients.
        gradient_kwargs: Kwargs for gradient strategy.
        update_kwargs: Kwargs for update strategy.
        constraint_kwargs: Kwargs for constraint strategy.
        feedback_kwargs: Kwargs for feedback strategy.
        registry: Optional augmented strategy registry (MEP hook).

    Returns:
        Configured StrategyOptimizer.
    """
    from .config import StrategyConfig, StrategyOptimizerConfig

    gradient_kwargs = gradient_kwargs or {}
    update_kwargs = update_kwargs or {}
    constraint_kwargs = constraint_kwargs or {}
    feedback_kwargs = feedback_kwargs or {}

    config = StrategyOptimizerConfig(
        gradient=StrategyConfig(name=gradient, kwargs=gradient_kwargs),
        update=StrategyConfig(name=update, kwargs=update_kwargs),
        constraint=StrategyConfig(name=constraint, kwargs=constraint_kwargs)
        if constraint
        else None,
        feedback=StrategyConfig(name=feedback, kwargs=feedback_kwargs)
        if feedback
        else None,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )
    return create_strategy_optimizer(config, model, registry, energy_fn)


# Convenience presets for common permutations
def muon_tp(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    target_lr: float = 0.1,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Target Propagation + Muon orthogonalization ("MuonTP")."""
    return make_strategy_optimizer(
        model=model,
        gradient="target_prop",
        update="muon",
        constraint="spectral",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_kwargs={"target_lr": target_lr},
        energy_fn=energy_fn,
        registry=registry,
    )


def muon_pc(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    pc_weight: float = 0.1,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Predictive Coding + Muon orthogonalization ("MuonPC")."""
    return make_strategy_optimizer(
        model=model,
        gradient="pc",
        update="muon",
        constraint="spectral",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_kwargs={"pc_weight": pc_weight},
        energy_fn=energy_fn,
        registry=registry,
    )


def muon_hebbian(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    hebbian_lr: float = 0.01,
    use_oja: bool = True,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Hebbian + Muon orthogonalization ("MuonHebbian")."""
    return make_strategy_optimizer(
        model=model,
        gradient="hebbian",
        update="muon",
        constraint="spectral",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_kwargs={"hebbian_lr": hebbian_lr, "use_oja": use_oja},
        energy_fn=energy_fn,
        registry=registry,
    )


def plain_tp(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    target_lr: float = 0.1,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Target Propagation + plain SGD ("PlainTP")."""
    return make_strategy_optimizer(
        model=model,
        gradient="target_prop",
        update="plain",
        constraint="none",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_kwargs={"target_lr": target_lr},
        energy_fn=energy_fn,
        registry=registry,
    )


def plain_pc(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    pc_weight: float = 0.1,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Predictive Coding + plain SGD ("PlainPC")."""
    return make_strategy_optimizer(
        model=model,
        gradient="pc",
        update="plain",
        constraint="none",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_kwargs={"pc_weight": pc_weight},
        energy_fn=energy_fn,
        registry=registry,
    )


def plain_hebbian(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    hebbian_lr: float = 0.01,
    use_oja: bool = True,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Hebbian + plain SGD ("PlainHebbian")."""
    return make_strategy_optimizer(
        model=model,
        gradient="hebbian",
        update="plain",
        constraint="none",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_kwargs={"hebbian_lr": hebbian_lr, "use_oja": use_oja},
        energy_fn=energy_fn,
        registry=registry,
    )


def backprop_muon(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Backprop + Muon orthogonalization (alias for muon_backprop in MEP)."""
    return make_strategy_optimizer(
        model=model,
        gradient="backprop",
        update="muon",
        constraint="spectral",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        energy_fn=energy_fn,
        registry=registry,
    )


def backprop_plain(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    energy_fn: Callable | None = None,
    registry: dict[str, StrategyFactory] | None = None,
) -> StrategyOptimizer:
    """Backprop + plain SGD (standard SGD)."""
    return make_strategy_optimizer(
        model=model,
        gradient="backprop",
        update="plain",
        constraint="none",
        feedback="none",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        energy_fn=energy_fn,
        registry=registry,
    )


__all__ = [
    "StrategyRegistry",
    "backprop_muon",
    "backprop_plain",
    "create_strategy_optimizer",
    "make_strategy_optimizer",
    "muon_hebbian",
    "muon_pc",
    "muon_tp",
    "plain_hebbian",
    "plain_pc",
    "plain_tp",
]


def _resolve(
    spec: StrategyConfig, registry: dict[str, StrategyFactory], kind: str
) -> GradientStrategy | UpdateStrategy | ConstraintStrategy | FeedbackStrategy:
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
