"""
Training step dispatch utilities.

Provides the single canonical ``train_step`` dispatcher shared by all training
loops. Legacy ``CoreTrainer`` and ``TrainerConfig`` have been removed in favor
of ``SystemTrainer`` and ``ExperimentConfig`` (Sprint 7.6.10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeIs

import torch
from torch import nn

from computronium.core.ebm import EBMTrainer
from computronium.core.losses import compute_loss

if TYPE_CHECKING:
    from collections.abc import Callable


class _TrainerConfigProtocol(Protocol):
    """Protocol for trainer config used in dispatch (legacy compat)."""

    optimizer_kwargs: dict[str, object]
    extra: dict[str, object]
    grad_clip: float | None


class _LearningRuleOptimizer(Protocol):
    """Duck-typed view of a zoo ``LearningRuleOptimizer`` callable surface."""

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None: ...


def _is_learning_rule_optimizer(o: object) -> TypeIs[_LearningRuleOptimizer]:
    """Type-narrowing guard for the learning-rule-optimizer calling convention."""
    return bool(getattr(type(o), "_is_learning_rule", False))


def _make_ebm_trainer(config: _TrainerConfigProtocol, model: nn.Module) -> EBMTrainer:
    """Create an EBMTrainer from trainer config (legacy compat)."""
    return EBMTrainer(
        model,
        lr=config.optimizer_kwargs.get("lr", 0.01),
        free_steps=config.extra.get("free_steps", 30),
        nudged_steps=config.extra.get("nudged_steps"),
        beta=config.extra.get("beta", 0.1),
        clip_grad_norm=config.grad_clip,
    )


def bptt_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor] | None = None,
) -> dict[str, object]:
    """Canonical BPTT fallback shared by all lightweight training loops."""
    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(loss_fn, logits, y)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "logits": logits}


def _default_bptt_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, object]]:
    """Build the canonical BPTT closure bound to ``model`` and ``optimizer``."""

    def _step(x: torch.Tensor, y: torch.Tensor) -> dict[str, object]:
        return bptt_step(model, optimizer, x, y)

    return _step


def dispatch_train_step(  # ruff: ignore[complex-structure, too-many-return-statements, too-many-branches]
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    adapt_input: Callable[[torch.Tensor], torch.Tensor],
    bptt_step: Callable[[torch.Tensor, torch.Tensor], dict[str, object]] | None = None,
    propagator: object | None = None,
    optimizer: object | None = None,
    config: _TrainerConfigProtocol | None = None,
    record_path: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Dispatch one training step to the active learning rule.

    The single canonical ``train_step`` dispatcher. Routes through, in order:
    the energy-model path, an explicit learning-rule propagator, a kernel backend,
    a model-side ``train_step``, a learning-rule optimizer, then plain BPTT.
    """
    x = adapt_input(x)

    if bptt_step is None:
        if optimizer is None or not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(
                "dispatch_train_step reached the BPTT fallback with no "
                "torch optimizer; pass bptt_step or an optimizer."
            )
        bptt_step = _default_bptt_step(model, optimizer)

    def _record(path: str) -> None:
        if record_path is not None:
            record_path(path)

    # Kernel backend path - consumes the attached backend directly
    if (
        config is not None
        and hasattr(model, "_kernel_backend")
        and model._kernel_backend is not None
    ):
        _record("kernel")
        bespoke_step = getattr(model._kernel_backend, "kernel_train_step", None)
        contrastive_step_fn = getattr(model._kernel_backend, "contrastive_step", None)
        if bespoke_step is not None:
            kernel_metrics = bespoke_step(model, config, x, y, optimizer)
            if kernel_metrics is not None:
                return kernel_metrics
        elif contrastive_step_fn is not None:
            from computronium.core.trainer import _run_contrastive_kernel_step

            kernel_metrics = _run_contrastive_kernel_step(
                model, model._kernel_backend, config, x, y
            )
            if kernel_metrics is not None:
                return kernel_metrics
        else:
            from computronium.core.trainer import _run_kernel_train_step

            kernel_metrics = _run_kernel_train_step(
                model, model._kernel_backend, config, x, y, optimizer=optimizer
            )
            if kernel_metrics is not None:
                return kernel_metrics

    from computronium.core.ebm import EnergyModel

    match model:
        case EnergyModel() if config is not None:
            _record("energy")
            return _make_ebm_trainer(config, model).train_step(x, y)

    rule = propagator
    if rule is not None and _is_learning_rule_optimizer(rule):
        _record("propagator")
        return rule.step(x=x, target=y) or {}

    if hasattr(model, "train_step"):
        try:
            metrics = model.train_step(x, y)
        except NotImplementedError:
            metrics = None
        if metrics is not None:
            _record("model_train_step")
            return metrics

    rule = optimizer
    if rule is not None and _is_learning_rule_optimizer(rule):
        _record("propagator")
        return rule.step(x=x, target=y) or {}

    if getattr(model, "gradient_method", None) == "equilibrium":
        _record("implicit_equilibrium")
    else:
        _record("bptt")
    return bptt_step(x, y)


__all__ = [
    "_LearningRuleOptimizer",
    "_is_learning_rule_optimizer",
    "bptt_step",
    "dispatch_train_step",
]
