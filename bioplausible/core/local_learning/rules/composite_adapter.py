"""Adapt a MEP :class:`CompositeOptimizer` to the learning-rule surface.

MEP presets compose gradient/update/constraint/feedback strategies into a
``CompositeOptimizer`` whose ``step()`` already accepts ``(x, target)``. This
adapter adds the ``_is_learning_rule`` marker so ``dispatch_train_step`` routes
those presets as learning-rule optimizers, unifying the calling convention with
:class:`LearningRuleOptimizer`. Kept dependency-free (no zoo import) per the
L1 core layering rule.
"""

from typing import TYPE_CHECKING, Protocol, cast

from .base import LearningRuleOptimizer

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = ["CompositeOptimizerAdapter"]


class _CompositeLike(Protocol):
    """Structural interface of the wrapped MEP composite optimizer."""

    param_groups: list[dict[str, object]]
    model: nn.Module

    def step(self, *, x: torch.Tensor | None, target: torch.Tensor | None) -> float | None: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...


class CompositeOptimizerAdapter(LearningRuleOptimizer):
    """Present a MEP composite optimizer through ``step(x, target)``.

    Delegates the update to the wrapped composite, which owns the parameter
    and optimizer state. The base :class:`LearningRuleOptimizer` machinery
    (``_is_learning_rule`` marker, ``buffers``) is retained so callers can
    treat EP presets identically to core learning rules.
    """

    def __init__(self, composite: _CompositeLike):
        params = [p for g in composite.param_groups for p in cast("list", g["params"])]
        super().__init__(params, model=composite.model)
        self._composite = composite

    def step(self, x: torch.Tensor | None = None, target: torch.Tensor | None = None) -> None:
        # ``x``/``target`` are optional so the adapter preserves the composite's
        # backprop mode (``loss.backward(); step()``) as well as its EP mode
        # (``step(x=x, target=y)``).
        self._composite.step(x=x, target=target)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._composite.zero_grad(set_to_none)
