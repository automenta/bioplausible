"""Dependency-free composite strategy optimizer (REFACTOR.md §7).

``StrategyOptimizer`` composes gradient/update/constraint/feedback strategies
into a working ``torch.optim.Optimizer``. The MEP package subclasses it to
add equilibrium-propagation strategies without duplicating the loop.

EP gradient strategies signal that they need the full model input/energy
context by setting ``requires_energy=True``; the loop then forwards the
``energy_fn``/``structure_fn`` callables and the input tensors to
``compute_gradients``.
"""

from collections.abc import Callable, Iterable
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer

from .strategies import (
    ConstraintStrategy,
    FeedbackStrategy,
    GradientStrategy,
    NoConstraint,
    NoFeedback,
    UpdateStrategy,
)

__all__ = ["StrategyOptimizer"]


def _get_structure(model: nn.Module) -> list[dict[str, object]]:
    """Extract transition-module structure for energy-based gradients."""
    if hasattr(model, "transition_modules"):
        try:
            return [
                {"type": "layer", "module": mod}
                for mod in model.transition_modules()
            ]
        except NotImplementedError:
            pass
    return []


class StrategyOptimizer(Optimizer):
    """Composable optimizer built from strategy components.

    Args:
        params: Iterable of parameters to optimize.
        gradient: Strategy for computing gradients.
        update: Strategy for transforming gradients to updates.
        constraint: Strategy for enforcing constraints (default: none).
        feedback: Strategy for error feedback (default: none).
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight-decay coefficient.
        model: Model instance (required by energy-based gradient strategies).
        max_grad_norm: Maximum gradient norm for clipping.
        energy_fn: Optional energy / loss callable for EP gradients.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        gradient: GradientStrategy,
        update: UpdateStrategy,
        constraint: ConstraintStrategy | None = None,
        feedback: FeedbackStrategy | None = None,
        lr: float = 0.02,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        model: nn.Module | None = None,
        max_grad_norm: float = 10.0,
        energy_fn: Callable | None = None,
    ):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= momentum < 1):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")

        defaults: dict[str, object] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

        self.model = model
        self.gradient = gradient
        self.update = update
        self.constraint = constraint or NoConstraint()
        self.feedback = feedback or NoFeedback()
        self._energy_fn = energy_fn

        loss_type = getattr(gradient, "loss_type", "mse")
        softmax_temperature = getattr(gradient, "softmax_temperature", 1.0)
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature

        self._free_states: list[torch.Tensor] | None = None
        self._nudged_states: list[torch.Tensor] | None = None
        self._last_input: torch.Tensor | None = None

        self._error_beta = getattr(feedback, "beta", 0.9)
        self._use_error_feedback = not isinstance(feedback, NoFeedback)

    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float] | None = None,
        x: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float | None:
        """Perform one optimization step.

        Supports backprop mode (``loss.backward()`` + ``step()``), explicit
        EP mode (``step(x=x, target=y)``), and wrapped-model EP mode.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if getattr(self.gradient, "requires_energy", False):
            if x is None and self._last_input is None:
                raise ValueError(
                    "Energy-based gradient strategies require an x tensor. "
                    "Pass x to step() or call model(x) first."
                )
            if target is None:
                raise ValueError(
                    "Energy-based gradient strategies require a target tensor"
                )
            if self.model is None:
                raise ValueError(
                    "Model must be provided to the optimizer for energy-based "
                    "gradient strategies"
                )

            x_input = x if x is not None else self._last_input
            if x_input is None:
                raise ValueError("Input tensor is None")

            self.gradient.compute_gradients(
                self.model,
                x_input,
                target,
                energy_fn=self._energy_fn,
                structure_fn=_get_structure,
                **kwargs,
            )

        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)

                    group["error_beta"] = self._error_beta
                    group["use_error_feedback"] = self._use_error_feedback

                    update = self.update.transform_gradient(
                        param, param.grad, state, group
                    )

                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(update)

                    param.data.mul_(1 - group["weight_decay"] * group["lr"])
                    param.data.add_(buf, alpha=-group["lr"])

                    self.constraint.enforce(param, state, group)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear accumulated gradients."""
        if self.param_groups:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad.zero_()

    def state_dict(self) -> dict[str, object]:
        """Return optimizer state augmented with strategy names."""
        state = super().state_dict()
        state["strategy_config"] = {
            "gradient": type(self.gradient).__name__,
            "update": type(self.update).__name__,
            "constraint": type(self.constraint).__name__,
            "feedback": type(self.feedback).__name__,
        }
        return cast("dict[str, object]", state)