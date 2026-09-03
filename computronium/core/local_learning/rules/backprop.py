"""
Standard autograd (backpropagation) wrapper.

Classes: Backprop
"""

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn

from computronium.core.registry import LocalityLevel, register_credit_assignment

from .base import LearningRuleOptimizer

__all__ = [
    "Backprop",
]


@register_credit_assignment(
    "backprop",
    family="backprop",
    locality_level=LocalityLevel.GLOBAL,
    bio_plausibility_score=0.3,
    credit_assignment_type="backpropagation",
    requires_backward=True,
    tags=["backprop", "standard"],
    description=(
        "Standard backpropagation via autograd: forward pass, loss"
        " computation, backward pass, and SGD parameter update."
    ),
)
class Backprop(LearningRuleOptimizer):
    """Standard backpropagation via autograd."""

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        loss_fn: str = "cross_entropy",
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.loss_fn = loss_fn

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("Backprop requires target")

        self.model.train()
        output = self.model(x)

        if self.loss_fn == "cross_entropy":
            loss = F.cross_entropy(output, target)
        elif self.loss_fn == "mse":
            loss = F.mse_loss(output, target)
        elif self.loss_fn == "binary_cross_entropy":
            loss = F.binary_cross_entropy(output, target.float())
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        self.model.zero_grad()
        loss.backward()

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
