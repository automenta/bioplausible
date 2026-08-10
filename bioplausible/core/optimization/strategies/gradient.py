"""Generic gradient computation strategies (REFACTOR.md §7).

Backpropagation and feedback-alignment gradients. MEP-specific EP
strategies live in ``zoo.mep.optimizers.strategies.gradient``.
"""

import torch
from torch import nn

from .base import GradientStrategy

__all__ = ["BackpropGradient", "FAGradient"]


class BackpropGradient(GradientStrategy):
    """Standard backpropagation via ``loss.backward()``."""

    def __init__(self, loss_fn: nn.Module | None = None):
        self.loss_fn = loss_fn

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: nn.Module | None = None,
        **kwargs: object,
    ) -> None:
        """Compute gradients via backpropagation."""
        fn = loss_fn or self.loss_fn
        if fn is None:
            raise ValueError("loss_fn must be provided to BackpropGradient")
        output = model(x)
        loss = fn(output, target)
        loss.backward()


class FAGradient(GradientStrategy):
    """Feedback alignment: errors routed through fixed random matrices.

    The loss error propagates backward through frozen random ``B_l`` matrices
    (one per linear layer) instead of the transposed forward weights,
    avoiding symmetric weight transport while keeping a global loss signal.
    """

    def __init__(self, loss_fn: nn.Module | None = None, feedback_seed: int = 0):
        self.loss_fn = loss_fn
        self.feedback_seed = feedback_seed
        self._feedback: dict[int, torch.Tensor] = {}

    def _feedback_for(self, module: nn.Linear) -> torch.Tensor:
        """Return (and lazily build) the fixed random feedback for a layer.

        The feedback `B_l` has shape ``(out, in)`` so a size-``out`` error
        maps back to the layer input dimension: ``delta @ B_l``.
        """
        weight = module.weight
        key = id(module)
        fb = self._feedback.get(key)
        if fb is None:
            g = torch.Generator(device=weight.device).manual_seed(self.feedback_seed)
            fb = torch.randn(
                weight.shape[0], weight.shape[1], device=weight.device, generator=g
            )
            self._feedback[key] = fb
        return fb

    def _linear_modules(self, model: nn.Module) -> list[nn.Linear]:
        return [m for m in model.modules() if isinstance(m, nn.Linear)]

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: nn.Module | None = None,
        **kwargs: object,
    ) -> None:
        """Compute feedback-alignment gradients for a sequential linear model.

        The model must be a feed-forward ``nn.Sequential`` of ``Linear`` and
        activation layers; each layer's input is recorded during the forward
        pass so ``grad_W_l = delta_l @ a_{l-1}^T``.
        """
        fn = loss_fn or self.loss_fn
        if fn is None:
            raise ValueError("loss_fn must be provided to FAGradient")
        if target is None:
            raise ValueError("target must be provided to FAGradient")

        layers = self._linear_modules(model)
        if not layers:
            raise ValueError("FAGradient requires at least one Linear layer")

        inputs: list[torch.Tensor] = []
        current = x
        with torch.enable_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    inputs.append(current)
                    current = module(current)

            loss = fn(current, target)
            error = torch.autograd.grad(loss, current, retain_graph=True)[0]

        # Backward through fixed random feedback matrices.
        delta = error.detach()
        for layer, layer_input in reversed(list(zip(layers, inputs))):
            weight = layer.weight
            weight.grad = delta.T @ layer_input.detach()
            if layer.bias is not None:
                layer.bias.grad = delta.sum(0)
            delta = delta @ self._feedback_for(layer)

        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad:
                param.grad = torch.zeros_like(param)
