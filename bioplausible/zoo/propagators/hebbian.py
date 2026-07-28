"""
Hebbian Learning family.

Classes: ContrastiveHebbianLearning (CHL)
"""

import torch
from torch import nn

from bioplausible.core.registry import LocalityLevel, register_propagator

from .base import LearningRuleOptimizer


@register_propagator(
    "contrastive_hebbian_learning",
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.85,
    credit_assignment_type="hebbian",
    requires_backward=False,
    requires=["transition_graph"],
    tags=["hebbian", "contrastive", "local"],
    description=(
        "Contrastive Hebbian Learning (CHL): local weight update from the"
        " difference between free and clamped Hebbian associations."
    ),
)
class ContrastiveHebbianLearning(LearningRuleOptimizer):
    """
    Contrastive Hebbian Learning (CHL).

    Updates weights based on the difference between Hebbian
    association in free vs clamped phases.

    Reference: Movellan, 1991
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        clamp_strength: float = 1.0,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.clamp_strength = clamp_strength

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("CHL requires target")

        self.model.train()

        free_states = self._forward_capture(x)
        clamped_states = self._forward_clamped(x, target)

        self._hebbian_update(free_states, clamped_states)

    def _forward_capture(self, x: torch.Tensor) -> list[torch.Tensor]:
        states = [x]
        h = x
        for layer in self._get_transitions():
            h = layer(h)
            states.append(h)
        return states

    def _forward_clamped(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> list[torch.Tensor]:
        states = [x]
        h = x
        for layer in self._get_transitions():
            h = layer(h)
            states.append(h)
        return states

    def _get_transitions(self) -> list[nn.Module]:
        if not hasattr(self.model, "transition_modules"):
            raise TypeError(
                f"CHL requires a model implementing TransitionGraph. "
                f"{type(self.model).__name__} does not implement "
                f"transition_modules(). "
                f"Either implement transition_modules() on your model, "
                f"or use a whole-model propagator (Backprop, FeedbackAlignment)."
            )
        return self.model.transition_modules()

    def _hebbian_update(
        self,
        free_states: list[torch.Tensor],
        clamped_states: list[torch.Tensor],
    ) -> None:
        layers = self._get_transitions()

        for i, layer in enumerate(layers):
            if i + 1 < len(free_states):
                pre_free = free_states[i]
                post_free = free_states[i + 1]

                pre_clamped = clamped_states[i]
                post_clamped = clamped_states[i + 1]

                delta_w = (
                    pre_clamped.T @ post_clamped - pre_free.T @ post_free
                ) / pre_free.shape[0]

                layer.weight.grad = delta_w.T

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
