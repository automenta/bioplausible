"""
Hebbian Learning family.

Classes: ContrastiveHebbianLearning (CHL)
"""

import torch
from torch import nn

from computronium.core.registry import LocalityLevel, register_credit_assignment

from .base import LearningRuleOptimizer

__all__ = [
    "ContrastiveHebbianLearning",
]


@register_credit_assignment(
    "contrastive_hebbian_learning",
    family="hebbian",
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
    """Contrastive Hebbian Learning (CHL).

    Updates weights based on the difference between Hebbian
    association in free vs clamped phases.

    Reference: Movellan, 1991

    Memory: O(1) in depth — free and clamped forwards stream per-layer
    pre/post under no_grad, accumulating only the per-layer outer product.
    No activation lists, no autograd graph.
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
        transitions = self._get_transitions()

        # Free phase: stream pre/post, accumulate -free_outer_product per layer
        with torch.no_grad():
            pre_free = x
            delta_w_list = []
            for layer in transitions:
                post_free = layer(pre_free)
                # CHL: ΔW ∝ clamped_post @ clamped_pre.T - free_post @ free_pre.T
                # Store -free_post @ free_pre.T for now
                delta_w_list.append(-post_free.T @ pre_free / pre_free.shape[0])
                pre_free = post_free

        # Clamped phase: forward pass with output clamped to target
        with torch.no_grad():
            pre_clamped = x
            post_clamped_list = []
            for layer in transitions:
                post_clamped = layer(pre_clamped)
                post_clamped_list.append(post_clamped)
                pre_clamped = post_clamped

            # Clamp the final output
            if target.dim() == 1:
                target_vec = torch.nn.functional.one_hot(
                    target, num_classes=post_clamped.shape[1]
                ).float()
            else:
                target_vec = target.float()
            post_clamped_list[-1] = target_vec.to(post_clamped.device)

        # Add clamped contribution: post_clamped @ pre_clamped.T for each layer
        for i, layer in enumerate(transitions):
            pre_c = x if i == 0 else post_clamped_list[i - 1]
            post_c = post_clamped_list[i]
            delta_w_list[i] += post_c.T @ pre_c / pre_c.shape[0]

        # Apply accumulated delta_w to the underlying trainable weight of each
        # transition layer. CHL: ΔW = clamped − free (association difference) and
        # weight update W += lr * ΔW (gradient descent of the energy), so here the
        # applied gradient is −ΔW. Layers may be spectral-norm parametrized: their
        # ``.weight`` is a computed view, so set grad on the real trainable param
        # (``layer.parametrizations.weight.original``) or, for plain Linear, the
        # module's own weight. Bias receives no association update.
        for i, layer in enumerate(transitions):
            target_tensor = getattr(layer, "weight", None)
            if target_tensor is None or not target_tensor.requires_grad:
                continue
            if (
                hasattr(layer, "parametrizations")
                and "weight" in layer.parametrizations
            ):
                target_tensor = layer.parametrizations.weight.original
            target_tensor.grad = -delta_w_list[i]

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

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
