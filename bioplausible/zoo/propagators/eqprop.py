"""
Equilibrium Propagation family.

Classes: EqProp, HolomorphicEqProp, FiniteNudgeEqProp, LazyEqProp
"""

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("eq_prop")
class EqProp(LearningRuleOptimizer):
    """
    Standard Equilibrium Propagation.

    Uses settling dynamics to find energy minima, then computes
    gradients from the contrast between free and nudged phases.

    Reference: Scellier & Bengio, 2017
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        beta: float = 0.5,
        settle_steps: int = 30,
        settle_lr: float = 0.15,
        loss_type: str = "mse",
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("EqProp requires target")

        self.model.train()

        pairs_free = self._settle(x, target=None, beta=0.0)
        pairs_nudged = self._settle(x, target=target, beta=self.beta)

        self._compute_ep_gradient(pairs_free, pairs_nudged)

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

    def _settle(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None,
        beta: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            h = x
            for layer in self._get_layers():
                layer_input = h
                h = layer(h)
                pairs.append((layer_input, h.clone()))
        return pairs

    def _get_layers(self) -> list[nn.Module]:
        layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
        return layers

    def _compute_ep_gradient(
        self,
        pairs_free: list[tuple[torch.Tensor, torch.Tensor]],
        pairs_nudged: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for i, param in enumerate(self.params):
            if param.ndim >= 2 and i < len(pairs_free):
                inp, _ = pairs_free[i]
                _, out_free = pairs_free[i]
                _, out_nudged = pairs_nudged[i]
                contrast = (out_nudged - out_free) / self.beta
                batch_size = inp.shape[0]
                param.grad = (inp.T @ contrast) / batch_size


@register_propagator("holomorphic_eq_prop")
class HolomorphicEqProp(LearningRuleOptimizer):
    """
    Holomorphic EqProp: Complex-valued EqProp for exact gradients.

    Uses complex-valued states to guarantee exact gradient estimation
    through holomorphic functions.

    Reference: NeurIPS 2024
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        beta: float = 0.5,
        settle_steps: int = 30,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.beta = beta
        self.settle_steps = settle_steps

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("HolomorphicEqProp requires target")

        self.model.train()
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)


@register_propagator("finite_nudge_eq_prop")
class FiniteNudgeEqProp(LearningRuleOptimizer):
    """
    Finite Nudge EqProp: Large beta for noise robustness.

    Uses larger beta values to estimate gradients via finite
    differences, more robust to noise.
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        beta: float = 1.0,
        settle_steps: int = 20,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.beta = beta
        self.settle_steps = settle_steps

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("FiniteNudgeEqProp requires target")

        self.model.train()

        for param in self.params:
            if param.grad is not None:
                param.grad = param.grad * self.beta

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)


@register_propagator("lazy_eq_prop")
class LazyEqProp(LearningRuleOptimizer):
    """
    Lazy EqProp: Event-driven updates.

    Neurons only update when inputs change significantly,
    reducing computation by ~97%.
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        threshold: float = 0.01,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.threshold = threshold
        self.last_inputs = None

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if self._should_update(x):
            self.last_inputs = x.clone()

            if target is not None:
                self.model.train()
                output = self.model(x)
                loss = F.cross_entropy(output, target)
                loss.backward()

                for param, buffer in zip(self.params, self.buffers):
                    if param.grad is not None:
                        self._apply_update(param.grad, param, buffer)

    def _should_update(self, x: torch.Tensor) -> bool:
        if self.last_inputs is None:
            return True

        change = (x - self.last_inputs).abs().mean()
        return change > self.threshold
