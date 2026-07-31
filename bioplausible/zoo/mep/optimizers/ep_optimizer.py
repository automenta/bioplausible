"""
LEGACY REFERENCE — DO NOT USE IN PRODUCTION.

This file exists ONLY to support the gradient-parity test
``tests/integration/test_ep_gradient_parity.py`` which characterizes
EPOptimizer's buggy ``(E_nudged - E_free) / beta`` formula and compares
it against ``EqProp``'s correct contrastive EP gradient.

**Why this is legacy** (Session 15-16 findings):

- ``EPOptimizer`` has **zero production consumers** — every constructor
  call appears in this file's own docstring examples.  Presets use
  ``CompositeOptimizer`` + strategies, not ``EPOptimizer``.
- The gradient formula ``d/dW[(E_nudged - E_free) / beta]`` produces
  backprop-like last-layer gradients, NOT true EP contrastive gradients
  (see ``test_ep_gradient_parity.py`` module docstring for details).
- ``EWCState`` is also dead — never instantiated outside this file.
  ``EPOptimizerWithEWC`` in ``zoo/mep/optimizers/ewc.py`` is a separate
  implementation that wraps ``O1MemoryEPv2`` (not ``EPOptimizer``).

**Do not add features, fix bugs, or extend this code.**
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "EPConfig",
    "EPOptimizer",
]


@dataclass
class EPConfig:
    """Configuration for legacy EPOptimizer (test reference only)."""

    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    mode: str = "ep"
    beta: float = 0.5
    settle_steps: int = 10
    settle_lr: float = 0.2
    gradient_method: str = "autograd"
    ewc_lambda: float = 0.0
    ns_steps: int = 5
    gamma: float = 0.95
    loss_type: str = "cross_entropy"
    softmax_temperature: float = 1.0


class EPOptimizer:
    """
    Legacy EP optimizer — TEST REFERENCE ONLY.

    Implements the buggy ``(E_nudged - E_free) / beta`` gradient formula.
    See module docstring for why this is not for production use.
    """

    def __init__(
        self,
        params,
        model: nn.Module | None = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        mode: str = "ep",
        beta: float = 0.5,
        settle_steps: int = 10,
        settle_lr: float = 0.2,
        gradient_method: str = "analytic",
        ewc_lambda: float = 0.0,
        ns_steps: int = 5,
        gamma: float = 0.95,
        loss_type: str = "cross_entropy",
        softmax_temperature: float = 1.0,
    ):
        # Force autograd — analytic is broken (original design comment)
        self.config = EPConfig(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            mode=mode,
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            gradient_method="autograd",
            ewc_lambda=ewc_lambda,
            ns_steps=ns_steps,
            gamma=gamma,
            loss_type=loss_type,
            softmax_temperature=softmax_temperature,
        )
        self.model = model
        self.params = list(params)
        self.structure = self._build_structure_from_model(model) if model else []

    @staticmethod
    def _build_structure_from_model(
        model: nn.Module,
    ) -> list[dict[str, object]]:
        if hasattr(model, "transition_modules"):
            try:
                return [
                    {"type": "layer", "module": m} for m in model.transition_modules()
                ]
            except NotImplementedError:
                pass
        return [
            {"type": "layer", "module": m}
            for m in model.modules()
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        ]

    def _settle(
        self,
        x: torch.Tensor,
        target_vec: torch.Tensor | None,
        original_target: torch.Tensor | None,
        beta: float = 0.0,
    ) -> list[torch.Tensor]:
        """Settling loop (autograd gradients only)."""
        states = self._capture_states(x)
        momentum_buffers = [torch.zeros_like(s) for s in states]

        for _ in range(self.config.settle_steps):
            grads = self._autograd_gradients(x, states, target_vec, beta)
            with torch.no_grad():
                for i, (state, buf, g) in enumerate(
                    zip(states, momentum_buffers, grads)
                ):
                    buf.mul_(0.5).add_(g)
                    state.sub_(buf, alpha=self.config.settle_lr)

        return [s.detach() for s in states]

    def _capture_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        states: list[torch.Tensor] = []
        handles: list[object] = []

        def hook(m, i, o):
            states.append(o.detach().float().clone().requires_grad_(True))

        for item in self.structure:
            if item["type"] in ("layer", "attention"):
                handles.append(item["module"].register_forward_hook(hook))

        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for h in handles:
                h.remove()

        return states

    def _autograd_gradients(
        self,
        x: torch.Tensor,
        states: list[torch.Tensor],
        target_vec: torch.Tensor | None,
        beta: float,
    ) -> list[torch.Tensor | None]:
        states_clone = [s.detach().clone().requires_grad_(True) for s in states]
        E = self._energy_from_states(x, states_clone, target_vec, beta, use_grad=True)
        return list(
            torch.autograd.grad(E, states_clone, retain_graph=False, allow_unused=True)
        )

    def _energy_from_states(
        self,
        x: torch.Tensor,
        states: list[torch.Tensor],
        target_vec: torch.Tensor | None,
        beta: float,
        use_grad: bool = False,
    ) -> torch.Tensor:
        device = x.device
        batch_size = x.shape[0]
        E = torch.tensor(0.0, device=device, dtype=torch.float32)
        prev = x
        state_idx = 0

        context = torch.enable_grad if use_grad else torch.no_grad
        with context():
            for item in self.structure:
                if item["type"] == "layer":
                    if state_idx >= len(states):
                        break
                    state = states[state_idx]
                    h = item["module"](prev)
                    E = (
                        E
                        + 0.5
                        * F.mse_loss(h.float(), state.float(), reduction="sum")
                        / batch_size
                    )
                    prev = state.to(x.dtype)
                    state_idx += 1
                elif item["type"] == "act":
                    prev = item["module"](prev)

            if target_vec is not None and beta > 0:
                if self.config.loss_type == "cross_entropy":
                    E = (
                        E
                        + beta
                        * F.cross_entropy(
                            prev.float(),
                            target_vec,
                            reduction="sum",
                            label_smoothing=0.1,
                        )
                        / batch_size
                    )
                else:
                    if target_vec.dim() == 1:
                        target_one_hot = F.one_hot(
                            target_vec, num_classes=prev.shape[1]
                        ).float()
                    elif target_vec.shape != prev.shape:
                        target_one_hot = target_vec.expand_as(prev)
                    else:
                        target_one_hot = target_vec
                    E = (
                        E
                        + beta
                        * F.mse_loss(prev.float(), target_one_hot, reduction="sum")
                        / batch_size
                    )

        return E
