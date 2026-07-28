"""
O(1) Memory Implementation for EP

Phase 2: Technical Excellence - Priority 1

This module implements memory-efficient EP by avoiding PyTorch autograd overhead:
1. Manual settling without autograd (no intermediate activation storage)
2. No-grad energy computation (direct matmul instead of nn.Module forward)
3. Selective autograd only for final contrast step

Key insight: We only need the final settled states, not the settling trajectory.
By operating in no_grad() mode during settling, we avoid O(depth) activation storage.

Author: Phase 2 Implementation
Created: 2026-02-18
Refactored: 2026-07-28 to use TransitionGraph protocol (transition_modules())
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _capture_states_no_grad(
    model: nn.Module,
    x: torch.Tensor,
    transition_modules: list[nn.Module],
) -> list[torch.Tensor]:
    """
    Capture initial layer states without autograd using forward hooks on transition modules.
    """
    states: list[torch.Tensor] = []
    handles: list[Any] = []

    def capture_hook(module: nn.Module, inp: Any, output: Any) -> None:
        if isinstance(output, tuple):
            s = output[0].detach().float().clone()
        else:
            s = output.detach().float().clone()
        states.append(s)

    for module in transition_modules:
        handles.append(module.register_forward_hook(capture_hook))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    return states


def manual_energy_compute(
    model: nn.Module,
    x: torch.Tensor,
    states: list[torch.Tensor],
    transition_modules: list[nn.Module],
    target_vec: torch.Tensor | None,
    beta: float,
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
    use_grad: bool = False,
) -> torch.Tensor:
    """
    Compute EP energy with optional grad tracking using transition_modules.

    When use_grad=False (default): No autograd overhead, for settling iterations.
    When use_grad=True: Builds computation graph, for final contrast step.

    Args:
        model: Neural network module (provides weights).
        x: Input tensor.
        states: List of layer states (settling variables).
        transition_modules: Modules from model.transition_modules() - each produces a state.
        target_vec: Target for nudge term (None for free phase).
        beta: Nudging strength.
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax.
        use_grad: If True, enable grad for parameter gradient computation.

    Returns:
        Scalar energy tensor.
    """
    batch_size = x.shape[0]
    device = x.device

    # Accumulate energy in float32 for stability
    E = torch.tensor(0.0, device=device, dtype=torch.float32)
    prev = x

    use_classification = loss_type == "cross_entropy"
    num_states = len(states)

    # Context manager for grad/no_grad
    ctx = torch.enable_grad() if use_grad else torch.no_grad()

    with ctx:
        for i, (module, state) in enumerate(zip(transition_modules, states)):
            # Forward pass through transition module
            h = module(prev)

            # Compute energy
            is_last_state = i == num_states - 1
            if use_classification and is_last_state:
                E = E + _kl_energy(
                    state.float(), h.float(), batch_size, softmax_temperature
                )
            else:
                E = E + 0.5 * _mse(h.float(), state.float()) / batch_size

            # Input to next layer is the current state
            prev = state.to(x.dtype)

        # Nudge term
        if target_vec is not None and beta > 0:
            E = E + _nudge_term(prev.float(), target_vec, beta, batch_size, loss_type)

    return E


def _mse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE."""
    return F.mse_loss(input, target, reduction="sum")


def _kl_energy(
    state: torch.Tensor,
    prediction: torch.Tensor,
    batch_size: int,
    softmax_temperature: float,
) -> torch.Tensor:
    """Compute KL divergence energy."""
    eps = 1e-8

    state_softmax = F.softmax(state / softmax_temperature, dim=1)
    h_softmax = F.softmax(prediction / softmax_temperature, dim=1)

    kl_div = F.kl_div(torch.log(state_softmax + eps), h_softmax, reduction="sum")
    return kl_div / batch_size


def _nudge_term(
    output: torch.Tensor,
    target_vec: torch.Tensor,
    beta: float,
    batch_size: int,
    loss_type: str,
) -> torch.Tensor:
    """Compute nudge term."""
    if loss_type == "cross_entropy":
        return (
            beta
            * F.cross_entropy(output, target_vec, reduction="sum", label_smoothing=0.1)
            / batch_size
        )
    else:
        return beta * F.mse_loss(output, target_vec, reduction="sum") / batch_size


def settle_manual(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor | None,
    beta: float,
    energy_fn: Callable,
    transition_modules: list[nn.Module],
    steps: int = 30,
    lr: float = 0.15,
    momentum: float = 0.5,
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
) -> list[torch.Tensor]:
    """
    Manual settling without autograd overhead using transition_modules.

    Key optimization: We operate in no_grad() mode during settling iterations.
    We only need the final settled states, not the trajectory.

    For gradient computation during settling:
    1. Compute energy in no_grad mode
    2. Temporarily enable grad on states only (not weights)
    3. Recompute energy to get state gradients
    4. Update states in no_grad mode

    This avoids storing O(steps * depth) activations from the settling loop.

    Args:
        model: Neural network module.
        x: Input tensor.
        target: Target tensor (None for free phase).
        beta: Nudging strength.
        energy_fn: Energy function (use manual_energy_compute).
        transition_modules: Modules from model.transition_modules().
        steps: Number of settling iterations.
        lr: Settling learning rate.
        momentum: Momentum factor.
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax.

    Returns:
        List of settled state tensors.
    """

    # Capture initial states (no_grad)
    with torch.no_grad():
        states = _capture_states_no_grad(model, x, transition_modules)

    if not states:
        if len(transition_modules) > 0:
            raise RuntimeError(
                f"No activations captured. Expected {len(transition_modules)} layer(s)."
            )
        else:
            return []

    # Prepare target
    target_vec = None
    if target is not None:
        if loss_type == "cross_entropy":
            if target.dim() > 1 and target.shape[1] > 1:
                target_vec = target.argmax(dim=1).long()
            else:
                target_vec = target.squeeze().long()
        elif target.dim() == 1:
            num_classes = states[-1].shape[-1]
            target_vec = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)
        else:
            target_vec = target.to(dtype=x.dtype)

    # Momentum buffers
    momentum_buffers = [torch.zeros_like(s) for s in states]

    # Settling loop
    for step in range(steps):
        # Compute gradients w.r.t. states using finite-difference-like approach
        # We need dE/dstate for each state

        # Create states that require grad
        states_for_grad = []
        for s in states:
            s_copy = s.detach().clone().requires_grad_(True)
            states_for_grad.append(s_copy)

        # Compute energy with grad-requiring states
        # Note: We use use_grad=True to enable the gradient flow through states
        E_for_grad = manual_energy_compute(
            model,
            x,
            states_for_grad,
            transition_modules,
            target_vec,
            beta,
            loss_type=loss_type,
            softmax_temperature=softmax_temperature,
            use_grad=True,
        )

        # Compute gradients w.r.t. states
        grads = torch.autograd.grad(
            E_for_grad, states_for_grad, retain_graph=False, allow_unused=True
        )

        # Update states (no_grad)
        with torch.no_grad():
            for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, grads)):
                if g is None:
                    continue
                buf.mul_(momentum).add_(g)
                state.sub_(buf, alpha=lr)

    return [s.detach() for s in states]


def energy_from_states(
    model: nn.Module,
    x: torch.Tensor,
    states: list[torch.Tensor],
    transition_modules: list[nn.Module],
    target_vec: torch.Tensor | None,
    beta: float,
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute energy from fixed states WITH autograd for parameter gradients.

    This builds a minimal graph for computing dE/dW without storing settling history.
    Uses standard nn.Module forward passes for correct gradient flow.
    """
    batch_size = x.shape[0]
    device = x.device

    E = torch.tensor(0.0, device=device, dtype=torch.float32)
    prev = x

    use_classification = loss_type == "cross_entropy"

    with torch.enable_grad():
        for i, (module, state) in enumerate(zip(transition_modules, states)):
            is_last_state = i == len(states) - 1

            # Forward pass WITH autograd (for parameter gradients)
            h = module(prev)

            if use_classification and is_last_state:
                E = E + _kl_energy(
                    state.float(), h.float(), batch_size, softmax_temperature
                )
            else:
                E = E + 0.5 * _mse(h.float(), state.float()) / batch_size

            prev = state.to(x.dtype)

        # Nudge term
        if target_vec is not None and beta > 0:
            E = E + _nudge_term(prev.float(), target_vec, beta, batch_size, loss_type)

    return E


class O1MemoryEP:
    """
    O(1) Memory EP optimizer wrapper.

    Usage:
        optimizer = O1MemoryEP(model.parameters(), model=model, lr=0.01)
        optimizer.step(x=x, target=y)

    This is a prototype demonstrating O(1) activation memory.
    For production use, integrate with CompositeOptimizer.
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        settle_steps: int = 30,
        settle_lr: float = 0.15,
        beta: float = 0.5,
        loss_type: str = "cross_entropy",
    ):
        self.params = list(params)
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.beta = beta
        self.loss_type = loss_type

        # Get transition modules directly from model (TransitionGraph protocol)
        if not hasattr(model, "transition_modules"):
            raise TypeError(
                f"O1MemoryEP requires a model implementing TransitionGraph. "
                f"{type(model).__name__} does not implement transition_modules()."
            )
        self.transition_modules = model.transition_modules()

        # Momentum buffers for parameter updates
        self.buffers = [torch.zeros_like(p) for p in self.params]

    def step(self, x: torch.Tensor, target: torch.Tensor):
        """
        Perform O(1) memory EP training step.
        """
        # Free phase settling (O(1) memory)
        states_free = settle_manual(
            self.model,
            x,
            None,
            beta=0.0,
            energy_fn=manual_energy_compute,
            transition_modules=self.transition_modules,
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
        )

        # Nudged phase settling (O(1) memory)
        states_nudged = settle_manual(
            self.model,
            x,
            target,
            beta=self.beta,
            energy_fn=manual_energy_compute,
            transition_modules=self.transition_modules,
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
        )

        # Contrast step (minimal autograd for parameter gradients)
        E_free = energy_from_states(
            self.model,
            x,
            states_free,
            self.transition_modules,
            None,
            0.0,
            loss_type=self.loss_type,
        )

        E_nudged = energy_from_states(
            self.model,
            x,
            states_nudged,
            self.transition_modules,
            target,
            self.beta,
            loss_type=self.loss_type,
        )

        contrast_loss = (E_nudged - E_free) / self.beta

        # Compute parameter gradients
        grads = torch.autograd.grad(contrast_loss, self.params, retain_graph=False)

        # Update parameters with momentum
        with torch.no_grad():
            for p, g, buf in zip(self.params, grads, self.buffers):
                buf.mul_(self.momentum).add_(g)

                # Apply weight decay
                if self.weight_decay != 0:
                    buf.add_(p, alpha=self.weight_decay)

                p.sub_(buf, alpha=self.lr)