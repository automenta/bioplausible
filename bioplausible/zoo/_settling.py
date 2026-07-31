"""Shared settling loop utilities for Equilibrium Propagation models.

Provides reusable helpers for the three common settling patterns:
1. Single-hidden-state settling (Family A — ``EqPropModel`` subclasses)
2. Activations-list settling (Family B — ``StandardEqProp``, ``DirectedEP``, etc.)
3. Implicit differentiation via ``EquilibriumFunction`` (autograd.Function)
"""

import logging
from collections.abc import Callable
from typing import cast

import torch
from torch import autograd, nn

logger = logging.getLogger(__name__)

_DynamicsDict = dict[str, object]


# ---------------------------------------------------------------------------
# Spectral norm freeze helper
# ---------------------------------------------------------------------------


def _run_with_sn_freeze(
    model: nn.Module,
    fn: Callable[[], None],
    steps: int,
    warmup_step: Callable[[], None],
) -> None:
    """Run ``fn`` with spectral norm frozen if needed.

    Spectral norm layers update internal ``u``/``v`` buffers in ``.train()``
    mode.  Inside an unrolled settling loop these updates cause graph breaks
    or in-place modification errors.  This helper runs one warmup step (to
    update SN statistics) then switches the model to ``.eval()`` for the
    main loop, restoring ``.train()`` on exit.

    Args:
        model: The model (checked for ``use_spectral_norm`` attribute).
        fn: Main settling loop body (called in eval mode with grad tracking
            disabled).
        steps: Total planned steps — used to decide whether to do the
            warmup step.
        warmup_step: A single forward step called *before* switching to eval
            mode (used to update SN statistics).
    """
    should_freeze = getattr(model, "use_spectral_norm", False) and model.training

    if should_freeze and steps > 0:
        warmup_step()

    if should_freeze:
        model.eval()

    try:
        fn()
    finally:
        if should_freeze:
            model.train()


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------


def _inf_norm_converged(
    h_new: torch.Tensor,
    h_old: torch.Tensor,
    step_idx: int,
    *,
    threshold_early: float = 2e-4,
    threshold_late: float = 1e-4,
    transition_step: int = 10,
    early_start: int = 5,
) -> bool:
    """Check if the inf-norm delta between consecutive states is below threshold.

    Args:
        h_new: Current state.
        h_old: Previous state.
        step_idx: Zero-based iteration index (used to decide early vs late
            threshold).

    Returns:
        True if the inf-norm delta is below the appropriate threshold.
    """
    if step_idx <= early_start:
        return False
    threshold = threshold_late if step_idx > transition_step else threshold_early
    return (
        torch.dist(h_new, h_old, p=float("inf")).item()  # type: ignore[reportUnknownMemberType]
        < threshold
    )


# ---------------------------------------------------------------------------
# Energy-based settling via gradient descent (unified primitive)
# ---------------------------------------------------------------------------


def energy_gradient_descent(
    states: list[torch.Tensor],
    energy_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    steps: int,
    *,
    lr: float = 0.15,
    momentum: float = 0.5,
    adaptive: bool = False,
    tol: float | None = None,
    patience: int = 5,
    step_size_growth: float = 1.1,
    step_size_decay: float = 0.5,
) -> list[torch.Tensor]:
    """Run gradient descent on state tensors to minimize an energy function.

    Each iteration:
      1. Computes energy via ``energy_fn(states)``.
      2. Checks for NaN/Inf divergence.
      3. Computes gradients of energy w.r.t. states via ``autograd.grad``.
      4. Applies momentum update: ``v = momentum * v + grad; state -= lr * v``.
      5. Optionally adapts LR (grow on energy decrease, decay on increase).
      6. Optionally checks early stopping via energy delta tolerance.

    Args:
        states: List of state tensors (must have ``requires_grad=True``).
        energy_fn: Callable ``f(states) -> scalar Tensor``.
        steps: Maximum number of settling iterations.
        lr: Learning rate for state updates.
        momentum: Momentum coefficient for state velocity buffers.
        adaptive: If True, use adaptive step size (grow on decrease, decay on
            increase).  Requires ``tol`` to be set.
        tol: Absolute tolerance for energy convergence.  If None, early
            stopping is disabled.
        patience: Number of consecutive steps below tolerance before
            declaring convergence.
        step_size_growth: Multiplier for LR when energy decreases.
        step_size_decay: Multiplier for LR when energy increases.

    Returns:
        Detached final state tensors.

    Raises:
        RuntimeError: If energy diverges (NaN/Inf).
    """
    momentum_buffers = [torch.zeros_like(s) for s in states]

    prev_energy: float | None = None
    patience_counter = 0
    current_lr = lr
    just_restored = False

    states_backup = [s.clone() for s in states] if adaptive else None

    for step in range(steps):
        with torch.enable_grad():
            E = energy_fn(states)

            if torch.isnan(E) or torch.isinf(E):
                raise RuntimeError(
                    f"Energy diverged at step {step}: E={E.item()}. "
                    f"Try reducing settle_lr or beta."
                )

            current_energy = float(E.item())

            # Adaptive step size: reject increase, decay LR; accept decrease, grow LR
            if adaptive and states_backup is not None:
                if prev_energy is not None:
                    if current_energy > prev_energy:
                        with torch.no_grad():
                            for s, b in zip(states, states_backup):
                                s.copy_(b)
                        current_lr *= step_size_decay
                        just_restored = True
                        continue
                    else:
                        if not just_restored:
                            current_lr = min(current_lr * step_size_growth, lr * 10)
                        with torch.no_grad():
                            for s, b in zip(states, states_backup):
                                b.copy_(s)
                else:
                    with torch.no_grad():
                        for s, b in zip(states, states_backup):
                            b.copy_(s)

            # Early stopping
            if tol is not None and prev_energy is not None and not just_restored:
                delta = abs(current_energy - prev_energy)
                rel_tol = tol * max(1.0, abs(prev_energy))
                if delta < tol or delta < rel_tol:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= patience:
                    break

            just_restored = False
            prev_energy = current_energy

            grads = torch.autograd.grad(
                E, states, retain_graph=False, allow_unused=True
            )

        # SGD with momentum
        with torch.no_grad():
            for i, (state, grad) in enumerate(zip(states, grads)):
                if grad is None:
                    continue
                momentum_buffers[i].mul_(momentum).add_(grad)
                state.sub_(momentum_buffers[i], alpha=current_lr)

    return [s.detach() for s in states]


# ---------------------------------------------------------------------------
# Family A — single-hidden-state settling (EqPropModel subclasses)
# ---------------------------------------------------------------------------


def settle_single_state(
    h_0: torch.Tensor,
    forward_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_transformed: torch.Tensor,
    steps: int,
    *,
    model: nn.Module | None = None,
    return_trajectory: bool = False,
    return_dynamics: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, _DynamicsDict | None]:
    """Run a single-hidden-state settling loop (Family A pattern).

    Handles spectral-norm freeze, early convergence detection, and optional
    trajectory/dynamics tracking.

    Args:
        h_0: Initial hidden state.
        forward_step: Callable ``f(h, x) -> h_next``.
        x_transformed: Transformed input (constant throughout the loop).
        steps: Number of iterations.
        model: Model instance (needed for spectral-norm freeze).  If None,
            the freeze logic is skipped.
        return_trajectory: If True, return the full trajectory.
        return_dynamics: If True, return per-step inf-norm deltas.

    Returns:
        ``(h_star, trajectory, dynamics)`` where:

        - ``h_star`` is the final settled state.
        - ``trajectory`` is ``None`` or a ``list[Tensor]`` of state snapshots
          (``trajectory[0]`` is ``h_0``).
        - ``dynamics`` is ``None`` or a dict with ``"deltas"`` and
          ``"final_delta"`` keys.
    """
    trajectory: list[torch.Tensor] | None = (
        cast("list[torch.Tensor]", [None] * (steps + 1)) if return_trajectory else None
    )
    if trajectory is not None:
        trajectory[0] = h_0
    deltas: list[float] | None = [] if return_dynamics else None

    h = h_0

    # Index into the trajectory buffer — incremented AFTER each snapshot.
    traj_idx = 0

    def warmup() -> None:
        nonlocal h, traj_idx, remaining
        h = forward_step(h, x_transformed)
        remaining -= 1
        if deltas is not None:
            deltas.append(torch.dist(h, h_0, p=float("inf")).item())
        if trajectory is not None:
            traj_idx += 1
            trajectory[traj_idx] = h

    remaining = steps

    def main_loop() -> None:
        nonlocal h, traj_idx, remaining

        for step_idx in range(remaining):
            h_new = forward_step(h, x_transformed)

            if deltas is not None:
                deltas.append(torch.dist(h_new, h, p=float("inf")).item())

            if _inf_norm_converged(h_new, h, step_idx):
                h = h_new
                if trajectory is not None:
                    traj_idx += 1
                    trajectory[traj_idx] = h
                break

            h = h_new
            if trajectory is not None:
                traj_idx += 1
                trajectory[traj_idx] = h

    if model is not None:
        _run_with_sn_freeze(model, main_loop, steps, warmup)
    else:
        main_loop()

    # Slice trajectory to actual length
    if trajectory is not None:
        trajectory = trajectory[: traj_idx + 1]

    dynamics: _DynamicsDict | None = None
    if deltas is not None:
        dynamics = {
            "deltas": deltas,
            "final_delta": deltas[-1] if deltas else 0.0,
        }

    return h, trajectory, dynamics


# ---------------------------------------------------------------------------
# Family B — activations-list settling (StandardEqProp, DirectedEP, etc.)
# ---------------------------------------------------------------------------


def settle_activations_list(
    activations_0: list[torch.Tensor],
    forward_dynamics: Callable[
        [list[torch.Tensor], float, torch.Tensor | None],
        list[torch.Tensor],
    ],
    steps: int,
    beta: float = 0.0,
    target: torch.Tensor | None = None,
    *,
    return_trajectory: bool = False,
    return_dynamics: bool = False,
    convergence_norm: int = 2,
    convergence_threshold: float = 1e-3,
    convergence_start: int = 5,
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]] | None,
    _DynamicsDict | None,
]:
    """Settle an activations-list model (Family B pattern).

    Iterates the bidirectional ``forward_dynamics`` function for a fixed
    number of steps, with optional early convergence detection and
    trajectory/dynamics tracking.

    Args:
        activations_0: Initial per-layer activations ``[x, h1, ..., out]``.
        forward_dynamics: ``f(activations, beta, target) -> new_activations``.
        steps: Number of settling iterations.
        beta: Nudge strength.
        target: Optional target tensor for nudged phase.
        return_trajectory: If True, record snapshot of activations per step.
        return_dynamics: If True, record per-step deltas.
        convergence_norm: p-norm for the convergence check.
        convergence_threshold: Threshold for early stopping.
        convergence_start: Step index after which convergence is checked.

    Returns:
        ``(final_activations, trajectory, dynamics)``.

        - ``trajectory`` is ``None`` or a ``list[list[Tensor]]`` where each
          entry is a CPU-detached snapshot of the activations list at a step.
        - ``dynamics`` is ``None`` or a dict with ``"deltas"`` and
          ``"final_delta"`` keys.
    """
    trajectory: list[list[torch.Tensor]] | None = [] if return_trajectory else None
    if trajectory is not None:
        trajectory.append([a.detach().cpu() for a in activations_0])

    # Convergence delta: only compute when we might use it
    need_delta = return_dynamics or convergence_start < steps
    deltas: list[float] | None = [] if return_dynamics else None
    activations = activations_0

    for step_idx in range(steps):
        prev = activations
        activations = forward_dynamics(activations, beta, target)

        if need_delta:
            delta = 0.0
            for k in range(1, len(activations)):
                delta += torch.dist(activations[k], prev[k], p=convergence_norm).item()

            if deltas is not None:
                deltas.append(delta)

            # Early stopping check (unconditional)
            if step_idx > convergence_start and delta < convergence_threshold:
                if trajectory is not None:
                    trajectory.append([a.detach().cpu() for a in activations])
                break

        if trajectory is not None:
            trajectory.append([a.detach().cpu() for a in activations])

    dynamics: _DynamicsDict | None = None
    if deltas is not None:
        dynamics = {
            "deltas": deltas,
            "final_delta": deltas[-1] if deltas else 0.0,
        }

    return activations, trajectory, dynamics


# ---------------------------------------------------------------------------
# Implicit differentiation via EquilibriumFunction
# ---------------------------------------------------------------------------


class EquilibriumFunction(autograd.Function):
    """Implicit differentiation for Equilibrium Propagation models.

    Implements O(1) memory backpropagation using the equilibrium property::

        dL/dtheta = dL/dh * dh/dtheta
        dh/dtheta = (I - J)^-1 * df/dtheta

    The backward pass solves for the adjoint state ``delta``::

        delta = (I - J ^ T) ^ -1 * dL / dh

    via fixed-point iteration::

        delta_{t+1} = J^T * delta_t + dL/dh
    """

    @staticmethod
    def forward(
        ctx: object,
        model: nn.Module,
        x_transformed: torch.Tensor,
        h_init: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.model = model

        should_freeze_sn = getattr(model, "use_spectral_norm", False) and model.training
        remaining_steps = model.max_steps

        with torch.no_grad():
            h = h_init

            if should_freeze_sn and remaining_steps > 0:
                h = model.forward_step(h, x_transformed)
                remaining_steps -= 1
                model.eval()

            try:
                for _ in range(remaining_steps):
                    h = model.forward_step(h, x_transformed)
            finally:
                if should_freeze_sn:
                    model.train()

        ctx.save_for_backward(h, x_transformed, *params)
        return h

    @staticmethod
    def backward(
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        h_star, x_transformed, *params = ctx.saved_tensors
        model = ctx.model

        was_training = model.training
        model.eval()

        try:
            delta = grad_output
            x_transformed_detached = x_transformed.detach()

            forward_fn = getattr(model, "_forward_step_impl", model.forward_step)

            delta_prev = None
            for _ in range(model.max_steps):
                with torch.enable_grad():
                    h_star_loop = h_star.detach().requires_grad_(True)
                    f_h = forward_fn(h_star_loop, x_transformed_detached)

                    vjp = autograd.grad(
                        f_h,
                        h_star_loop,
                        grad_outputs=delta.detach(),
                        retain_graph=False,
                        create_graph=False,
                    )[0]

                    delta_next = (vjp + grad_output).detach()

                # Early convergence check for adjoint iteration
                if delta_prev is not None and _ > 3:
                    with torch.no_grad():
                        if (
                            torch.dist(delta_next, delta_prev, p=float("inf")).item()
                            < 1e-4
                        ):
                            delta = delta_next
                            break
                delta_prev = delta_next
                delta = delta_next

            # Compute parameter gradients
            delta = delta.detach()

            with torch.enable_grad():
                h_star_detached = h_star.detach()
                x_detached = x_transformed.detach()

                params_with_grad = [p for p in params if p.requires_grad]
                grads_params_list: list[torch.Tensor | None] = [None] * len(params)

                if params_with_grad:
                    f_h_params = forward_fn(h_star_detached, x_detached)
                    computed_grads = autograd.grad(
                        f_h_params,
                        params,
                        grad_outputs=delta,
                        allow_unused=True,
                        retain_graph=False,
                    )
                    grads_params_list = list(computed_grads)

                # Input gradient
                grad_x = None
                if x_transformed.requires_grad:
                    f_h_x = model.forward_step(h_star_detached, x_transformed)
                    grad_x = autograd.grad(
                        f_h_x,
                        x_transformed,
                        grad_outputs=delta,
                        retain_graph=False,
                    )[0]
        finally:
            model.train(was_training)

        return (None, grad_x, None, *grads_params_list)


__all__ = [
    "EquilibriumFunction",
    "_run_with_sn_freeze",
    "energy_gradient_descent",
    "settle_activations_list",
    "settle_single_state",
]
