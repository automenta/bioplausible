"""Shared settling loop utilities for Equilibrium Propagation models.

Provides reusable helpers for the three common settling patterns:
1. Single-hidden-state settling (Family A — ``EqPropModel`` subclasses)
2. Activations-list settling (Family B — ``StandardEqProp``, ``DirectedEP``, etc.)
3. Implicit differentiation via ``EquilibriumFunction`` (autograd.Function)

Phase 3 (REFACTOR7): Unified SettleProtocol + settle_universal primitive
for cross-algorithm convergence instrumentation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

import torch
from torch import autograd, nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from computronium.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

logger = get_logger()

_DynamicsDict = dict[str, object]


# ============================================================
# Phase 3: Unified SettleProtocol for all settling families
# ============================================================


@dataclass(frozen=True, slots=True)
class SettleConfig:
    """Unified settling configuration (sweepable hyperparameters)."""

    max_steps: int = 30
    convergence_threshold: float = 1e-4
    convergence_start: int = 5
    convergence_norm: int = 2
    convergence_relative: bool = True


@dataclass(frozen=True, slots=True)
class SettleTelemetry:
    """Unified settling telemetry surface (JSON-serializable)."""

    algorithm: str
    family: Literal["A", "B", "energy", "o1memory", "ep"]
    steps_taken: int
    max_steps: int
    converged: bool
    final_delta: float
    deltas: list[float]
    settle_time_ms: float
    memory_mb: float
    hardware: str
    backend: str


@runtime_checkable
class SettleProtocol(Protocol):
    """Unified settling / compute telemetry surface for all bio-plausible algorithms.

    Extends EquilibriumSettleProtocol to cover:
    - Family A: single-hidden-state (EqProp, LoopedMLP)
    - Family B: activations-list (StandardEqProp, DirectedEP)
    - Energy-based: PCN, Tile, Hopfield
    - O(1) Memory: O1MemoryEPv2 analytic
    - EP: MEP EP settling

    Implementations provide algorithm-specific dynamics; the shared
    ``settle_universal`` primitive handles iteration, convergence, checkpointing,
    and telemetry.
    """

    # Config knobs (sweepable)
    convergence_threshold: float
    convergence_start: int
    max_steps: int

    # Core dynamics (algorithm-specific signature)
    def _initialize_state(
        self, x: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]: ...
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor: ...
    def _step(
        self, state: torch.Tensor | list[torch.Tensor], x_transformed: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]: ...

    # Optional: algorithm-specific convergence check
    def _check_converged(
        self,
        state_new: torch.Tensor | list[torch.Tensor],
        state_old: torch.Tensor | list[torch.Tensor],
        step: int,
    ) -> bool: ...

    # Telemetry hooks (called by shared primitive)
    def _on_step_end(
        self, step: int, state: torch.Tensor | list[torch.Tensor], delta: float
    ): ...
    def _on_converged(self, step: int, final_delta: float): ...
    def _on_max_steps(self, step: int, final_delta: float): ...


@runtime_checkable
class EquilibriumSettleProtocol(Protocol):
    """Structural contract for a single-hidden-state equilibrium rule (Family A).

    A model adopting this protocol routes its settling loop through the shared
    :func:`settle_state` primitive (P1), inheriting early convergence stopping
    instead of a hand-rolled fixed-iteration loop. The members are the knobs and
    dynamics an equilibrium model must expose:

    - ``convergence_threshold`` / ``convergence_start``: the early-stop gate.
    - ``max_steps``: the hard iteration ceiling.
    - ``_initialize_hidden_state(x)``: the zero (or other) starting hidden state.
    - ``_transform_input(x)``: project the raw input once, before settling.
    - ``_forward_step_impl(h, x_transform)``: one recurrent step.

    ``LoopedMLP`` and its substrate facades already satisfy this surface;
    ``NeuralCube`` adopts it in P1.
    """

    convergence_threshold: float
    convergence_start: int
    max_steps: int

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor: ...
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor: ...
    def _forward_step_impl(
        self, h: torch.Tensor, x_transform: torch.Tensor
    ) -> torch.Tensor: ...


def _inf_norm_delta(h_new: torch.Tensor, h_old: torch.Tensor) -> float:
    """Inf-norm distance between consecutive settle states (moved to CPU)."""
    return torch.dist(h_new, h_old, p=float("inf")).item()  # type: ignore[reportUnknownMemberType]


def settle_state(
    model: EquilibriumSettleProtocol,
    x: torch.Tensor,
    *,
    steps: int | None = None,
    return_trajectory: bool = False,
) -> tuple[torch.Tensor, int, bool]:
    """Settle a single-hidden-state equilibrium rule to a fixed point.

    P1's shared primitive: runs the model's recurrence with early convergence
    detection and gradient checkpointing, and reports how many steps it actually
    took and whether it converged — turning the §7 early-stop win into a
    framework property available to **any** protocol-adopting rule, not just
    ``eqprop``.

    The convergence gate honours the model's *own* ``convergence_threshold`` /
    ``convergence_start`` (the knobs a search space can now legitimately sweep
    after P0a/P1), rather than a hard-coded epsilon.

    Args:
        model: A model satisfying :class:`EquilibriumSettleProtocol`.
        x: Raw input batch.
        steps: Override for the settle ceiling (defaults to ``model.max_steps``).
        return_trajectory: Reserved for callers that need per-step snapshots; the
            settled ``h`` is returned either way.

    Returns:
        ``(h_final, steps_taken, converged)``:

        - ``h_final`` is the settled hidden state (grad-tracked under autograd).
        - ``steps_taken`` is the number of recurrence steps executed.
        - ``converged`` is True if the inf-norm delta fell below
          ``convergence_threshold`` after ``convergence_start`` steps.
    """
    del return_trajectory  # trajectory snapshots are handled by the caller if needed
    max_steps = steps if steps is not None else model.max_steps
    threshold = float(model.convergence_threshold)
    start = int(model.convergence_start)
    x_transform = model._transform_input(x)
    h = model._initialize_hidden_state(x)

    def _step(state: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return _checkpoint(
                model._forward_step_impl, state, x_transform, use_reentrant=False
            )
        return model._forward_step_impl(state, x_transform)

    converged = False
    steps_taken = 0
    for step_idx in range(max_steps):
        h_new = _step(h)
        steps_taken += 1
        if step_idx > start and _inf_norm_delta(h_new, h) < threshold:
            h = h_new
            converged = True
            break
        h = h_new

    return h, steps_taken, converged


def settle_universal(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
    model: SettleProtocol,
    x: torch.Tensor,
    *,
    config: SettleConfig | None = None,
    algorithm: str = "unknown",
    family: Literal["A", "B", "energy", "o1memory", "ep"] = "A",
    hardware: str = "cpu",
    backend: str = "pytorch",
    return_trajectory: bool = False,
) -> tuple[torch.Tensor | list[torch.Tensor], int, bool, SettleTelemetry]:
    """Universal settling primitive for all bio-plausible algorithms.

    Runs the model's recurrence with unified early convergence detection,
    gradient checkpointing, and telemetry collection. Works with any model
    satisfying :class:`SettleProtocol`.

    Args:
        model: A model satisfying :class:`SettleProtocol`.
        x: Raw input batch.
        config: Optional :class:`SettleConfig` (uses model's knobs if None).
        algorithm: Algorithm name for telemetry.
        family: Settling family for telemetry.
        hardware: Hardware target for telemetry.
        backend: Backend name for telemetry.
        return_trajectory: If True, return per-step state snapshots.

    Returns:
        ``(state_final, steps_taken, converged, telemetry)``:
        - ``state_final`` is the settled state(s).
        - ``steps_taken`` is the number of iterations executed.
        - ``converged`` is True if early-stop triggered.
        - ``telemetry`` is a :class:`SettleTelemetry` with full convergence profile.
    """
    import time

    config = config or SettleConfig(
        max_steps=model.max_steps,
        convergence_threshold=model.convergence_threshold,
        convergence_start=model.convergence_start,
    )

    settle_start = time.monotonic()
    max_steps = config.max_steps
    threshold = config.convergence_threshold
    start = config.convergence_start

    x_transformed = model._transform_input(x)
    state = model._initialize_state(x)

    deltas: list[float] = []
    trajectory: list[object] | None = [] if return_trajectory else None

    def _step_fn(s):
        if torch.is_grad_enabled():
            return _checkpoint(model._step, s, x_transformed, use_reentrant=False)
        return model._step(s, x_transformed)

    converged = False
    steps_taken = 0
    for step_idx in range(max_steps):
        state_new = _step_fn(state)
        steps_taken += 1

        # Compute convergence delta
        if isinstance(state_new, list) and isinstance(state, list):
            # Multi-state: max relative delta across layers
            max_delta = 0.0
            for s_new, s_old in zip(state_new, state):
                abs_delta = torch.dist(s_new, s_old, p=config.convergence_norm).item()
                if config.convergence_relative:
                    norm = s_old.norm(p=config.convergence_norm).item() + 1e-8
                    rel_delta = abs_delta / norm
                else:
                    rel_delta = abs_delta
                max_delta = max(max_delta, rel_delta)
            delta = max_delta
        else:
            # Single-state
            delta = torch.dist(state_new, state, p=config.convergence_norm).item()

        deltas.append(delta)

        if trajectory is not None:
            if isinstance(state_new, list):
                trajectory.append([s.detach().cpu() for s in state_new])
            else:
                trajectory.append(state_new.detach().cpu())

        # Check convergence (with custom hook if provided)
        converged_check = False
        custom_check = getattr(model, "_check_converged", None)
        if custom_check is not None:
            converged_check = custom_check(state_new, state, step_idx)
        elif step_idx > start and delta < threshold:
            converged_check = True

        if converged_check:
            state = state_new
            converged = True
            if hasattr(model, "_on_converged"):
                model._on_converged(steps_taken, delta)
            break

        if hasattr(model, "_on_step_end"):
            model._on_step_end(steps_taken, state_new, delta)

        state = state_new

    if not converged and hasattr(model, "_on_max_steps"):
        model._on_max_steps(steps_taken, deltas[-1] if deltas else 0.0)

    settle_time_ms = (time.monotonic() - settle_start) * 1000

    # Estimate memory (rough: state size in MB)
    if isinstance(state, list):
        total_elements = sum(s.numel() for s in state)
    else:
        total_elements = state.numel()
    memory_mb = total_elements * 4 / 1e6  # float32

    telemetry = SettleTelemetry(
        algorithm=algorithm,
        family=family,
        steps_taken=steps_taken,
        max_steps=max_steps,
        converged=converged,
        final_delta=deltas[-1] if deltas else 0.0,
        deltas=deltas,
        settle_time_ms=settle_time_ms,
        memory_mb=memory_mb,
        hardware=hardware,
        backend=backend,
    )

    return state, steps_taken, converged, telemetry


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


def energy_gradient_descent(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments]
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

    for step in range(steps):  # ruff: ignore[too-many-nested-blocks]
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


def settle_single_state(  # ruff: ignore[complex-structure]
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
        - ``dynamics`` is ``None`` or a dict with the uniform telemetry surface
          shared with :func:`settle_activations_list`: ``"deltas"``,
          ``"final_delta"``, ``"steps_taken"``, ``"converged"`` and
          ``"settle_time_s"``.
    """
    import time

    settle_start = time.monotonic()
    trajectory: list[torch.Tensor] | None = (
        cast("list[torch.Tensor]", [None] * (steps + 1)) if return_trajectory else None
    )
    if trajectory is not None:
        trajectory[0] = h_0
    deltas: list[float] | None = [] if return_dynamics else None

    steps_taken = 0
    converged = False
    h = h_0

    # Gradient checkpointing: during training the unrolled settling loop builds a
    # graph of `steps` sequential `forward_step` calls; backprop then materializes
    # all of them → memory ∝ steps × activations (OOM on CIFAR-10 for deep/high-
    # channel conv setups). Checkpointing the step recomputes it on backward, so
    # memory becomes O(1) in steps regardless of sampled steps/channels/task.
    # It is a no-op under ``no_grad`` (inference/validation), so it adds zero
    # overhead there. Compile is disabled for settling loops, so no dynamo
    # checkpoint conflict (FIX.md §39).
    def _step(state: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return _checkpoint(forward_step, state, x_transformed, use_reentrant=False)
        return forward_step(state, x_transformed)

    # Index into the trajectory buffer — incremented AFTER each snapshot.
    traj_idx = 0

    def warmup() -> None:
        nonlocal h, traj_idx, remaining, steps_taken
        h = _step(h)
        remaining -= 1
        steps_taken += 1
        if deltas is not None:
            deltas.append(torch.dist(h, h_0, p=float("inf")).item())
        if trajectory is not None:
            traj_idx += 1
            trajectory[traj_idx] = h

    remaining = steps

    def main_loop() -> None:
        nonlocal h, traj_idx, remaining, steps_taken, converged

        for step_idx in range(remaining):
            h_new = _step(h)
            steps_taken += 1

            if deltas is not None:
                deltas.append(torch.dist(h_new, h, p=float("inf")).item())

            if _inf_norm_converged(h_new, h, step_idx):
                h = h_new
                converged = True
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
            "steps_taken": steps_taken,
            "converged": converged,
            "settle_time_s": time.monotonic() - settle_start,
        }

    return h, trajectory, dynamics


# ---------------------------------------------------------------------------
# Family B — activations-list settling (StandardEqProp, DirectedEP, etc.)
# ---------------------------------------------------------------------------


def settle_activations_list(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments]
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
    convergence_relative: bool = True,
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]] | None,
    _DynamicsDict | None,
]:
    """Settle an activations-list model (Family B pattern).

    Iterates the bidirectional ``forward_dynamics`` function for a fixed
    number of steps, with optional early convergence detection and
    trajectory/dynamics tracking.

    Convergence (when it fires) is judged on the **max relative change** across
    the hidden+output layers: each layer's p-norm delta is normalised by that
    layer's p-norm, and the worst layer gates the early stop. This is the
    standard relative fixed-point test — unlike an absolute delta summed over
    raw activations (which is dominated by layer magnitude and so almost never
    drops below a tight ``convergence_threshold``, forcing every model to pay
    the full ``max_steps`` settle cost). Spectral-norm-contractive maps then
    terminate in a handful of steps, which is what makes shallow eqprop probes
    complete inside a wall-clock epoch budget.

    Args:
        activations_0: Initial per-layer activations ``[x, h1, ..., out]``.
        forward_dynamics: ``f(activations, beta, target) -> new_activations``.
        steps: Number of settling iterations.
        beta: Nudge strength.
        target: Optional target tensor for nudged phase.
        return_trajectory: If True, record snapshot of activations per step.
        return_dynamics: If True, record per-step deltas and settle stats.
        convergence_norm: p-norm for the per-layer distance and normalisation.
        convergence_threshold: Early-stop tolerance (relative when
            ``convergence_relative``, absolute otherwise).
        convergence_start: Step index after which convergence is checked.
        convergence_relative: Normalise each layer's delta by its own norm so
            the threshold has scale-invariant meaning (default True).

    Returns:
        ``(final_activations, trajectory, dynamics)``.

        - ``trajectory`` is ``None`` or a ``list[list[Tensor]]`` where each
          entry is a CPU-detached snapshot of the activations list at a step.
        - ``dynamics`` is ``None`` or a dict with ``"deltas"`` (per-step
          convergence deltas), ``"final_delta"``, ``"steps_taken"``,
          ``"converged"`` and ``"settle_time_s"`` keys.
    """
    import time

    settle_start = time.monotonic()
    trajectory: list[list[torch.Tensor]] | None = [] if return_trajectory else None
    if trajectory is not None:
        trajectory.append([a.detach().cpu() for a in activations_0])

    # Convergence delta: only compute when we might use it.
    need_delta = return_dynamics or convergence_start < steps
    deltas: list[float] | None = [] if return_dynamics else None
    activations = activations_0

    # Layer norms (p-norm) used as relative denominators for the convergence
    # check. Computed once from the initial activations; ``x`` (layer 0) is
    # fixed and never free-settles, so only the driven output layers count.
    layer_norms = [a.norm(p=convergence_norm).item() + 1e-8 for a in activations_0]

    def _layer_delta(a_new: torch.Tensor, a_prev: torch.Tensor, k: int) -> float:
        abs_delta = torch.dist(a_new, a_prev, p=convergence_norm).item()
        if convergence_relative:
            return abs_delta / layer_norms[k]
        return abs_delta

    converged = False
    steps_taken = 0
    for step_idx in range(steps):
        if need_delta:
            steps_taken = step_idx + 1
        prev = activations
        activations = forward_dynamics(activations, beta, target)

        if need_delta and step_idx > convergence_start:
            max_rel_delta = 0.0
            for k in range(1, len(activations)):
                max_rel_delta = max(
                    max_rel_delta, _layer_delta(activations[k], prev[k], k)
                )

            if deltas is not None:
                deltas.append(max_rel_delta)

            if max_rel_delta < convergence_threshold:
                converged = True
                if trajectory is not None:
                    trajectory.append([a.detach().cpu() for a in activations])
                break
        elif need_delta and deltas is not None:
            # Before convergence_start nothing can stop, but still record the
            # metric for a full convergence profile when debug asked for it.
            max_rel_delta = 0.0
            for k in range(1, len(activations)):
                max_rel_delta = max(
                    max_rel_delta, _layer_delta(activations[k], prev[k], k)
                )
            deltas.append(max_rel_delta)
        else:
            steps_taken += 1

        if trajectory is not None:
            trajectory.append([a.detach().cpu() for a in activations])

    dynamics: _DynamicsDict | None = None
    if deltas is not None:
        dynamics = {
            "deltas": deltas,
            "final_delta": deltas[-1] if deltas else 0.0,
            "steps_taken": steps_taken,
            "converged": converged,
            "settle_time_s": time.monotonic() - settle_start,
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
        # Contractive models (spectral norm) converge well before ``max_steps``;
        # skipping the tail of the settle saves most of the forward cost while
        # still producing a valid near-fixed-point for implicit differentiation
        # (mirrors the early stop in ``settle_single_state``).
        early_start = getattr(model, "convergence_start", 5)
        threshold_early = getattr(model, "convergence_threshold", 2e-4)
        threshold_late = getattr(model, "convergence_threshold_late", 1e-4)
        transition_step = getattr(model, "convergence_transition", 10)

        with torch.no_grad():
            h = h_init

            if should_freeze_sn and remaining_steps > 0:
                h = model.forward_step(h, x_transformed)
                remaining_steps -= 1
                model.eval()

            try:
                step_idx = 0
                while step_idx < remaining_steps:
                    h_new = model.forward_step(h, x_transformed)
                    step_idx += 1
                    if step_idx > early_start:
                        d = torch.dist(h_new, h, p=float("inf"))
                        if d.item() < (
                            threshold_late
                            if step_idx > transition_step
                            else threshold_early
                        ):
                            h = h_new
                            break
                    h = h_new
            finally:
                if should_freeze_sn:
                    model.train()

        ctx.save_for_backward(h, x_transformed, *params)
        return h

    @staticmethod
    def backward(  # ruff: ignore[too-many-locals]
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
    "EquilibriumSettleProtocol",
    "SettleConfig",
    "SettleProtocol",
    "SettleTelemetry",
    "_run_with_sn_freeze",
    "energy_gradient_descent",
    "settle_activations_list",
    "settle_single_state",
    "settle_state",
    "settle_universal",
]
