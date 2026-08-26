"""Family-neutral training pipeline: the single canonical train-step loop.

Every composed system generation (5-D ``_ComposedSystem``, 6-D
``_JointSystem``, adapted legacy models) delegates here. The loop settles
exactly the phases the credit rule declares (``credit.phases``), enables
autograd through settling only when the rule declares
``requires_autograd``, and returns a parity-guaranteed metrics schema.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.ontology import (
    Geometry,
    ParameterUpdate,
    Phase,
    StateDynamics,
    Substrate,
    SystemState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from computronium.core.ontology import CreditAssignment

type PhaseStates = Mapping[Phase, SystemState]

__all__ = ["forward_pass", "phase_states", "run_forward", "run_train_step", "task_loss"]


def phase_states(
    *, free: SystemState | None = None, nudged: SystemState | None = None
) -> dict[Phase, SystemState]:
    """Build the phase-keyed state mapping consumed by credit rules."""
    states: dict[Phase, SystemState] = {}
    if free is not None:
        states[Phase.FREE] = free
    if nudged is not None:
        states[Phase.NUDGED] = nudged
    return states


def forward_pass(
    substrate: Substrate, geometry: Geometry, x: Tensor
) -> list[Tensor] | Tensor | None:
    """Substrate+geometry forward with state-noise injection."""
    acts = geometry.forward(x, substrate)
    if acts is not None:
        acts = substrate.inject_state_noise(acts)
    return acts


def task_loss(state: SystemState, y: Tensor) -> Tensor:
    """Cross-entropy on the state's output activations.

    Writes accuracy into ``state.metrics`` (pre-aggregated float contract).
    """
    acts = state.activations
    if acts is None:
        return torch.tensor(0.0)
    logits = acts[-1] if isinstance(acts, list) else acts
    loss = torch.nn.functional.cross_entropy(logits, y)
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
    state.metrics = {**state.metrics, "accuracy": acc}
    return loss


def _scalar(value: Tensor | float) -> float:
    return value.item() if isinstance(value, Tensor) else float(value)


def run_train_step(  # noqa: PLR0913, PLR0917  # 5-axis pipeline contract + x/y
    substrate: Substrate,
    geometry: Geometry,
    dynamics: StateDynamics,
    credit: CreditAssignment,
    update: ParameterUpdate,
    x: Tensor,
    y: Tensor,
) -> dict[str, float]:
    """Execute one training step through the 5-layer pipeline.

    Settles exactly the phases declared by ``credit.phases``; runs under
    ``no_grad`` unless ``credit.requires_autograd`` is declared (pseudo-
    gradients are consumed as plain values — keeping settle graphs from
    accumulating is the default for every non-autograd family).

    Returns:
        Metrics with parity-guaranteed keys ``loss``/``energy``/``accuracy``
        plus any float extras the output state or dynamics expose.
    """
    grad_ctx = nullcontext() if credit.requires_autograd else torch.no_grad()
    with grad_ctx:
        states: dict[Phase, SystemState] = {}
        initial_activations = forward_pass(substrate, geometry, x)

        for phase in credit.phases:
            state = SystemState(x=x, y=y)
            state.activations = initial_activations
            target = y if phase is Phase.NUDGED else None
            settled = dynamics.settle(state, geometry, substrate, target=target)
            if phase is Phase.NUDGED:
                settled.loss = task_loss(settled, y)
            settled.energy = dynamics.compute_energy(settled, geometry)
            states[phase] = settled

        output = states.get(Phase.NUDGED, states.get(Phase.FREE))
        if output is None:
            # No declared phases: activity comes from the bare forward pass.
            output = SystemState(x=x, y=y)
            output.activations = initial_activations
        loss = output.loss
        if loss is None:
            loss = task_loss(output, y)
            output.loss = loss
        elif not isinstance(loss, Tensor):
            loss = torch.as_tensor(loss)
        if output.energy is None:
            output.energy = dynamics.compute_energy(output, geometry)

        pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)
        geometry.update_params(update.step(geometry.params, pseudo_grads, geometry))

        metrics = {
            "loss": _scalar(loss),
            "energy": _scalar(output.energy),
            "accuracy": output.metrics.get("accuracy", 0.0),
        }
        metrics.update({
            k: v
            for k, v in output.metrics.items()
            if isinstance(v, (int, float)) and k != "accuracy"
        })
        return metrics


def run_forward(
    substrate: Substrate, geometry: Geometry, dynamics: StateDynamics, x: Tensor
) -> Tensor:
    """Inference forward pass (free-phase activity generation, no update)."""
    state = SystemState(x=x)
    state.activations = forward_pass(substrate, geometry, x)
    settled = dynamics.settle(state, geometry, substrate, target=None)
    acts = settled.activations
    if acts is None:
        return torch.empty(0)
    return acts[-1] if isinstance(acts, list) else acts
