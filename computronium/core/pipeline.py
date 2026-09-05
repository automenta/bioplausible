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

from computronium.ontology import (
    Geometry,
    ParameterUpdate,
    Phase,
    StateDynamics,
    Substrate,
    SystemState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from computronium.ontology import CreditAssignment

type PhaseStates = Mapping[Phase, SystemState]

__all__ = [
    "METRIC_SCHEMA",
    "apply_autograd_update",
    "forward_pass",
    "phase_states",
    "run_forward",
    "run_train_step",
    "task_loss",
]

# imp-46: closed metric schema of ``run_train_step``. Claim-grade keys are the
# post-update target-free ``free_*`` reads; output-phase diagnostics carry the
# ``nudged_fit_accuracy`` name. A bare ``accuracy`` key must never reappear.
METRIC_SCHEMA: frozenset[str] = frozenset({
    "loss",
    "energy",
    "nudged_fit_accuracy",
    "free_loss",
    "free_energy",
    "free_accuracy",
})


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


def run_train_step(  # 5/6-axis pipeline contract + x/y  # ruff: ignore[too-many-arguments, too-many-locals]
    substrate: Substrate,
    geometry: Geometry,
    dynamics: StateDynamics,
    credit: CreditAssignment,
    update: ParameterUpdate,
    x: Tensor,
    y: Tensor,
    *,
    plasticity: object | None = None,
    psi: dict[str, Tensor] | None = None,
    context: object | None = None,
) -> dict[str, float]:
    """Execute one training step through the 5/6-layer pipeline.

    Settles exactly the phases declared by ``credit.phases``; runs under
    ``no_grad`` unless ``credit.requires_autograd`` is declared (pseudo-
    gradients are consumed as plain values — keeping settle graphs from
    accumulating is the default for every non-autograd family).

    If ``plasticity`` and ``psi`` are provided, the P-axis is engaged:
    ψ steps once per episode via ``plasticity.step(psi, z, context)`` and
    can modulate activity via ``plasticity.modulate(activations, psi)`` if
    the primitive implements it. J2/J3 invariants: θ untouched intra-episode;
    ψ mutates only via ``plasticity.step``.

    Returns:
        Metrics with parity-guaranteed keys ``loss``/``energy``/``nudged_fit_accuracy``
        (output-phase state — target-conditioned for contrastive credits; training
        diagnostics only) plus ``free_loss``/``free_energy``/``free_accuracy``
        (post-update target-free settle — the only claim-grade metrics, imp-20/imp-46).
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

            # P-axis: ψ steps ONCE per episode, on the first phase's settled
            # activity (real fast-weight content: the Hebbian outer over
            # settled pre/post, not the raw target). Modulation applies to
            # every phase after the step.
            if plasticity is not None and psi is not None:
                if phase is credit.phases[0] and context is not None:
                    from computronium.state import CompositeState

                    acts = settled.activations
                    post = acts[-1] if isinstance(acts, list) else acts
                    z = CompositeState(
                        activity={"x": x, "y": post}, plastic=psi, substrate={}
                    )
                    psi = plasticity.step(psi, z, context)
                modulate = getattr(plasticity, "modulate", None)
                if modulate is not None:
                    settled.activations = modulate(settled.activations, psi)

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

        # Post-update, target-free forward+settle for honest learning metrics.
        # This is the "free" readout: what the model actually predicts without
        # supervision leakage. Legacy "accuracy" = nudged-settle fit (may be leaked).
        with torch.no_grad():
            free_state = SystemState(x=x, y=y)
            free_state.activations = forward_pass(substrate, geometry, x)
            free_settled = dynamics.settle(free_state, geometry, substrate, target=None)
            free_loss = task_loss(free_settled, y)
            free_energy = dynamics.compute_energy(free_settled, geometry)
            free_accuracy = free_settled.metrics.get("accuracy", 0.0)

        metrics = {
            "loss": _scalar(loss),
            "energy": _scalar(output.energy),
            "nudged_fit_accuracy": output.metrics.get(
                "accuracy", 0.0
            ),  # output-phase fit; target-conditioned when a NUDGED phase ran
            "free_loss": _scalar(free_loss),
            "free_energy": _scalar(free_energy),
            "free_accuracy": free_accuracy,
        }
        metrics.update({
            k: v
            for k, v in output.metrics.items()
            if isinstance(v, (int, float))
            and k not in {"accuracy", "free_accuracy", "nudged_fit_accuracy"}
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


def apply_autograd_update(system: object) -> None:
    """Consolidate autograd gradients through the System's own ParameterUpdate.

    For harnesses that compute a custom task loss outside the standard
    pipeline (EWC penalties, probe objectives, annealing probes): Δθ must
    still flow through the composed update rule — never an external torch
    optimizer, which would bypass the U-axis entirely.
    """
    from computronium.ontology.utils import _learnable_weight_names

    geometry = system.geometry  # type: ignore[attr-defined]
    params = geometry.params
    names = _learnable_weight_names(params)
    grads = [params[n].grad for n in names]
    if all(g is None for g in grads):
        return
    pseudo_grads = [
        g if g is not None else torch.zeros_like(params[n])
        for n, g in zip(names, grads, strict=True)
    ]
    geometry.update_params(
        system.update.step(params, pseudo_grads, geometry)  # type: ignore[attr-defined]
    )
