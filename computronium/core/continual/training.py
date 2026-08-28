"""Continual learning training step functions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor

from computronium.core.continual.constants import CL_CLASSES_PER_TASK
from computronium.core.joint.state import CompositeState
from computronium.core.ontology import Phase, SystemState
from computronium.core.pipeline import forward_pass


def _masked_task_loss(state, local_y: Tensor, task_start: int, task_end: int) -> Tensor:
    """Compute cross-entropy loss only on task-relevant logits."""
    acts = state.activations
    if acts is None:
        return torch.tensor(0.0, device=local_y.device)
    logits = acts[-1] if isinstance(acts, list) else acts  # [batch, 10]
    task_logits = logits[:, task_start:task_end]  # [batch, 2]
    loss = F.cross_entropy(task_logits, local_y)
    with torch.no_grad():
        acc = (task_logits.argmax(dim=-1) == local_y).float().mean().item()
    state.metrics = {**state.metrics, "accuracy": acc}
    return loss


def run_continual_train_step(
    joint_system,
    x: Tensor,
    y: Tensor,
    task_id: int,
    psi: dict[str, Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, Tensor] | None]:
    """Execute one training step through the joint system with task-masked loss and plasticity stepping.

    The joint system outputs 10-class logits. We mask the loss to only the
    current task's 2 classes (task_id * 2 : task_id * 2 + 2).

    The labels y are already 0/1 (local to the task) from SplitMNIST.

    This uses the joint system's credit assignment and parameter update,
    ensuring ψ/θ decoupling (FastWeightPlasticity) and other components
    are actually invoked. Also steps the plasticity to update ψ.

    Returns:
        Tuple of (metrics, updated_psi)
    """
    # Labels are already 0/1 from SplitMNIST
    local_y = y

    # Task logit slice
    task_start = task_id * CL_CLASSES_PER_TASK
    task_end = task_start + CL_CLASSES_PER_TASK

    # Get the joint system components
    substrate = joint_system.substrate
    geometry = joint_system.geometry
    dynamics = joint_system.dynamics
    credit = joint_system.credit
    update = joint_system.update
    plasticity = joint_system.plasticity

    # Initialize plastic state if needed
    if psi is None and hasattr(plasticity, "initial_psi") and plasticity is not None:
        psi = plasticity.initial_psi(joint_system.context, batch_size=x.shape[0])
    # Ensure psi is on the same device as x and matches batch size
    if psi is not None:
        device = x.device
        batch_size = x.shape[0]
        new_psi = {}
        for k, v in psi.items():
            if v.shape[0] != batch_size:
                if v.shape[0] == 1:
                    new_psi[k] = v.expand(batch_size, -1).to(device)
                else:
                    new_psi[k] = v[:batch_size].to(device)
            else:
                new_psi[k] = v.to(device)
        psi = new_psi

    # Run the pipeline with masked target
    grad_ctx = nullcontext() if credit.requires_autograd else torch.no_grad()
    with grad_ctx:
        states: dict[Phase, SystemState] = {}
        initial_activations = forward_pass(substrate, geometry, x)

        for phase in credit.phases:
            state = SystemState(x=x, y=local_y)  # Use local_y for nudged phase
            state.activations = initial_activations
            target = local_y + task_start if phase is Phase.NUDGED else None
            settled = dynamics.settle(state, geometry, substrate, target=target)
            if phase is Phase.NUDGED:
                # Compute loss only on task-relevant logits
                settled.loss = _masked_task_loss(settled, local_y, task_start, task_end)
            settled.energy = dynamics.compute_energy(settled, geometry)
            states[phase] = settled

        output = states.get(Phase.NUDGED, states.get(Phase.FREE))
        if output is None:
            output = SystemState(x=x, y=local_y)
            output.activations = initial_activations
        loss = output.loss
        if loss is None:
            loss = _masked_task_loss(output, local_y, task_start, task_end)
        elif not isinstance(loss, Tensor):
            loss = torch.as_tensor(loss)
        if output.energy is None:
            output.energy = dynamics.compute_energy(output, geometry)

        pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)
        geometry.update_params(update.step(geometry.params, pseudo_grads, geometry))

        # Step plasticity if present (ψ update)
        if psi is not None and hasattr(plasticity, "step") and plasticity is not None:
            # Create CompositeState from settled output
            acts = output.activations
            if acts is not None:
                final_acts = acts[-1] if isinstance(acts, list) else acts
                z = CompositeState(
                    activity={"x": x, "y": final_acts},
                    plastic=psi,
                    substrate={},
                )
                psi = plasticity.step(psi, z, joint_system.context)

        loss_val = loss.item() if isinstance(loss, Tensor) else float(loss)
        energy_val = (
            output.energy.item()
            if isinstance(output.energy, Tensor)
            else float(output.energy)
            if output.energy is not None
            else 0.0
        )
        metrics = {
            "loss": loss_val,
            "energy": energy_val,
            "accuracy": output.metrics.get("accuracy", 0.0),
        }
        metrics.update({
            k: v
            for k, v in output.metrics.items()
            if isinstance(v, (int, float)) and k != "accuracy"
        })
        return metrics, psi


def _continual_step(
    model,
    x: Tensor,
    y: Tensor,
    task_id: int,
    extra_loss_fn: Callable[[object, object, int], Tensor | None],
    si_tracker=None,
) -> dict[str, float]:
    """Run the joint training pipeline, folding an extra loss term into the task loss.

    ``extra_loss_fn(model, output, task_id)`` returns an additive regularizer /
    distillation term (or None). The combined loss drives credit assignment so the
    term actually influences ``theta`` (unlike a post-hoc ``.backward()`` with no
    optimizer step).

    If si_tracker is provided, accumulates pseudo-gradients for SI importance computation.
    """
    substrate = model.substrate
    geometry = model.geometry
    dynamics = model.dynamics
    credit = model.credit
    update = model.update
    plasticity = model.plasticity

    task_start = task_id * CL_CLASSES_PER_TASK
    task_end = task_start + CL_CLASSES_PER_TASK
    local_y = y

    psi = getattr(model, "_psi", None)
    if psi is None and hasattr(plasticity, "initial_psi") and plasticity is not None:
        psi = plasticity.initial_psi(model.context, batch_size=x.shape[0])
    if psi is not None:
        device = x.device
        batch_size = x.shape[0]
        new_psi: dict[str, Tensor] = {}
        for k, v in psi.items():
            if v.shape[0] != batch_size:
                if v.shape[0] == 1:
                    new_psi[k] = v.expand(batch_size, -1).to(device)
                else:
                    new_psi[k] = v[:batch_size].to(device)
            else:
                new_psi[k] = v.to(device)
        psi = new_psi

    grad_ctx = nullcontext() if credit.requires_autograd else torch.no_grad()
    with grad_ctx:
        states: dict[Phase, SystemState] = {}
        initial_activations = forward_pass(substrate, geometry, x)

        for phase in credit.phases:
            state = SystemState(x=x, y=local_y)
            state.activations = initial_activations
            target = local_y + task_start if phase is Phase.NUDGED else None
            settled = dynamics.settle(state, geometry, substrate, target=target)
            if phase is Phase.NUDGED:
                settled.loss = _masked_task_loss(settled, local_y, task_start, task_end)
            settled.energy = dynamics.compute_energy(settled, geometry)
            states[phase] = settled

        output = states.get(Phase.NUDGED, states.get(Phase.FREE))
        if output is None:
            output = SystemState(x=x, y=local_y)
            output.activations = initial_activations
        loss = output.loss
        if loss is None:
            loss = _masked_task_loss(output, local_y, task_start, task_end)
        elif not isinstance(loss, Tensor):
            loss = torch.as_tensor(loss)
        if output.energy is None:
            output.energy = dynamics.compute_energy(output, geometry)

        extra = extra_loss_fn(model, output, task_id)
        total_loss = loss if extra is None else loss + extra

        pseudo_grads = credit.compute_pseudo_gradient(states, total_loss, geometry)

        # Accumulate pseudo-gradients for SI if tracker provided
        if si_tracker is not None:
            si_tracker.accumulate_pseudo_grads(pseudo_grads, geometry)

        geometry.update_params(update.step(geometry.params, pseudo_grads, geometry))

        if psi is not None and hasattr(plasticity, "step") and plasticity is not None:
            acts = output.activations
            if acts is not None:
                final_acts = acts[-1] if isinstance(acts, list) else acts
                z = CompositeState(
                    activity={"x": x, "y": final_acts},
                    plastic=psi,
                    substrate={},
                )
                psi = plasticity.step(psi, z, model.context)

    model._psi = psi
    loss_val = (
        total_loss.item() if isinstance(total_loss, Tensor) else float(total_loss)
    )  # noqa: E501
    energy_val = (
        output.energy.item()
        if isinstance(output.energy, Tensor)
        else float(output.energy)
        if output.energy is not None
        else 0.0
    )
    metrics: dict[str, float] = {
        "loss": loss_val,
        "energy": energy_val,
        "accuracy": output.metrics.get("accuracy", 0.0),
    }
    metrics.update({
        k: v
        for k, v in output.metrics.items()
        if isinstance(v, (int, float)) and k != "accuracy"
    })
    return metrics


def _lwf_train_step(
    model,
    x: Tensor,
    y: Tensor,
    task_id: int,
    lwf_loss_fn,
) -> dict[str, float]:
    """LwF training step: task CE + distillation from a frozen previous model."""
    prev_logits = None
    prev_model = lwf_loss_fn.prev_model
    if prev_model is not None and task_id > 0:
        prev_model.eval()
        with torch.no_grad():
            prev_logits = prev_model(x, task_id=task_id)

    model.train()

    def extra_loss_fn(_model, output, tid):
        if prev_logits is None:
            return None
        logits = (
            output.activations[-1]
            if isinstance(output.activations, list)
            else output.activations
        )
        return lwf_loss_fn.distill_only(logits, tid, prev_logits)

    return _continual_step(model, x, y, task_id, extra_loss_fn)


def _si_train_step(
    model,
    x: Tensor,
    y: Tensor,
    task_id: int,
    si_tracker,
) -> dict[str, float]:
    """SI training step: task loss + importance-weighted consolidation penalty."""
    model.train()

    def extra_loss_fn(_model, _output, _tid):
        return si_tracker.regularization_loss()

    return _continual_step(model, x, y, task_id, extra_loss_fn, si_tracker=si_tracker)


__all__ = [
    "_continual_step",
    "_lwf_train_step",
    "_masked_task_loss",
    "_si_train_step",
    "run_continual_train_step",
]
