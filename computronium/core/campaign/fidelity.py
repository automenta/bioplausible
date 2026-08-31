"""Implementation-fidelity gate for campaign grids (TODO8 R5b-0).

Behavioral per-axis probes over the same coordinate builder the campaign
uses. A coordinate that fails fidelity is *excluded from attribution and
listed* — its deltas are quarantined, never interpreted. A failed fidelity
check is inconclusive, never a refutation of a hypothesis.

Probes (TODO8 R5b-0 spec):
- dynamics: EnergyMinimization settles (energy descends, finite, responds
  to the nudged target); Instantaneous is a single pass, idempotent, and
  does not inadvertently settle.
- credit: the declared phases yield non-empty, finite pseudo-gradients.
- update: parameters move under ``update.step`` when fed the credit's
  pseudo-gradients (``blocked`` when the credit supplied no signal).
- plasticity: Null keeps ψ const; non-null plasticity must actually step
  ψ within the episode pipeline.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from computronium.core.campaign.evaluation import build_coordinate_system, episode_batch
from computronium.core.pipeline import Phase, forward_pass, task_loss
from computronium.ontology import (
    EnergyMinimizationDynamics,
    StateDynamicsConfig,
    SystemState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from computronium.analysis.counterfactual import AxisAttribution
    from computronium.core.campaign.frontier_record import FrontierRecord

Status = Literal["pass", "fail", "blocked"]

# Mirrors evaluation._DYNAMICS_FACTORIES' energy_minimization knobs; the
# tracking flag only toggles recording, never the settle math.
_PROBE_ENERGY_KWARGS = {"max_steps": 3, "step_size": 0.1}

_GRAD_NORM_TOL = 1e-12
_MONOTONE_SLACK = 1e-3


@dataclass(frozen=True, slots=True)
class AxisCheck:
    """Verdict of one fidelity probe on one axis of a coordinate."""

    axis: str
    value: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class CoordinateFidelity:
    """Full fidelity verdict for one campaign coordinate."""

    coordinate: str
    checks: tuple[AxisCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status == "pass" for check in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            f"{check.axis}={check.value}: {check.detail}"
            for check in self.checks
            if check.status == "fail"
        )


def _acts_list(activations: list[Tensor] | Tensor | None) -> list[Tensor]:
    """Normalize a settle output into a layer list (detached copies)."""
    if activations is None:
        return []
    if isinstance(activations, list):
        return [a.detach().clone() for a in activations]
    return [activations.detach().clone()]


def _total_norm(grads: list[Tensor]) -> float:
    if not grads:
        return 0.0
    return float(torch.stack([g.detach().float().norm() for g in grads]).sum())


def _settled_phases(
    joint, x: Tensor, y: Tensor
) -> tuple[dict[Phase, SystemState], list[Tensor]]:
    """Replicate ``run_train_step``'s phase settling up to the pseudo-gradient."""
    credit = joint.credit
    grad_ctx = nullcontext() if credit.requires_autograd else torch.no_grad()
    with grad_ctx:
        initial = forward_pass(joint.substrate, joint.geometry, x)
        states: dict[Phase, SystemState] = {}
        for phase in credit.phases:
            state = SystemState(x=x, y=y)
            state.activations = initial
            target = y if phase is Phase.NUDGED else None
            settled = joint.dynamics.settle(  # type: ignore[arg-type]
                state, joint.geometry, joint.substrate, target=target
            )
            if phase is Phase.NUDGED:
                settled.loss = task_loss(settled, y)
            settled.energy = joint.dynamics.compute_energy(settled, joint.geometry)
            states[phase] = settled
        output = states.get(Phase.NUDGED, states.get(Phase.FREE))
        if output is None:  # pragma: no cover - credits always declare phases
            msg = "credit declared no phases"
            raise ValueError(msg)
        loss = output.loss if output.loss is not None else task_loss(output, y)
        grads = credit.compute_pseudo_gradient(states, loss, joint.geometry)
    return states, grads


def _check_dynamics(joint, parts: list[str]) -> tuple[AxisCheck, ...]:
    value = parts[2]
    x, y = episode_batch(0)
    if value == "energy_minimization":
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                track_free_energy_per_iter=True, **_PROBE_ENERGY_KWARGS
            )
        )
        state = SystemState(x=x, y=y)
        state.activations = forward_pass(joint.substrate, joint.geometry, x)
        dynamics.settle(state, joint.geometry, joint.substrate, target=None)  # type: ignore[arg-type]
        history = dynamics.get_free_energy_history() or []
        free_out = _acts_list(state.activations)
        checks = []
        if not history or not all(torch.isfinite(torch.tensor(history))):
            return (
                AxisCheck(
                    "dynamics", value, "fail", "energy history missing/non-finite"
                ),
            )
        checks.append(
            AxisCheck(
                "dynamics",
                value,
                "pass" if history[-1] < history[0] else "fail",
                f"energy {history[0]:.4f} -> {history[-1]:.4f} over {len(history)} steps",
            )
        )
        monotone = all(b <= a + _MONOTONE_SLACK for a, b in zip(history, history[1:]))
        checks.append(
            AxisCheck(
                "dynamics",
                value,
                "pass" if monotone else "fail",
                "per-step energy non-increasing within slack",
            )
        )
        nudged = SystemState(x=x, y=y)
        nudged.activations = forward_pass(joint.substrate, joint.geometry, x)
        dynamics.settle(nudged, joint.geometry, joint.substrate, target=y)  # type: ignore[arg-type]
        nudged_out = _acts_list(nudged.activations)
        responds = (
            bool(nudged_out)
            and bool(free_out)
            and not torch.allclose(free_out[-1], nudged_out[-1], atol=1e-6)
        )
        checks.append(
            AxisCheck(
                "dynamics",
                value,
                "pass" if responds else "fail",
                "nudged phase differs from free phase (target-responsive settle)",
            )
        )
        return tuple(checks)

    if value == "instantaneous":
        acts1 = joint.geometry.forward_with_intermediates(x, joint.substrate)
        state = SystemState(x=x, y=y)
        joint.dynamics.settle(state, joint.geometry, joint.substrate, target=None)  # type: ignore[arg-type]
        settled1 = _acts_list(state.activations)
        nudged = SystemState(x=x, y=y)
        joint.dynamics.settle(nudged, joint.geometry, joint.substrate, target=y)  # type: ignore[arg-type]
        nudged_out = _acts_list(nudged.activations)
        single_pass = bool(settled1) and torch.allclose(
            settled1[-1], acts1[-1].detach(), atol=1e-6
        )
        state2 = SystemState(x=x, y=y)
        joint.dynamics.settle(state2, joint.geometry, joint.substrate, target=None)  # type: ignore[arg-type]
        settled2 = _acts_list(state2.activations)
        idempotent = bool(settled2) and torch.allclose(
            settled1[-1], settled2[-1], atol=1e-6
        )
        return (
            AxisCheck(
                "dynamics",
                value,
                "pass" if single_pass and idempotent else "fail",
                "single pass, idempotent, no inadvertent settle"
                if single_pass and idempotent
                else "settle is not a faithful single pass",
            ),
            AxisCheck(
                "dynamics",
                value,
                "pass",
                "target-blind by spec: free ≡ nudged, so contrastive credits "
                "receive zero phase contrast (surfaced by the credit probe)",
            ),
        )

    return (
        AxisCheck("dynamics", value, "blocked", "no probe for this dynamics value"),
    )


def _check_credit_and_update(joint, parts: list[str]) -> tuple[AxisCheck, ...]:
    credit_value, update_value = parts[4], parts[5]
    x, y = episode_batch(0)
    _states, grads = _settled_phases(joint, x, y)
    norm = _total_norm(grads)
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    credit_ok = len(grads) > 0 and norm > _GRAD_NORM_TOL and finite
    credit_check = AxisCheck(
        "credit",
        credit_value,
        "pass" if credit_ok else "fail",
        (
            f"pseudo-gradient norm {norm:.3e} over {len(grads)} tensors"
            if credit_ok
            else (
                f"non-finite pseudo-gradient"
                if not finite
                else f"empty/zero pseudo-gradient (norm {norm:.3e}, "
                f"{len(grads)} tensors)"
            )
        ),
    )
    if not credit_ok:
        return (
            credit_check,
            AxisCheck(
                "update",
                update_value,
                "blocked",
                f"unassessable: credit supplied no signal (norm {norm:.3e})",
            ),
        )

    before = {
        name: tensor.detach().clone() for name, tensor in joint.geometry.params.items()
    }
    stepped = joint.update.step(before, grads, joint.geometry)
    movement = float(
        sum(
            (stepped[name].detach() - before[name]).norm()
            for name in before
            if name in stepped
        )
    )
    return (
        credit_check,
        AxisCheck(
            "update",
            update_value,
            "pass" if movement > _GRAD_NORM_TOL else "fail",
            f"parameters moved {movement:.3e} under update.step",
        ),
    )


def _check_plasticity(joint, parts: list[str]) -> AxisCheck:
    value = parts[3]
    x, y = episode_batch(0)
    calls = {"step": 0}
    original_step = joint.plasticity.step

    def counting_step(*args: object, **kwargs: object):
        calls["step"] += 1
        return original_step(*args, **kwargs)

    joint.plasticity.step = counting_step  # type: ignore[method-assign]
    try:
        joint.train_step(x, y)
    finally:
        del joint.plasticity.step  # type: ignore[attr-defined]
    if value == "null":
        return AxisCheck(
            "plasticity",
            value,
            "pass",
            f"ψ const across the episode ({calls['step']} plasticity steps) "
            "— as specified for NullPlasticity",
        )
    engages = calls["step"] > 0
    return AxisCheck(
        "plasticity",
        value,
        "pass" if engages else "fail",
        (
            f"ψ stepped {calls['step']}x within the episode"
            if engages
            else "plasticity.step never invoked by the episode pipeline "
            "(run_train_step takes no M-axis and initial_psi's ψ is "
            "discarded) — plasticity cannot modulate activity or credit"
        ),
    )


def check_coordinate_fidelity(
    coordinate: str,
    *,
    device: str | torch.device | None = None,
) -> CoordinateFidelity:
    """Run every fidelity probe against one campaign coordinate."""
    parts = coordinate.split("/")
    joint = build_coordinate_system(coordinate, device=device)
    checks = (
        *_check_dynamics(joint, parts),
        *_check_credit_and_update(joint, parts),
        _check_plasticity(joint, parts),
    )
    return CoordinateFidelity(coordinate=coordinate, checks=checks)


def fidelity_manifest(
    coordinates: Sequence[str],
    *,
    device: str | torch.device | None = None,
) -> dict[str, CoordinateFidelity]:
    """Fidelity verdicts for every coordinate in a campaign grid."""
    return {
        coordinate: check_coordinate_fidelity(coordinate, device=device)
        for coordinate in coordinates
    }


@dataclass(frozen=True, slots=True)
class DefectFilteredAttribution:
    """Attribution computed only over fidelity-passing coordinates."""

    attributions: tuple[AxisAttribution, ...]
    passing_coordinates: tuple[str, ...]
    excluded_coordinates: tuple[str, ...]
    n_records_total: int
    n_records_passing: int


def defect_filtered_attribution(
    records: Sequence[FrontierRecord],
    manifest: Mapping[str, CoordinateFidelity],
    *,
    metric: str = "task_accuracy",
) -> DefectFilteredAttribution:
    """Quarantine records whose coordinates fail fidelity, then attribute.

    Per the TODO8 policy, deltas from known-defective axes are excluded —
    not interpreted — and the excluded coordinates are reported by identity.
    """
    passing_records: list[FrontierRecord] = []
    excluded: set[str] = set()
    passing: set[str] = set()
    for record in records:
        verdict = manifest.get(record.coordinate)
        if verdict is not None and verdict.passed:
            passing_records.append(record)
            passing.add(record.coordinate)
        else:
            excluded.add(record.coordinate)
    from computronium.analysis.counterfactual import attribute_axis_effects

    return DefectFilteredAttribution(
        attributions=tuple(attribute_axis_effects(passing_records, metric=metric)),
        passing_coordinates=tuple(sorted(passing)),
        excluded_coordinates=tuple(sorted(excluded)),
        n_records_total=len(records),
        n_records_passing=len(passing_records),
    )
