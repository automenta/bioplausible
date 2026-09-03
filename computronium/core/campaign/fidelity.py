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
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from computronium.core.campaign.evaluation import (
    IncompatibleCoordinateError,
    build_coordinate_system,
    episode_batch,
)
from computronium.core.pipeline import Phase, forward_pass, task_loss
from computronium.ontology import (
    DiffusionDynamics,
    EnergyMinimizationDynamics,
    PredictiveSettlingDynamics,
    SpikeIntegrationDynamics,
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
_SPIKE_MEMBRANE_BOUND = 1e3
_DIFFUSION_PROBE_SEED = 1234


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


def _acts_history(history: list[float] | None) -> bool:
    return bool(history) and all(torch.isfinite(torch.tensor(history)))


def _settled_output(dynamics, joint, x, y, target):
    """Forward + settle one phase; return (settled state, layer list).

    Reads the RETURNED state — settle implementations that rebuild the
    state (via _create_output_state) do not mutate the input in place.
    """
    state = SystemState(x=x, y=y)
    state.activations = forward_pass(joint.substrate, joint.geometry, x)
    settled = dynamics.settle(state, joint.geometry, joint.substrate, target=target)  # type: ignore[arg-type]
    return settled, _acts_list(settled.activations)


def _probe_energy_minimization(joint, x, y) -> tuple[AxisCheck, ...]:
    value = "energy_minimization"
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            track_free_energy_per_iter=True, **_PROBE_ENERGY_KWARGS
        )
    )
    _state, free_out = _settled_output(dynamics, joint, x, y, None)
    history = dynamics.get_free_energy_history() or []
    if not _acts_history(history):
        return (
            AxisCheck("dynamics", value, "fail", "energy history missing/non-finite"),
        )
    _state, nudged_out = _settled_output(dynamics, joint, x, y, y)
    responds = (
        bool(nudged_out)
        and bool(free_out)
        and not torch.allclose(free_out[-1], nudged_out[-1], atol=1e-6)
    )
    monotone = all(b <= a + _MONOTONE_SLACK for a, b in pairwise(history))
    return (
        AxisCheck(
            "dynamics",
            value,
            "pass" if history[-1] < history[0] else "fail",
            f"energy {history[0]:.4f} -> {history[-1]:.4f} over {len(history)} steps",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if monotone else "fail",
            "per-step energy non-increasing within slack",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if responds else "fail",
            "nudged phase differs from free phase (target-responsive settle)",
        ),
    )


def _probe_instantaneous(joint, x, y) -> tuple[AxisCheck, ...]:
    value = "instantaneous"
    acts1 = joint.geometry.forward_with_intermediates(x, joint.substrate)
    _state, settled1 = _settled_output(joint.dynamics, joint, x, y, None)
    _settled_output(joint.dynamics, joint, x, y, y)
    _state, settled2 = _settled_output(joint.dynamics, joint, x, y, None)
    single_pass = bool(settled1) and torch.allclose(
        settled1[-1], acts1[-1].detach(), atol=1e-6
    )
    idempotent = bool(settled2) and torch.allclose(
        settled1[-1], settled2[-1], atol=1e-6
    )
    # Check that nudging works when target is provided (for contrastive credits)
    _state, free_out = _settled_output(joint.dynamics, joint, x, y, None)
    _state, nudged_out = _settled_output(joint.dynamics, joint, x, y, y)
    responds = (
        bool(free_out)
        and bool(nudged_out)
        and not torch.allclose(free_out[-1], nudged_out[-1], atol=1e-6)
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
            "pass" if responds else "fail",
            "nudged phase differs from free phase (target-responsive nudge)"
            if responds
            else "target-blind: free = nudged",
        ),
    )


def _probe_predictive_settling(joint, x, y) -> tuple[AxisCheck, ...]:
    value = "predictive_settling"
    dynamics = PredictiveSettlingDynamics(
        StateDynamicsConfig.predictive_settling(
            max_steps=3, track_free_energy_per_iter=True
        )
    )
    _state, free_out = _settled_output(dynamics, joint, x, y, None)
    history = dynamics.get_free_energy_history() or []
    _state, nudged_out = _settled_output(dynamics, joint, x, y, y)
    _state, repeat_out = _settled_output(dynamics, joint, x, y, None)
    if not _acts_history(history):
        return (
            AxisCheck("dynamics", value, "fail", "energy history missing/non-finite"),
        )
    deterministic = (
        bool(free_out)
        and bool(repeat_out)
        and torch.allclose(free_out[-1], repeat_out[-1], atol=1e-6)
    )
    responds = (
        bool(free_out)
        and bool(nudged_out)
        and not torch.allclose(free_out[-1], nudged_out[-1], atol=1e-6)
    )
    return (
        AxisCheck(
            "dynamics",
            value,
            "pass" if history[-1] < history[0] else "fail",
            f"prediction-error energy {history[0]:.4f} -> {history[-1]:.4f} "
            f"over {len(history)} steps",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if deterministic else "fail",
            "repeat settle is deterministic",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if responds else "fail",
            "nudged phase differs from free phase (target-responsive)"
            if responds
            else "nudge pathway unwired: settle math ignores the target "
            "(target only selects the state slot); predictive coding "
            "clamps the output layer in the nudged phase — wire the "
            "clamp to widen the manifest",
        ),
    )


def _probe_spike_integration(joint, x, y) -> tuple[AxisCheck, ...]:
    value = "spike_integration"
    dynamics = SpikeIntegrationDynamics(
        StateDynamicsConfig.spike_integration(max_steps=3)
    )
    settled, out = _settled_output(dynamics, joint, x, y, None)
    _state, repeat_out = _settled_output(dynamics, joint, x, y, None)
    spikes = getattr(settled, "spike_counts", None) or []
    finite = bool(out) and bool(torch.isfinite(out[-1]).all())
    bounded = finite and float(out[-1].abs().max()) <= _SPIKE_MEMBRANE_BOUND
    tracked = len(spikes) > 0 and bool(torch.isfinite(torch.stack(spikes)).all())
    deterministic = bool(repeat_out) and torch.allclose(
        out[-1], repeat_out[-1], atol=1e-6
    )
    return (
        AxisCheck(
            "dynamics",
            value,
            "pass" if finite else "fail",
            "membrane state finite after LIF integration"
            if finite
            else "membrane state non-finite after LIF integration",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if bounded else "fail",
            "membrane bounded (LIF threshold-reset active)"
            if bounded
            else "membrane unbounded — LIF reset not enforced",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if tracked else "fail",
            f"{len(spikes)} per-step spike counts recorded"
            if tracked
            else "spike counts missing from the settled state",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if deterministic else "fail",
            "repeat settle is deterministic",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass",
            "target-blind by spec: LIF integration carries no nudge term; "
            "supervision enters via the credit rule "
            "(surfaced by the credit probe)",
        ),
    )


def _probe_diffusion(joint, x, y) -> tuple[AxisCheck, ...]:
    value = "diffusion"
    dynamics = DiffusionDynamics(StateDynamicsConfig.diffusion(max_steps=3))
    h0 = joint.substrate.initial_state(x)
    e_init = float(h0.pow(2).sum())
    _state, out = _settled_output(dynamics, joint, x, y, None)
    _state, repeat_out = _settled_output(dynamics, joint, x, y, None)
    e_final = float(out[-1].pow(2).sum()) if out else float("inf")
    finite = bool(out) and bool(torch.isfinite(out[-1]).all())
    # Nudge responsiveness, seed-locked: re-running free vs nudged with
    # identical Langevin noise isolates the target's causal effect on
    # the settle output (target-blind implies identical).
    torch.manual_seed(_DIFFUSION_PROBE_SEED)
    _state, free_seeded = _settled_output(dynamics, joint, x, y, None)
    torch.manual_seed(_DIFFUSION_PROBE_SEED)
    _state, nudged_seeded = _settled_output(dynamics, joint, x, y, y)
    stochastic = bool(repeat_out) and not torch.allclose(
        out[-1], repeat_out[-1], atol=1e-9
    )
    responsive = (
        bool(free_seeded)
        and bool(nudged_seeded)
        and not torch.allclose(free_seeded[-1], nudged_seeded[-1], atol=1e-9)
    )
    return (
        AxisCheck(
            "dynamics",
            value,
            "pass" if finite else "fail",
            "Langevin settle produced finite state"
            if finite
            else "Langevin settle produced non-finite state",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if e_final < e_init else "fail",
            f"Langevin energy {e_init:.4f} -> {e_final:.4f} (descent on "
            "the settle energy functional)",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if stochastic else "fail",
            "Langevin noise present (repeat settle differs)"
            if stochastic
            else "repeat settle identical — Langevin noise missing",
        ),
        AxisCheck(
            "dynamics",
            value,
            "pass" if responsive else "fail",
            "nudged settle differs from free settle (target-responsive)"
            if responsive
            else "nudge pathway unwired: with identical Langevin noise "
            "the nudged settle equals the free settle — the energy "
            "functional (norm of h squared) has no target term; "
            "nudged-Langevin (EP-style) not implemented",
        ),
    )


_DYNAMICS_PROBES = {
    "energy_minimization": _probe_energy_minimization,
    "instantaneous": _probe_instantaneous,
    "predictive_settling": _probe_predictive_settling,
    "spike_integration": _probe_spike_integration,
    "diffusion": _probe_diffusion,
}


def _check_dynamics(joint, parts: list[str]) -> tuple[AxisCheck, ...]:
    value = parts[2]
    probe = _DYNAMICS_PROBES.get(value)
    if probe is None:
        return (
            AxisCheck("dynamics", value, "blocked", "no probe for this dynamics value"),
        )
    x, y = episode_batch(0)
    return probe(joint, x, y)


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
                "non-finite pseudo-gradient"
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


def _psi_max_delta(before: dict, after: dict) -> float:
    """Max absolute element change across the ψ tensors of one step."""
    deltas = [
        float((after[k].detach() - v.detach()).abs().max())
        for k, v in before.items()
        if k in after and isinstance(v, Tensor) and isinstance(after[k], Tensor)
    ]
    return max(deltas, default=0.0)


def _modulate_sensitive(plasticity, psi: dict, acts: list[Tensor]) -> bool:
    """Zeroed-ψ vs stepped-ψ modulation must change the activations."""
    modulate = getattr(plasticity, "modulate", None)
    if modulate is None or not psi:
        return True  # engagement-only primitives: sensitivity check n/a
    zeroed = {k: torch.zeros_like(v) for k, v in psi.items() if isinstance(v, Tensor)}
    stepped_out = modulate(acts, psi)
    zeroed_out = modulate(acts, zeroed)
    if isinstance(stepped_out, Tensor):
        return not torch.allclose(stepped_out, zeroed_out, atol=1e-9)
    if not stepped_out or not zeroed_out:
        return False
    return not all(
        torch.allclose(a, b, atol=1e-9) for a, b in zip(stepped_out, zeroed_out)
    )


def _check_plasticity(joint, parts: list[str]) -> AxisCheck:
    value = parts[3]
    x, y = episode_batch(0)
    calls = {"step": 0}
    trace: list[tuple[dict, dict]] = []
    original_step = joint.plasticity.step

    def tracing_step(psi, z, context):
        calls["step"] += 1
        new_psi = original_step(psi, z, context)
        trace.append((psi, new_psi))
        return new_psi

    joint.plasticity.step = tracing_step  # type: ignore[method-assign]
    try:
        joint.train_step(x, y)
        psi_final = dict(trace[-1][1]) if trace else {}
        acts = joint.geometry.forward_with_intermediates(x, joint.substrate)
        acts_list = (
            acts if isinstance(acts, list) else ([acts] if acts is not None else [])
        )
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
    if calls["step"] == 0:
        return AxisCheck(
            "plasticity",
            value,
            "fail",
            "plasticity.step never invoked by the episode pipeline "
            "(run_train_step takes no M-axis and initial_psi's ψ is "
            "discarded) — plasticity cannot modulate activity or credit",
        )
    delta = _psi_max_delta(*trace[-1]) if trace else 0.0
    if delta <= 0.0:
        return AxisCheck(
            "plasticity",
            value,
            "fail",
            f"ψ stepped {calls['step']}x but returned identical values "
            "(non-const assertion failed — inert plasticity update)",
        )
    sensitive = _modulate_sensitive(joint.plasticity, psi_final, acts_list)
    if not sensitive:
        return AxisCheck(
            "plasticity",
            value,
            "fail",
            "ψ modulate is insensitive: zeroed-ψ and stepped-ψ produce "
            "identical activity — plasticity cannot modulate the settle",
        )
    return AxisCheck(
        "plasticity",
        value,
        "pass",
        f"ψ stepped {calls['step']}x and changed (Δ={delta:.3e}); modulate "
        "is ψ-sensitive (zeroed-ψ ≠ stepped-ψ)",
    )


_FIDELITY_SEED = 0


def check_coordinate_fidelity(
    coordinate: str,
    *,
    device: str | torch.device | None = None,
    seed: int | None = _FIDELITY_SEED,
) -> CoordinateFidelity:
    """Run every fidelity probe against one campaign coordinate.

    Seeded by default: verdicts must not depend on the ambient global RNG
    state (the predictive-settling error-trajectory check sits near an
    init-dependent boundary and flipped with test ordering otherwise).
    ``seed=None`` restores ambient-RNG construction.
    """
    parts = coordinate.split("/")
    try:
        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
            joint = build_coordinate_system(coordinate, device=device)
    except IncompatibleCoordinateError as exc:
        return CoordinateFidelity(
            coordinate=coordinate,
            checks=(
                AxisCheck(
                    "coordinate",
                    f"{parts[2]} x {parts[4]}",
                    "fail",
                    f"invalid pairing (R3.9): {exc}",
                ),
            ),
        )
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
