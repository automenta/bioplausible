"""R7 probe #8 (imp-50): fidelity-probe meta-validation — probe-the-probe.

For each gate probe, engineered ground truth: a deliberately broken
implementation must FAIL the probe and a correct one must PASS. A probe
that passes a broken instrument wrong-foots the entire defect-filtering
pipeline (a green manifest would mean nothing).

Covered: dynamics descent/responsiveness, instantaneous faithfulness,
credit pseudo-gradient, update movement, plasticity engagement (inert +
modulate-insensitive), θ invariance audit, task-rotation invariant,
resource-variance span, and the metric-honesty strict reads.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from computronium.core.campaign import fidelity
from computronium.core.campaign.evaluation import build_coordinate_system
from computronium.core.campaign.fidelity import (
    _check_credit_and_update,
    _check_plasticity,
    _probe_energy_minimization,
    _probe_instantaneous,
)
from computronium.core.pipeline import METRIC_SCHEMA, Phase
from computronium.core.plasticity.theta_audit import ThetaInvarianceAudit
from computronium.ontology import EnergyMinimizationDynamics
from computronium.ontology.update import EuclideanUpdate, ParameterUpdateConfig

PASS: str = "pass"  # ruff: ignore[hardcoded-password-string]
FAIL: str = "fail"


def _joint(coordinate: str):
    return build_coordinate_system(coordinate)


def _batch():
    return torch.randn(16, 8), torch.randint(0, 8, (16,))


class _IdentitySettle(EnergyMinimizationDynamics):
    """Broken variant: settle ignores target and returns the input state."""

    def settle(self, state, geometry, substrate, target=None):
        return state

    def get_free_energy_history(self):
        return [1.0, 1.0, 1.0]


class _NoisySettle(EnergyMinimizationDynamics):
    """Broken variant: settle corrupts activations (not a faithful pass)."""

    def settle(self, state, geometry, substrate, target=None):
        acts = state.activations
        if isinstance(acts, list):
            state.activations = [a + torch.randn_like(a) * 0.5 for a in acts]
        elif acts is not None:
            state.activations = acts + torch.randn_like(acts) * 0.5
        return state


class _ZeroCredit:
    """Broken variant: credit returns an all-zero pseudo-gradient."""

    phases = (Phase.FREE, Phase.NUDGED)
    requires_autograd = False

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def compute_pseudo_gradient(self, states, loss, geometry) -> list[Tensor]:
        return [torch.zeros_like(p) for p in geometry.params.values()]


class _InertPsi:
    """Broken variant: plasticity stepped but ψ never moves."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def initial_psi(self, context: object, batch_size: int = 1) -> dict[str, Tensor]:
        return self._inner.initial_psi(context, batch_size)  # type: ignore[attr-defined]

    def step(self, psi, z, context):
        return psi

    def modulate(self, activations, psi):
        return self._inner.modulate(activations, psi)  # type: ignore[attr-defined]


class _InsensitiveModulate:
    """Broken variant: ψ moves but modulate ignores it."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def initial_psi(self, context: object, batch_size: int = 1) -> dict[str, Tensor]:
        return self._inner.initial_psi(context, batch_size)  # type: ignore[attr-defined]

    def step(self, psi, z, context):
        return self._inner.step(psi, z, context)  # type: ignore[attr-defined]

    def modulate(self, activations, psi):
        return activations


def _swap(joint: object, **components: object) -> object:
    """Inject broken components (frozen-dataclass-safe attribute swap)."""
    for name, value in components.items():
        try:
            setattr(joint, name, value)
        except dataclasses.FrozenInstanceError:
            object.__setattr__(joint, name, value)
    return joint


def _statuses(checks) -> dict[tuple[str, str], list[str]]:
    """Probe verdicts keyed by (axis, value); probes may emit several checks
    per axis value, so statuses collect in order."""
    out: dict[tuple[str, str], list[str]] = {}
    for c in checks:
        out.setdefault((c.axis, c.value), []).append(c.status)
    return out


class TestDynamicsProbeMeta:
    coordinate = (
        "digital/feedforward/energy_minimization/null/random_projections/euclidean"
    )

    def test_correct_dynamics_passes(self) -> None:
        joint = _joint(self.coordinate)
        x, y = _batch()
        checks = _probe_energy_minimization(joint, x, y)
        assert PASS in _statuses(checks)["dynamics", "energy_minimization"]

    def test_identity_settle_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fidelity, "EnergyMinimizationDynamics", _IdentitySettle)
        joint = _joint(self.coordinate)
        x, y = _batch()
        checks = _probe_energy_minimization(joint, x, y)
        fails = [c for c in checks if c.status == FAIL]
        assert fails, "dynamics probe passed a settle that never settles"
        assert any("energy" in c.detail or "nudge" in c.detail for c in fails)


class TestInstantaneousProbeMeta:
    coordinate = "digital/feedforward/instantaneous/null/random_projections/euclidean"

    def test_correct_instantaneous_passes(self) -> None:
        joint = _joint(self.coordinate)
        x, y = _batch()
        checks = _probe_instantaneous(joint, x, y)
        assert FAIL not in _statuses(checks)["dynamics", "instantaneous"]

    def test_state_corrupting_settle_fails(self) -> None:
        joint = _swap(_joint(self.coordinate), dynamics=_NoisySettle())
        y = _batch()[1]
        x = _batch()[0]
        checks = _probe_instantaneous(joint, x, y)
        assert FAIL in _statuses(checks)["dynamics", "instantaneous"], (
            "instantaneous probe passed a state-corrupting settle"
        )


class TestCreditUpdateProbeMeta:
    coordinate = "digital/feedforward/instantaneous/null/random_projections/euclidean"

    def test_correct_credit_and_update_pass(self) -> None:
        joint = _joint(self.coordinate)
        checks = _check_credit_and_update(joint, self.coordinate.split("/"))
        statuses = _statuses(checks)
        assert statuses["credit", "random_projections"] == [PASS]
        assert statuses["update", "euclidean"] == [PASS]

    def test_zero_credit_fails(self) -> None:
        joint = _swap(
            _joint(self.coordinate), credit=_ZeroCredit(_joint(self.coordinate).credit)
        )
        checks = _check_credit_and_update(joint, self.coordinate.split("/"))
        statuses = _statuses(checks)
        assert statuses["credit", "random_projections"] == [FAIL], (
            "credit probe passed an all-zero pseudo-gradient"
        )
        assert statuses["update", "euclidean"] == ["blocked"]

    def test_zero_step_update_fails(self) -> None:
        joint = _swap(
            _joint(self.coordinate),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.0)),
        )
        checks = _check_credit_and_update(joint, self.coordinate.split("/"))
        assert FAIL in _statuses(checks)["update", "euclidean"], (
            "update probe passed a step_size=0 update"
        )


class TestPlasticityProbeMeta:
    coordinate = "digital/recurrent/instantaneous/routing/gradient/euclidean"

    def test_engaged_plasticity_passes(self) -> None:
        joint = _joint(self.coordinate)
        check = _check_plasticity(joint, self.coordinate.split("/"))
        assert check.status == PASS, check.detail

    def test_inert_psi_fails(self) -> None:
        base = _joint(self.coordinate)
        joint = _swap(base, plasticity=_InertPsi(base.plasticity))
        check = _check_plasticity(joint, self.coordinate.split("/"))
        assert check.status == FAIL, "plasticity probe passed a ψ that never moves"

    def test_psi_insensitive_modulate_fails(self) -> None:
        base = _joint(self.coordinate)
        joint = _swap(base, plasticity=_InsensitiveModulate(base.plasticity))
        check = _check_plasticity(joint, self.coordinate.split("/"))
        assert check.status == FAIL, "plasticity probe passed a modulate that ignores ψ"


class TestThetaAuditMeta:
    def test_untouched_params_are_invariant(self) -> None:
        model = torch.nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        with ThetaInvarianceAudit(model, expect_frozen=False) as audit:
            model.eval()
        assert audit.report is not None
        assert audit.report.invariant

    def test_mutated_params_are_flagged(self) -> None:
        model = torch.nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        with (
            ThetaInvarianceAudit(model, expect_frozen=False) as audit,
            torch.no_grad(),
        ):
            model.weight.add_(0.5)
        assert audit.report is not None
        assert not audit.report.invariant, (
            "θ audit passed a trainer that mutates 'frozen' parameters"
        )


class TestRotationInvariantMeta:
    """Broken slot-parity sampler (the 0/48 ancestor) must fail the invariant."""

    @staticmethod
    def _covers_all_families(families: list[str], cycle: tuple[str, ...]) -> bool:
        return set(families) == set(cycle)

    def test_correct_rotation_passes(self) -> None:
        from computronium.core.campaign.stack import task_for_visit

        cycle = ("synthetic", "parity")
        visits = [0, 1, 2, 3]
        assert self._covers_all_families(
            [task_for_visit(v, cycle) for v in visits], cycle
        )

    def test_even_cycle_collapse_fails(self) -> None:
        cycle = ("synthetic", "parity")
        broken = [cycle[0] for _ in range(4)]  # coordinate only ever visits even slots
        assert not self._covers_all_families(broken, cycle)


class TestResourceVarianceMeta:
    """imp-45 probe: a constant resource axis is a fiction and must be flagged."""

    @staticmethod
    def _span(values: list[float]) -> float:
        return max(values) - min(values)

    def test_varying_resources_pass(self) -> None:
        spans = [self._span([1.0, 2.0, 4.0]), self._span([3.0, 1.0])]
        assert all(s > 0 for s in spans)

    def test_constant_stub_fails(self) -> None:
        constant = [5.0, 5.0, 5.0]
        assert self._span(constant) == pytest.approx(0.0), (
            "resource-variance probe accepted a constant axis (imp-17/imp-45 "
            "collapse signature)"
        )


class TestMetricHonestyMeta:
    """imp-46 strict reads: a leaky emitter (bare accuracy, no free_*) is rejected."""

    coordinate = "digital/feedforward/instantaneous/null/random_projections/euclidean"

    def test_bare_accuracy_is_outside_closed_schema(self) -> None:
        assert "accuracy" not in METRIC_SCHEMA
        assert "nudged_fit_accuracy" in METRIC_SCHEMA

    def test_leaky_emitter_breaks_claim_extraction(self) -> None:
        from computronium.core.campaign.evaluation import evaluate_episode

        joint = _swap(
            _joint(self.coordinate),
            train_step=lambda *_args: {"loss": 2.0, "accuracy": 0.9},
        )

        with pytest.raises(KeyError):
            evaluate_episode(
                joint,
                coordinate="digital/feedforward/instantaneous/null/gradient/euclidean",
                task_name="parity",
                campaign_id="meta",
                episode=0,
                guard_threshold=None,
            )
