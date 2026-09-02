"""R9.2 constraint-trial locks: the S-axis stress test's construct validity.

Locks pin the machinery, never the scientific outcome: the run-twice design
(unconstrained Digital baseline + analog-noise severity sweep), arm
coordinates that name the substrate they actually ran on (imp-48 identity),
the frozen control's declared coordinate/composition agreement, baseline
learnability (a walk too short for the slowest arm turns degradation curves
into noise), severity actually degrading the stream, and the pilot
preregistration's scope/label with per-environment embedded controls.
"""

import numpy as np
import pytest
import torch

from computronium.core.campaign.evaluation import (
    build_coordinate_system,
    evaluate_episode,
)
from computronium.experiments.joint.constraint_trial import (
    BASELINE_ENV,
    CONSTRAINT_CAMPAIGN_ID,
    TRIAL_ARMS,
    ConstraintConfig,
    _arm_coordinate,
    _collapse_severity,
    _compose,
    _environments,
    run_trial,
)

CHANCE = 0.125  # registered shape (8 inputs / 8 classes)
LOCK_CONFIG = ConstraintConfig(
    episodes=2, probe_episodes=1, seeds=(0, 1), severities=(0.5,)
)


def _walk_late(credit: str, env_name: str, episodes: int, lr: float = 0.03) -> float:
    """Direct single-walk helper: late-window training accuracy."""
    env = {e.name: e for e in _environments(ConstraintConfig())}[env_name]
    joint = _compose(credit, env, ConstraintConfig(lr=lr))
    accs = []
    for episode in range(episodes):
        record, _ = evaluate_episode(
            joint,
            coordinate=_arm_coordinate(credit, env),
            task_name="synthetic",
            campaign_id=CONSTRAINT_CAMPAIGN_ID,
            episode=episode,
            batch_size=16,
            guard_threshold=None,
            seed=0,
            stationary_teacher=True,
        )
        accs.append(record.task_accuracy)
    return float(np.mean(accs[-10:]))


class TestConstraintEnvironments:
    def test_baseline_is_unconstrained_digital(self) -> None:
        envs = _environments(ConstraintConfig())
        assert envs[0].name == BASELINE_ENV
        assert envs[0].unconstrained
        assert envs[0].substrate_axis == "digital"
        assert all(not env.unconstrained for env in envs[1:])
        assert [env.severity for env in envs[1:]] == [0.0, 0.5, 1.0]

    def test_coordinates_name_the_substrate_they_run_on(self) -> None:
        for env in _environments(ConstraintConfig()):
            for credit in TRIAL_ARMS:
                assert _arm_coordinate(credit, env).split("/")[0] == env.substrate_axis

    def test_coordinates_are_dispatchable_ontology_coordinates(self) -> None:
        env = _environments(ConstraintConfig())[0]
        for credit in TRIAL_ARMS:
            system = build_coordinate_system(_arm_coordinate(credit, env))
            assert system is not None

    def test_control_coordinate_declares_frozen_and_agrees(self) -> None:
        env = _environments(ConstraintConfig())[0]
        coordinate = _arm_coordinate("gradient", env, frozen=True)
        assert coordinate.endswith("/gradient/frozen")
        joint = _compose("gradient", env, ConstraintConfig(lr=0.03), frozen=True)
        update = joint.update.config
        assert float(getattr(update, "step_size", 1.0)) == pytest.approx(0.0)

    def test_frozen_control_never_trains(self) -> None:
        env = _environments(ConstraintConfig())[0]
        joint = _compose("gradient", env, ConstraintConfig(lr=0.03), frozen=True)
        before = {k: v.clone() for k, v in joint.geometry.params.items()}
        for episode in range(2):
            evaluate_episode(
                joint,
                coordinate=_arm_coordinate("gradient", env, frozen=True),
                task_name="synthetic",
                campaign_id=CONSTRAINT_CAMPAIGN_ID,
                episode=episode,
                batch_size=16,
                guard_threshold=None,
                seed=0,
                stationary_teacher=True,
            )
        for name, param in joint.geometry.params.items():
            assert torch.equal(param, before[name])


class TestConstraintPilot:
    def test_pilot_labels_itself_pilot(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.label() == "pilot"
        assert result.preregistration.claim_scope == "resource_efficiency"
        assert result.preregistration.task_stream == "stationary"
        assert result.preregistration.embedded_control is not None
        assert set(result.arms) == {
            "gradient",
            "random_projections",
            "thermodynamic_contrast",
            "control",
        }

    def test_per_env_controls_all_pass(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert set(result.control_verdicts) == {BASELINE_ENV, "analog_0.5"}
        assert all(v["verdict"] == "passed" for v in result.control_verdicts.values())
        assert not result.quarantined

    def test_contrasts_cover_constrained_envs(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert set(result.contrasts_vs_gradient) == {
            "random_projections@digital",
            "random_projections@analog_0.5",
            "thermodynamic_contrast@digital",
            "thermodynamic_contrast@analog_0.5",
        }


class TestTrialIsLearnableAtBaseline:
    """Construct validity: the arms must clear chance at the unconstrained
    baseline, else the degradation curves measure noise (imp-36 class)."""

    def test_gradient_reaches_above_chance_at_baseline(self) -> None:
        assert _walk_late("gradient", BASELINE_ENV, episodes=60) > CHANCE + 0.1

    def test_severity_degrades_the_stream(self) -> None:
        baseline = _walk_late("gradient", BASELINE_ENV, episodes=60)
        severe = _walk_late("gradient", "analog_1", episodes=60)
        assert severe < baseline


class TestCollapseBoundary:
    def test_boundary_takes_max_severity_above_margin(self) -> None:
        probes = {
            "digital": (0.7,),
            "analog_0.0": (0.5,),
            "analog_0.5": (0.25,),
            "analog_1": (0.1,),
        }
        assert _collapse_severity(probes, chance=CHANCE) == pytest.approx(0.5)
        collapsed = {"digital": (0.1,), "analog_0.0": (0.1,)}
        assert _collapse_severity(collapsed, chance=CHANCE) is None
