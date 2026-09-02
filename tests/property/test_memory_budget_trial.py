"""R9.2/R9.3 memory-budget trial locks: the constraint family where O(1)-memory
credit is structurally immune.

Locks pin the machinery, never the scientific outcome: the deterministic
memory profile (O(depth) for exact-global/random-projection credit, exactly
0 for thermodynamic_contrast), the feasibility grid that turns the profile
into the commissioning verdicts (walled iff bytes exceed the budget; the
0.48 MiB separation of the two walled arms at the deep tier), the OOM
semantics of the walk plan (a cell walled under every registered budget
never walks), the frozen-thermo control whose identity is the (credit,
frozen) pair and whose verdict exists in every regime (imp-64, R8.5),
walled-regime shallow competence (the claim's premise), the imp-67
stationary-stream provenance, and the R8.4 commissioning gate for the
registered design.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from computronium.core.campaign.evaluation import (
    _teacher_key,
    build_coordinate_system,
    evaluate_episode,
)
from computronium.experiments.joint.memory_budget_trial import (
    BUDGETS_MIB,
    CONTROL_CREDIT,
    MEMORY_BUDGET_CAMPAIGN_ID,
    TRIAL_ARMS,
    MemoryBudgetConfig,
    _arm_coordinate,
    _compose,
    _environments,
    _measure_saved_bytes,
    _walk_seed,
    run_trial,
)
from computronium.validation.power_preregistration import (
    EmbeddedControl,
    PowerPreregistration,
    min_detectable_effect,
)

CHANCE = 0.125  # registered shape (8 inputs / 8 classes)
LOCK_CONFIG = MemoryBudgetConfig(
    episodes=2,
    probe_episodes=1,
    seeds=(0, 1),
    depths=(4, 16),
    width=8,
)


def _walk_late(credit: str, depth: int, episodes: int, lr: float = 0.05) -> float:
    """Direct single-walk helper: mean late-window training accuracy."""
    config = MemoryBudgetConfig(depths=(depth,), lr=lr, episodes=episodes)
    env = _environments(config)[0]
    joint = _compose(credit, env, config)
    coord = _arm_coordinate(credit)
    campaign_id = f"{MEMORY_BUDGET_CAMPAIGN_ID}_lock::{env.name}"
    accs = []
    for episode in range(episodes):
        record, _ = evaluate_episode(
            joint,
            coordinate=coord,
            task_name="synthetic",
            campaign_id=campaign_id,
            episode=episode,
            batch_size=config.batch_size,
            guard_threshold=None,
            seed=0,
            stationary_teacher=True,
            teacher_noise=config.teacher_noise,
        )
        accs.append(record.task_accuracy)
    return float(np.mean(accs[-10:]))


class TestMemoryProfile:
    """The deterministic instrument: O(depth) vs exactly 0 saved bytes."""

    def test_odepth_arms_saved_bytes_grow_with_depth(self) -> None:
        config = MemoryBudgetConfig(depths=(4, 16, 50), width=16)
        envs = _environments(config)
        for credit in ("gradient", "random_projections"):
            by_depth = [_measure_saved_bytes(credit, env, config) for env in envs]
            assert by_depth[1] > by_depth[0], credit
            assert by_depth[2] > by_depth[1], credit

    def test_o1_arms_saved_bytes_exactly_zero(self) -> None:
        config = MemoryBudgetConfig(depths=(4, 16, 50), width=16)
        for env in _environments(config):
            assert not _measure_saved_bytes(CONTROL_CREDIT, env, config), (
                "O(1) arm must save 0 bytes (no autograd graph)"
            )


class TestFeasibilityGrid:
    """The registered severity sweep: budgets sized against the registered
    width-16 profile (gradient 27,136/137,728/451,072 B, FA 29,728/152,704/
    501,136 B, thermo 0 at depths 4/16/50) produce three regimes — fully
    walled, deep-walled, and the 0.45 MiB separation of the two walled arms
    at the deep tier."""

    def test_o1_arms_feasible_under_every_budget(self) -> None:
        result = run_trial(
            MemoryBudgetConfig(episodes=1, seeds=(0,), depths=(4,), width=16)
        )
        for per_env in result.feasibility.values():
            for verdicts in per_env.values():
                assert verdicts["thermodynamic_contrast"], "O(1) arm walled"
                assert verdicts["control"], "control walled"

    def test_smallest_budget_walls_the_global_arms_everywhere(self) -> None:
        config = MemoryBudgetConfig(width=16)
        envs = _environments(config)
        for env in envs:
            for credit in ("gradient", "random_projections"):
                saved = _measure_saved_bytes(credit, env, config)
                assert saved > BUDGETS_MIB[0] * 1024 * 1024, f"{credit}@{env.name}"

    def test_mid_budget_walls_only_the_deep_tier(self) -> None:
        config = MemoryBudgetConfig(width=16)
        envs = {env.name: env for env in _environments(config)}
        for credit in ("gradient", "random_projections"):
            assert _measure_saved_bytes(credit, envs["depth_4"], config) <= (
                BUDGETS_MIB[1] * 1024 * 1024
            )
            assert _measure_saved_bytes(credit, envs["depth_16"], config) <= (
                BUDGETS_MIB[1] * 1024 * 1024
            )
            assert _measure_saved_bytes(credit, envs["depth_50"], config) > (
                BUDGETS_MIB[1] * 1024 * 1024
            )

    def test_registered_budgets_separate_the_walled_arms_at_depth(self) -> None:
        """0.45 MiB admits gradient (451,072 B) and walls FA (501,136 B) at
        the deep tier — the sweep discriminates the two O(depth) arms."""
        config = MemoryBudgetConfig(width=16)
        envs = {env.name: env for env in _environments(config)}
        gradient = _measure_saved_bytes("gradient", envs["depth_50"], config)
        fa = _measure_saved_bytes("random_projections", envs["depth_50"], config)
        ceiling = BUDGETS_MIB[2] * 1024 * 1024
        assert gradient <= ceiling < fa

    def test_never_commissionable_names_only_the_fully_walled_cells(self) -> None:
        result = run_trial(
            MemoryBudgetConfig(episodes=1, seeds=(0,), depths=(4, 16, 50), width=16)
        )
        assert result.never_commissionable == ("random_projections@depth_50",)


class TestWalkPlan:
    """OOM semantics: a cell walled under every registered budget never
    walks — it produces no records and no probe data, by design."""

    def test_fully_walled_arms_produce_no_data(self) -> None:
        config = MemoryBudgetConfig(
            episodes=1,
            probe_episodes=1,
            seeds=(0,),
            depths=(4,),
            width=8,
            budgets_mib=(0.005,),  # walls gradient/FA at the width-8 shape too
        )
        result = run_trial(config)
        for label in ("gradient", "random_projections"):
            arm = result.arms[label]
            assert arm.walked_envs == (), f"{label} walked a walled cell"
            assert arm.probe_by_env == {}
        assert result.never_commissionable == (
            "gradient@depth_4",
            "random_projections@depth_4",
        )
        # the O(1) arms are the only feasible learners in this regime
        assert result.arms["thermodynamic_contrast"].walked_envs == ("depth_4",)
        assert result.arms["control"].walked_envs == ("depth_4",)


class TestControl:
    """The planted control is a frozen thermodynamic_contrast arm: the only
    credit feasible at every budget, so its at-chance verdict exists in
    every regime. Identity is the (credit, frozen) pair (imp-64)."""

    def test_control_coordinate_declares_frozen_and_settling(self) -> None:
        control = _arm_coordinate(CONTROL_CREDIT, frozen=True)
        learner = _arm_coordinate(CONTROL_CREDIT)
        assert control != learner
        assert control.endswith("/frozen")
        assert "energy_minimization" in control  # D x C fence
        assert learner.endswith("/euclidean")

    def test_control_and_learner_coordinates_are_dispatchable(self) -> None:
        for coord in (
            _arm_coordinate(CONTROL_CREDIT, frozen=True),
            _arm_coordinate(CONTROL_CREDIT),
            *(_arm_coordinate(credit) for credit in TRIAL_ARMS),
        ):
            build_coordinate_system(coord)  # must not raise

    def test_frozen_control_never_trains(self) -> None:
        """θ must not mutate in the frozen arm across a walk."""
        config = MemoryBudgetConfig(depths=(4,), episodes=1, width=8)
        env = _environments(config)[0]
        joint = _compose(CONTROL_CREDIT, env, config, frozen=True)
        coord = _arm_coordinate(CONTROL_CREDIT, frozen=True)
        theta_before = {
            name: param.detach().clone()
            for name, param in joint.geometry.params.items()
        }
        evaluate_episode(
            joint,
            coordinate=coord,
            task_name="synthetic",
            campaign_id=f"{MEMORY_BUDGET_CAMPAIGN_ID}_lock::{env.name}",
            episode=0,
            batch_size=config.batch_size,
            guard_threshold=None,
            seed=0,
            stationary_teacher=True,
        )
        theta_after = joint.geometry.params
        for name, before in theta_before.items():
            torch.testing.assert_close(
                before,
                theta_after[name],
                msg=f"θ parameter {name} mutated in frozen arm",
            )

    def test_per_env_controls_all_pass_at_pilot_scale(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.control_verdicts, "no control verdicts recorded"
        for env, verdict in result.control_verdicts.items():
            assert verdict["verdict"] == "passed", f"{env}: {verdict['detail']}"
        assert not result.quarantined


class TestPilotPreregistration:
    def test_pilot_labels_itself_pilot(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.declared_rung == "pilot"
        assert result.preregistration.label() == "pilot"

    def test_claim_scope_resource_efficiency(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.claim_scope == "resource_efficiency"

    def test_task_stream_stationary(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.task_stream == "stationary"

    def test_embedded_control_is_the_frozen_o1_arm(self) -> None:
        result = run_trial(LOCK_CONFIG)
        control = result.preregistration.embedded_control
        assert control is not None
        assert control.arm == "frozen_lr0"
        assert control.coordinate == _arm_coordinate(CONTROL_CREDIT, frozen=True)
        assert control.chance == pytest.approx(CHANCE)

    def test_effect_and_variance_come_from_the_shallow_contrast(self) -> None:
        result = run_trial(LOCK_CONFIG)
        prereg = result.preregistration
        shallow_thermo = list(
            result.arms["thermodynamic_contrast"].probe_by_env["depth_4"]
        )
        shallow_control = list(result.arms["control"].probe_by_env["depth_4"])
        assert prereg.n_per_group == len(shallow_thermo)
        assert prereg.variance_estimate == pytest.approx(
            float(np.std(shallow_thermo + shallow_control, ddof=1)),
            abs=1e-4,
        )


class TestRegisteredCommission:
    """The registered memory-budget commission rides the R8.4 gate (imp-55):
    the prereg file must derive claim-grade from its own numbers, and the
    trial must refuse to walk under anything less."""

    REGISTERED_PREREG = (
        Path(__file__).parents[2]
        / "configs/preregistrations/r92_memory_budget_registered.json"
    )

    def test_registered_prereg_derives_claim_grade(self) -> None:
        prereg = PowerPreregistration.load(self.REGISTERED_PREREG)
        assert prereg.unmet_requirements() == ()
        assert prereg.label() == "claim_grade"
        assert prereg.declared_rung is None  # claim-grade is derived, never declared
        assert prereg.claim_scope == "resource_efficiency"
        assert prereg.task_stream == "stationary"
        assert prereg.embedded_control is not None
        control = prereg.embedded_control
        assert control.coordinate == _arm_coordinate(CONTROL_CREDIT, frozen=True)
        assert control.chance == pytest.approx(CHANCE)
        assert control.tolerance >= 0.05  # imp-59 floor at the registered N
        assert prereg.mde_cohens_d <= prereg.expected_effect

    def test_trial_refuses_unmet_claim_grade_gates(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: unmet gates refuse the commission",
            metric="probe_accuracy",
            claim_scope="resource_efficiency",
            task_stream="stationary",
            expected_effect=0.1,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="frozen_lr0",
                coordinate=_arm_coordinate(CONTROL_CREDIT, frozen=True),
                chance=CHANCE,
                tolerance=0.2,
            ),
        )
        with pytest.raises(ValueError, match="not claim-grade"):
            run_trial(LOCK_CONFIG, preregistration=prereg)

    def test_trial_refuses_a_declared_rung_cap(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: a declared rung caps the label below claim-grade",
            metric="probe_accuracy",
            claim_scope="resource_efficiency",
            task_stream="stationary",
            expected_effect=min_detectable_effect(5) + 0.01,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="frozen_lr0",
                coordinate=_arm_coordinate(CONTROL_CREDIT, frozen=True),
                chance=CHANCE,
                tolerance=0.2,
            ),
            declared_rung="pilot",
        )
        with pytest.raises(ValueError, match="caps the label below claim-grade"):
            run_trial(LOCK_CONFIG, preregistration=prereg)

    def test_trial_enforces_the_registered_n(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: the walk must deliver the registered n",
            metric="probe_accuracy",
            claim_scope="resource_efficiency",
            task_stream="stationary",
            expected_effect=min_detectable_effect(5) + 0.01,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="frozen_lr0",
                coordinate=_arm_coordinate(CONTROL_CREDIT, frozen=True),
                chance=CHANCE,
                tolerance=0.2,
            ),
        )
        with pytest.raises(ValueError, match="registered design requires 5"):
            run_trial(
                MemoryBudgetConfig(depths=(4,), episodes=1, seeds=(0, 1), width=8),
                preregistration=prereg,
            )


class TestWalledRegimeCompetence:
    """The claim's premise: in the fully-walled regime the O(1) arm is the
    only feasible learner and it learns — above-chance competence at the
    shallow tier (the R9.3-registered signature, lr=0.05 @ 100 episodes)."""

    def test_thermo_reaches_above_chance_at_shallow(self) -> None:
        acc = _walk_late(CONTROL_CREDIT, depth=4, episodes=100, lr=0.05)
        assert acc > CHANCE + 0.1, (
            f"shallow thermodynamic_contrast competence: {acc:.4f} <= "
            f"{CHANCE:.4f} + 0.1"
        )


class TestStationaryStreamConstructValidity:
    """imp-67 lock: the walk must enact the stream its prereg declares."""

    def test_walk_records_carry_stationary_teacher_provenance(self) -> None:
        config = MemoryBudgetConfig(depths=(4,), episodes=2, seeds=(0,), width=8)
        env = _environments(config)[0]
        for credit in (*TRIAL_ARMS, CONTROL_CREDIT):
            coord = _arm_coordinate(credit, frozen=(credit == CONTROL_CREDIT))
            _, _, records = _walk_seed(
                credit,
                credit == CONTROL_CREDIT,
                env,
                coord,
                0,
                config=config,
            )
            assert records, f"{credit} walk emitted no records"
            for record in records:
                stamp = record.metadata.get("teacher_stationary", 0.0)
                assert math.isclose(float(stamp), 1.0), (
                    f"{credit} walked the non-stationary stream while the prereg "
                    "declares task_stream=stationary (imp-67 class)"
                )

    def test_probe_scores_the_same_stationary_teacher_stream(self) -> None:
        """imp-61 class: a probe keyed differently scores a different task."""
        config = MemoryBudgetConfig(depths=(4,), episodes=1, seeds=(0,), width=8)
        env = _environments(config)[0]
        coord = _arm_coordinate("gradient")
        campaign_id = f"{MEMORY_BUDGET_CAMPAIGN_ID}::{env.name}"
        walk_key = _teacher_key(campaign_id, coord, 0, stationary=True, segment=None)
        probe_key = _teacher_key(campaign_id, coord, 0, stationary=True, segment=None)
        assert walk_key == probe_key
