"""R9.3 deep credit trial locks: the C-axis temporal-dependency stress test.

Locks pin the machinery, never the scientific outcome: the depth-sweep design
(competence baseline + >=50 credit-step tier), arm coordinates that name the
dynamics they actually run on (DxC fence: thermo requires settling; gradient/FA
use instantaneous), the frozen control's declared coordinate/composition agreement,
shallow-depth competence (a walk too short for the slowest arm turns degradation
curves into noise), depth actually degrading the exact-global arm's memory
profile, and the pilot preregistration's scope/label with per-depth embedded
controls.
"""

import numpy as np
import torch

from computronium.core.campaign.evaluation import (
    build_coordinate_system,
    evaluate_episode,
)
from computronium.experiments.joint.deep_credit_trial import (
    CONTROL_CREDIT,
    DEEP_CREDIT_CAMPAIGN_ID,
    TRIAL_ARMS,
    DeepCreditConfig,
    _arm_coordinate,
    _compose,
    _environments,
    run_trial,
)

CHANCE = 0.125  # registered shape (8 inputs / 8 classes)
LOCK_CONFIG = DeepCreditConfig(
    episodes=2,
    probe_episodes=1,
    seeds=(0, 1),
    depths=(4, 16),
    width=8,
)


def _walk_late(credit: str, depth: int, episodes: int, lr: float = 0.03) -> float:
    """Direct single-walk helper: late-window training accuracy."""
    config = DeepCreditConfig(depths=(depth,), lr=lr)
    env = {e.name: e for e in _environments(config)}[f"depth_{depth}"]
    joint = _compose(credit, env, config)
    accs = []
    for episode in range(episodes):
        record, _ = evaluate_episode(
            joint,
            coordinate=_arm_coordinate(credit, env),
            task_name="synthetic",
            campaign_id=DEEP_CREDIT_CAMPAIGN_ID,
            episode=episode,
            batch_size=16,
            guard_threshold=None,
            seed=0,
            stationary_teacher=True,
        )
        accs.append(record.task_accuracy)
    return float(np.mean(accs[-10:]))


class TestDepthEnvironments:
    def test_depths_include_deep_tier(self) -> None:
        envs = _environments(DeepCreditConfig())
        depths = [env.depth for env in envs]
        assert 50 in depths, "registered deep tier (50+) must be present"
        assert depths[0] < depths[-1], "shallow competence tier first"

    def test_hidden_dims_match_depth(self) -> None:
        envs = _environments(DeepCreditConfig(width=16))
        for env in envs:
            assert len(env.hidden_dims) == env.depth - 1
            assert all(h == 16 for h in env.hidden_dims)

    def test_coordinates_name_the_dynamics_they_run_on(self) -> None:
        envs = _environments(LOCK_CONFIG)
        for env in envs:
            for credit in TRIAL_ARMS:
                coord = _arm_coordinate(credit, env)
                dynamics = coord.split("/")[2]
                if credit == "thermodynamic_contrast":
                    assert dynamics == "energy_minimization", (
                        f"thermo must use energy_minimization dynamics, got {dynamics}"
                    )
                else:
                    assert dynamics == "instantaneous", (
                        f"{credit} must use instantaneous dynamics, got {dynamics}"
                    )

    def test_coordinates_are_dispatchable_ontology_coordinates(self) -> None:
        env = _environments(LOCK_CONFIG)[0]
        for credit in TRIAL_ARMS:
            system = build_coordinate_system(_arm_coordinate(credit, env))
            assert system is not None

    def test_control_coordinate_declares_frozen_and_agrees(self) -> None:
        env = _environments(LOCK_CONFIG)[0]
        coord = _arm_coordinate(CONTROL_CREDIT, env, frozen=True)
        parts = coord.split("/")
        assert parts[5] == "frozen", "control must declare frozen update"
        # Same credit as gradient (imp-64: role is frozen flag, not credit name)
        assert parts[4] == "gradient"

    def test_frozen_control_never_trains(self) -> None:
        """θ must not mutate in the frozen arm across a walk."""
        env = _environments(LOCK_CONFIG)[0]
        joint = _compose(CONTROL_CREDIT, env, LOCK_CONFIG, frozen=True)
        coord = _arm_coordinate(CONTROL_CREDIT, env, frozen=True)
        theta_before = {
            name: param.detach().clone()
            for name, param in joint.geometry.params.items()
        }
        for episode in range(5):
            evaluate_episode(
                joint,
                coordinate=coord,
                task_name="synthetic",
                campaign_id=f"{DEEP_CREDIT_CAMPAIGN_ID}::{env.name}",
                episode=episode,
                batch_size=16,
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


class TestMemoryProfile:
    def test_gradient_saved_bytes_grows_with_depth(self) -> None:
        from computronium.core.campaign.evaluation import episode_batch
        from computronium.core.profiling import measure_saved_activation_bytes

        saved_by_depth = {}
        for depth in [4, 16]:
            config = DeepCreditConfig(depths=(depth,), width=8)
            env = _environments(config)[0]
            joint = _compose("gradient", env, config)
            x_batch, y_batch = episode_batch(
                episode=0,
                task_name="synthetic",
                batch_size=16,
                input_dim=8,
                num_classes=8,
                teacher_key=(1,),
                teacher_noise=0.5,
            )
            x_batch, y_batch = x_batch.to(joint.device), y_batch.to(joint.device)
            _, saved = measure_saved_activation_bytes(
                joint.train_step, x_batch, y_batch
            )
            saved_by_depth[depth] = saved.total_bytes

        assert saved_by_depth[16] > saved_by_depth[4] * 2, (
            "gradient saved bytes should grow with depth: "
            f"{saved_by_depth[4]} -> {saved_by_depth[16]}"
        )

    def test_thermo_saved_bytes_flat_across_depth(self) -> None:
        from computronium.core.campaign.evaluation import episode_batch
        from computronium.core.profiling import measure_saved_activation_bytes

        for depth in [4, 16]:
            config = DeepCreditConfig(depths=(depth,), width=8)
            env = _environments(config)[0]
            joint = _compose("thermodynamic_contrast", env, config)
            x_batch, y_batch = episode_batch(
                episode=0,
                task_name="synthetic",
                batch_size=16,
                input_dim=8,
                num_classes=8,
                teacher_key=(1,),
                teacher_noise=0.5,
            )
            x_batch, y_batch = x_batch.to(joint.device), y_batch.to(joint.device)
            _, saved = measure_saved_activation_bytes(
                joint.train_step, x_batch, y_batch
            )
            assert saved.total_bytes == 0, (
                f"thermo must have 0 saved bytes (no autograd), got {saved.total_bytes}"
            )


class TestImp60Regression:
    def test_nonsquare_geometry_no_crash_on_guard_probe(self) -> None:
        """imp-60: evaluate_episode with input_dim != num_classes must not crash."""
        config = DeepCreditConfig(
            episodes=1, depths=(4,), width=16, input_dim=16, num_classes=2
        )
        env = _environments(config)[0]
        joint = _compose("gradient", env, config)
        record, _ = evaluate_episode(
            joint,
            coordinate=_arm_coordinate("gradient", env),
            task_name="parity",
            campaign_id=f"{DEEP_CREDIT_CAMPAIGN_ID}::test",
            episode=0,
            batch_size=16,
            input_dim=16,
            num_classes=2,
            guard_threshold=None,
            seed=0,
            stationary_teacher=False,
        )
        assert record is not None

    def test_thermo_nonsquare_no_crash(self) -> None:
        """imp-60: thermo on nonsquare (energy_minimization dynamics) must not crash."""
        config = DeepCreditConfig(
            episodes=1, depths=(4,), width=16, input_dim=16, num_classes=2
        )
        env = _environments(config)[0]
        joint = _compose("thermodynamic_contrast", env, config)
        record, _ = evaluate_episode(
            joint,
            coordinate=_arm_coordinate("thermodynamic_contrast", env),
            task_name="parity",
            campaign_id=f"{DEEP_CREDIT_CAMPAIGN_ID}::test",
            episode=0,
            batch_size=16,
            input_dim=16,
            num_classes=2,
            guard_threshold=None,
            seed=0,
            stationary_teacher=False,
        )
        assert record is not None


class TestPilotPreregistration:
    def test_pilot_labels_itself_pilot(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.declared_rung == "pilot"
        assert result.preregistration.label() == "pilot"

    def test_claim_scope_credit_at_depth(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.claim_scope == "credit_at_depth"

    def test_task_stream_stationary(self) -> None:
        result = run_trial(LOCK_CONFIG)
        # The synthetic task with stationary_teacher=True is the stationary stream
        assert result.preregistration.task_stream == "stationary"

    def test_embedded_control_declared(self) -> None:
        result = run_trial(LOCK_CONFIG)
        assert result.preregistration.embedded_control is not None
        assert result.preregistration.embedded_control.arm == "frozen_lr0"
        assert "frozen" in result.preregistration.embedded_control.coordinate

    def test_per_depth_controls_all_pass(self) -> None:
        result = run_trial(LOCK_CONFIG)
        # All controls must pass; a failed control quarantines
        for verdict in result.control_verdicts.values():
            assert verdict["verdict"] == "passed", (
                f"control failed: {verdict['detail']}"
            )
        assert not result.quarantined


class TestContrasts:
    def test_contrasts_cover_deep_tier(self) -> None:
        result = run_trial(DeepCreditConfig(depths=(4, 16), seeds=(0, 1)))
        contrast_keys = list(result.contrasts_vs_gradient.keys())
        assert any("depth_16" in k for k in contrast_keys), (
            "contrasts must include the deep tier"
        )

    def test_gradient_reaches_above_chance_at_shallow(self) -> None:
        """Competence floor: at the shallow tier, gradient must learn."""
        acc = _walk_late("gradient", depth=4, episodes=100, lr=0.05)
        assert acc > CHANCE + 0.1, (
            f"shallow gradient competence: {acc:.4f} <= {CHANCE:.4f} + 0.1"
        )


class TestDepthDegradation:
    def test_gradient_memory_increases_with_depth(self) -> None:
        """The memory profile is the primary instrument (O(depth) vs O(1))."""
        from computronium.core.campaign.evaluation import episode_batch
        from computronium.core.profiling import measure_saved_activation_bytes

        saved_by_depth = {}
        for depth in [4, 16, 50]:
            config = DeepCreditConfig(depths=(depth,), width=8)
            env = _environments(config)[0]
            joint = _compose("gradient", env, config)
            x_batch, y_batch = episode_batch(
                episode=0,
                task_name="synthetic",
                batch_size=16,
                input_dim=8,
                num_classes=8,
                teacher_key=(1,),
                teacher_noise=0.5,
            )
            x_batch, y_batch = x_batch.to(joint.device), y_batch.to(joint.device)
            _, saved = measure_saved_activation_bytes(
                joint.train_step, x_batch, y_batch
            )
            saved_by_depth[depth] = saved.total_bytes

        assert saved_by_depth[16] > saved_by_depth[4]
        assert saved_by_depth[50] > saved_by_depth[16]

    def test_thermo_memory_constant_across_depth(self) -> None:
        from computronium.core.campaign.evaluation import episode_batch
        from computronium.core.profiling import measure_saved_activation_bytes

        for depth in [4, 16, 50]:
            config = DeepCreditConfig(depths=(depth,), width=8)
            env = _environments(config)[0]
            joint = _compose("thermodynamic_contrast", env, config)
            x_batch, y_batch = episode_batch(
                episode=0,
                task_name="synthetic",
                batch_size=16,
                input_dim=8,
                num_classes=8,
                teacher_key=(1,),
                teacher_noise=0.5,
            )
            x_batch, y_batch = x_batch.to(joint.device), y_batch.to(joint.device)
            _, saved = measure_saved_activation_bytes(
                joint.train_step, x_batch, y_batch
            )
            assert saved.total_bytes == 0, (
                f"thermo depth {depth}: {saved.total_bytes} != 0"
            )
