"""R7 imp-52 positive control: planted-effect instrument self-test.

The termination criterion is "if it works it will be obvious". This module
makes that operational: plant an obvious synthetic effect and require the
campaign instrument to detect it. An instrument self-test, not a scientific
claim — it proves the microscope can see *something*.

Control A (lr=0 vs lr>0, subsuming trained-vs-untrained and
zero-update-vs-real-credit): both arms run real campaign episodes through
``evaluate_episode`` — identical task batch, identical θ init (seeded
composition), identical claim-field extraction. On a stationary task the
lr>0 arm must drive the target-free claim metrics to ceiling while the
lr=0 arm stays at chance.

Task-stream property discovered while building this control: the campaign's
per-episode ``synthetic`` family redraws a fresh random teacher each
episode (non-stationary by design), so *no* fixed θ can accumulate learning
across the smoke-scale stream — both arms sit at chance there. That is a
task-design property, not an instrument defect (the stationary control
below detects the planted effect at ceiling); it is pinned here because it
caps what any pooled smoke-campaign task_loss delta can ever show.

Control D (ψ engaged vs ψ frozen) is pinned at the pipeline level by
``tests/property/test_psi_engagement.py`` (frozen-ψ metrics-response lock).

Resource revalidation rides along (R7 second-pass guardrail): across
heterogeneous coordinates the work-derived compute/energy axes must vary,
be nonnegative, and split from the free-energy state variable (imp-45).

Policy: no campaign is interpreted unless this probe detects its planted
effect.
"""

from __future__ import annotations

import pytest
import torch

from computronium.core.campaign.evaluation import (
    build_coordinate_system,
    evaluate_episode,
)
from computronium.core.system_trainer import compose_joint_system_from_configs
from computronium.ontology import (
    CreditAssignmentConfig,
    GeometryConfig,
    ParameterUpdateConfig,
    PlasticityConfig,
    StateDynamicsConfig,
    SubstrateConfig,
)

COORDINATE = "digital/feedforward/energy_minimization/null/gradient/euclidean"
EPISODES = 30
SEEDS = (0, 1, 2)


def _system(step_size: float, seed: int):
    torch.manual_seed(seed)
    return compose_joint_system_from_configs(
        SubstrateConfig.digital(),
        GeometryConfig.feedforward(input_dim=8, output_dim=8, hidden_dims=(16,)),
        StateDynamicsConfig.energy_minimization(max_steps=3, step_size=0.1),
        PlasticityConfig.null(),
        CreditAssignmentConfig.gradient(),
        ParameterUpdateConfig.euclidean(step_size=step_size),
    )


def _run_arm(step_size: float, *, stream: bool = False) -> list[float]:
    """Per-seed accuracy curves. ``stream=False`` pins one stationary task
    batch (fixed episode); ``stream=True`` walks the standard per-episode
    stream (fresh random teacher each episode)."""
    curves: list[float] = []
    for seed in SEEDS:
        system = _system(step_size, seed)
        for episode in range(EPISODES):
            record, _ = evaluate_episode(
                system,
                coordinate=COORDINATE,
                task_name="synthetic",
                campaign_id="imp52-positive-control",
                episode=episode if stream else 0,
                guard_threshold=None,
            )
            curves.append(record.task_accuracy)
    return curves


class TestPositiveControlLearning:
    @pytest.fixture(scope="class")
    def arms(self) -> tuple[list[float], list[float]]:
        return _run_arm(0.0), _run_arm(0.1)

    def test_trained_arm_reaches_ceiling(self, arms) -> None:
        _, trained = arms
        late = [
            trained[i * EPISODES + EPISODES - 5 : (i + 1) * EPISODES]
            for i in range(len(SEEDS))
        ]
        flat = [v for window in late for v in window]
        mean = sum(flat) / len(flat)
        assert mean > 0.8, (
            f"trained arm failed to learn a stationary task: late mean={mean:.3f}"
        )

    def test_untrained_arm_stays_at_chance(self, arms) -> None:
        untrained, _ = arms
        assert sum(untrained) / len(untrained) < 0.3, (
            "lr=0 arm 'learned' — evaluation path leaks or task is degenerate"
        )

    def test_planted_effect_detected_for_every_seed(self, arms) -> None:
        untrained, trained = arms
        n = len(SEEDS)
        for i in range(n):
            u = untrained[i * EPISODES : (i + 1) * EPISODES]
            t = trained[i * EPISODES : (i + 1) * EPISODES]
            gap = sum(t[-5:]) / 5 - sum(u[-5:]) / 5
            assert gap > 0.5, f"seed {i}: planted effect not detected, gap={gap:.3f}"

    def test_effect_direction_is_improvement(self, arms) -> None:
        _, trained = arms
        for i in range(len(SEEDS)):
            curve = trained[i * EPISODES : (i + 1) * EPISODES]
            early, late = sum(curve[:5]) / 5, sum(curve[-5:]) / 5
            assert late > early, (
                f"seed {i}: no improvement (early={early:.3f}, late={late:.3f})"
            )

    def test_per_episode_stream_is_non_stationary_by_design(self) -> None:
        """Task-design lock: fresh teacher per episode caps pooled smoke claims.

        On the standard per-episode ``synthetic`` stream a single θ cannot
        accumulate learning — the lr>0 arm sits at chance just like lr=0.
        Pooled smoke-campaign task_loss/task_accuracy deltas are therefore
        capped at chance-level by construction and must never be read as
        (evidence of absence of) learning effects.
        """
        torch.manual_seed(SEEDS[0])
        system = _system(0.1, SEEDS[0])
        accs = []
        for episode in range(EPISODES):
            record, _ = evaluate_episode(
                system,
                coordinate=COORDINATE,
                task_name="synthetic",
                campaign_id="imp52-positive-control",
                episode=episode,
                guard_threshold=None,
            )
            accs.append(record.task_accuracy)
        mean = sum(accs) / len(accs)
        assert mean < 0.3, (
            f"per-episode stream unexpectedly learnable (mean={mean:.3f}) — "
            "episode_batch semantics changed; re-evaluate the smoke-campaign "
            "interpretation"
        )


class TestResourceRevalidation:
    """R7 guardrail: the fixed resource semantics must hold on live episodes."""

    COORDS = (
        "digital/feedforward/energy_minimization/null/gradient/euclidean",
        "digital/recurrent/instantaneous/null/local_goodness/euclidean",
        "digital/feedforward/instantaneous/fast_weights/gradient/euclidean",
    )

    @pytest.fixture(scope="class")
    def records(self):
        out = []
        for coordinate in self.COORDS:
            joint = build_coordinate_system(coordinate)
            record, metrics = evaluate_episode(
                joint,
                coordinate=coordinate,
                task_name="parity",
                campaign_id="imp52-resource-reval",
                episode=0,
                guard_threshold=None,
            )
            out.append((record, metrics))
        return out

    def test_compute_axes_vary_and_are_nonnegative(self, records) -> None:
        compute = [r.resources.compute for r, _ in records]
        energy = [r.resources.energy for r, _ in records]
        assert all(c > 0 for c in compute)
        assert all(e > 0 for e in energy)
        assert len(set(compute)) == len(compute), (
            f"compute axis constant across heterogeneous grid: {compute} — "
            "resource fiction (imp-17/imp-45 signature)"
        )

    def test_state_energy_split_from_consumption(self, records) -> None:
        for record, metrics in records:
            assert record.resources.state_energy_j == pytest.approx(
                metrics["free_energy"]
            )
            assert record.resources.energy > 0

    def test_psi_capacity_discriminates_primitives(self, records) -> None:
        capacities = [r.resources.plastic_state_capacity for r, _ in records]
        assert capacities[0] == pytest.approx(0.0)  # NullPlasticity
        assert capacities[2] > 0.0  # FastWeightPlasticity
