"""R9.1 segmented-stream locks: the retention trial's construct validity.

The structured task-sequence stream (A→B) is the environment retention is
defined on: teachers are stationary within a segment (accumulation
representable per segment — R8.3 machinery keyed per segment) and shift
across segments (forgetting measurable). Locks pin the machinery, never the
scientific outcome: segment-keying semantics, the no-train retention probe's
honesty (θ untouched; identical readout to the pipeline's post-update
``free_accuracy``), the preregistration scope rule, and the trial harness's
pilot labeling with its embedded control.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from computronium.core.campaign.evaluation import (
    build_coordinate_system,
    evaluate_episode,
    probe_episode,
)
from computronium.experiments.joint.forgetting_trial import (
    CONTROL_COORDINATE,
    PROBE_EPISODE_BASE,
    PROBED_SEGMENT,
    TRIAL_CAMPAIGN_ID,
    TrialConfig,
    run_trial,
)
from computronium.validation.power_preregistration import (
    EmbeddedControl,
    PowerPreregistration,
    min_detectable_effect,
)

if TYPE_CHECKING:
    from computronium.core.system_trainer import JointSystem

BATCH, DIM, CLASSES = 64, 4, 4  # square shape: the guard probe feeds output→input
CHANCE = 1.0 / CLASSES
COORDINATE = "digital/feedforward/instantaneous/null/gradient/euclidean"


def probe_batch(episode: int, segment: str) -> tuple[torch.Tensor, torch.Tensor]:
    from computronium.core.campaign.evaluation import _teacher_key, episode_batch

    key = _teacher_key(
        TRIAL_CAMPAIGN_ID, COORDINATE, 0, stationary=True, segment=segment
    )
    return episode_batch(
        episode, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=key
    )


def _linear_probe_fit(
    train: list[tuple[torch.Tensor, torch.Tensor]],
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    torch.manual_seed(0)
    probe = torch.nn.Linear(DIM, CLASSES)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    x_train = torch.cat([x for x, _ in train])
    y_train = torch.cat([y for _, y in train])
    for _ in range(60):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(probe(x_train), y_train).backward()
        optimizer.step()
    with torch.no_grad():
        return (probe(test_x).argmax(dim=-1) == test_y).float().mean().item()


class TestSegmentedStream:
    def test_teacher_is_stationary_within_a_segment(self) -> None:
        early = [probe_batch(e, PROBED_SEGMENT) for e in range(8)]
        late = [probe_batch(PROBE_EPISODE_BASE + e, PROBED_SEGMENT) for e in range(8)]
        accuracy = _linear_probe_fit(early, *[torch.cat(t) for t in zip(*late)])
        assert accuracy > 0.9  # accumulation representable per segment

    def test_teacher_shifts_across_segments(self) -> None:
        a_train = [probe_batch(e, PROBED_SEGMENT) for e in range(8)]
        b_x, b_y = (torch.cat(t) for t in zip(*[probe_batch(e, "B") for e in range(8)]))
        accuracy = _linear_probe_fit(a_train, b_x, b_y)
        assert accuracy < 0.7  # a different segment is a different task

    def test_segment_requires_stationary_teacher(self) -> None:
        joint = build_coordinate_system(COORDINATE)
        with pytest.raises(ValueError, match="segment-keyed teachers require"):
            evaluate_episode(
                joint,
                coordinate=COORDINATE,
                task_name="synthetic",
                campaign_id=TRIAL_CAMPAIGN_ID,
                episode=0,
                stationary_teacher=False,
                segment="A",
            )
        with pytest.raises(ValueError, match="segment-keyed teachers require"):
            probe_episode(
                joint,
                coordinate=COORDINATE,
                campaign_id=TRIAL_CAMPAIGN_ID,
                episode=0,
                stationary_teacher=False,
                segment="A",
            )


class TestRetentionProbe:
    def _square_system(self) -> JointSystem:
        return build_coordinate_system(COORDINATE, input_dim=DIM, output_dim=CLASSES)

    def test_probe_does_not_train(self) -> None:
        joint = self._square_system()
        before = {k: v.clone() for k, v in joint.geometry.params.items()}
        probe_episode(
            joint,
            coordinate=COORDINATE,
            campaign_id=TRIAL_CAMPAIGN_ID,
            episode=PROBE_EPISODE_BASE,
            batch_size=16,
            input_dim=DIM,
            num_classes=CLASSES,
            stationary_teacher=True,
            segment=PROBED_SEGMENT,
        )
        for name, param in joint.geometry.params.items():
            assert torch.equal(param, before[name])

    def test_probe_matches_the_pipeline_free_readout(self) -> None:
        joint = self._square_system()
        episode = 3
        record, _ = evaluate_episode(
            joint,
            coordinate=COORDINATE,
            task_name="synthetic",
            campaign_id=TRIAL_CAMPAIGN_ID,
            episode=episode,
            batch_size=BATCH,
            input_dim=DIM,
            num_classes=CLASSES,
            guard_threshold=None,
            stationary_teacher=True,
            segment=PROBED_SEGMENT,
        )
        accuracy = probe_episode(
            joint,
            coordinate=COORDINATE,
            campaign_id=TRIAL_CAMPAIGN_ID,
            episode=episode,
            batch_size=BATCH,
            input_dim=DIM,
            num_classes=CLASSES,
            stationary_teacher=True,
            segment=PROBED_SEGMENT,
        )
        assert accuracy == pytest.approx(record.task_accuracy)

    def test_probe_scores_the_probed_segment_not_the_walk(self) -> None:
        from computronium.experiments.joint.forgetting_trial import _compose

        config = TrialConfig(
            segments=((PROBED_SEGMENT, 30), ("B", 1)),
            seeds=(0,),
            num_classes=CLASSES,
            input_dim=DIM,
            lr=0.05,
            teacher_noise=0.0,
        )
        joint = _compose(COORDINATE, config)
        for episode in range(30):  # train on A only
            evaluate_episode(
                joint,
                coordinate=COORDINATE,
                task_name="synthetic",
                campaign_id=TRIAL_CAMPAIGN_ID,
                episode=episode,
                batch_size=BATCH,
                input_dim=DIM,
                num_classes=CLASSES,
                guard_threshold=None,
                stationary_teacher=True,
                segment=PROBED_SEGMENT,
            )
        kwargs: dict[str, object] = {
            "batch_size": BATCH,
            "input_dim": DIM,
            "num_classes": CLASSES,
            "stationary_teacher": True,
        }
        a = probe_episode(
            joint,
            coordinate=COORDINATE,
            campaign_id=TRIAL_CAMPAIGN_ID,
            episode=PROBE_EPISODE_BASE,
            segment=PROBED_SEGMENT,
            **kwargs,  # type: ignore[arg-type]
        )
        b = probe_episode(
            joint,
            coordinate=COORDINATE,
            campaign_id=TRIAL_CAMPAIGN_ID,
            episode=PROBE_EPISODE_BASE,
            segment="B",
            **kwargs,  # type: ignore[arg-type]
        )
        assert a > CHANCE + 0.3  # the walk accumulated on A
        assert a > b + 0.1  # the probe measures the probed segment, not the walk


class TestRetentionPreregistration:
    def _prereg(self) -> PowerPreregistration:
        return PowerPreregistration(
            claim="trial lock: the pilot prereg records the retention scope",
            metric="task_accuracy",
            claim_scope="retention",
            task_stream="segmented",
            expected_effect=0.8,
            variance_estimate=0.05,
            n_per_group=8,
            embedded_control=EmbeddedControl(
                arm="null_frozen_lr0",
                coordinate=CONTROL_COORDINATE,
                chance=0.125,
                tolerance=0.2,
            ),
            declared_rung="pilot",
        )

    def test_trial_labels_itself_pilot(self) -> None:
        result = run_trial(
            TrialConfig(
                segments=((PROBED_SEGMENT, 2), ("B", 2)),
                probe_episodes=2,
                seeds=(0,),
            )
        )
        assert result.preregistration.label() == "pilot"
        assert result.preregistration.claim_scope == "retention"
        assert result.preregistration.task_stream == "segmented"
        assert result.control_verdict is not None
        assert set(result.arms) == {"null", "fast_weights", "routing", "control"}

    def test_per_arm_records_carry_segment_provenance(self) -> None:
        from computronium.experiments.joint.forgetting_trial import _walk_arm

        config = TrialConfig(
            segments=((PROBED_SEGMENT, 2), ("B", 2)),
            probe_episodes=2,
            seeds=(0,),
        )
        _arm, records = _walk_arm("control", CONTROL_COORDINATE, config)
        segments = {r.metadata["segment"] for r in records}
        assert segments == {PROBED_SEGMENT, "B"}
        assert all(
            r.metadata["teacher_stationary"] == pytest.approx(1.0) for r in records
        )

    def test_schedule_must_start_with_the_probed_segment(self) -> None:
        with pytest.raises(ValueError, match="must start with segment"):
            TrialConfig(segments=(("B", 5),))


class TestTrialIsLearnablePerSegment:
    """Construct validity: the walk's training stream is accumulable — the
    null arm must reach above-chance mastery on A at pilot scale, else the
    whole retention contrast measures noise (imp-36 class, by design).
    Multi-seed by intent: a probe keyed to one seed's teacher while another
    seed walks scores chance by construction (the _probe seed-threading
    defect this lock catches)."""

    def test_null_reaches_above_chance_mastery_on_a(self) -> None:
        result = run_trial(
            TrialConfig(
                segments=((PROBED_SEGMENT, 30), ("B", 5)),
                probe_episodes=8,
                seeds=(0, 1),
                lr=0.05,
            )
        )
        # registered shape (8 inputs / 8 classes): chance = 1/8
        assert min(result.arms["null"].a_mastery) > 0.3
        assert not result.quarantined


class TestRegisteredCommission:
    """The registered R9.1 commission rides the R8.4 gate (imp-55 policy:
    pre-registration precedes comparison) — the prereg file must derive
    claim-grade from its own numbers, and the trial must refuse to walk
    under anything less."""

    REGISTERED_PREREG = (
        Path(__file__).parents[2]
        / "configs/preregistrations/r91_retention_registered.json"
    )

    def test_registered_prereg_derives_claim_grade(self) -> None:
        prereg = PowerPreregistration.load(self.REGISTERED_PREREG)
        assert prereg.unmet_requirements() == ()
        assert prereg.label() == "claim_grade"
        assert prereg.declared_rung is None  # claim-grade is derived, never declared
        assert prereg.claim_scope == "retention"
        assert prereg.task_stream == "segmented"
        assert prereg.embedded_control is not None
        control = prereg.embedded_control
        assert control.coordinate == CONTROL_COORDINATE
        assert control.chance == pytest.approx(0.125)  # registered 8-class shape
        assert control.tolerance >= 0.05  # imp-59 floor at the registered N
        assert prereg.mde_cohens_d <= prereg.expected_effect

    def test_trial_refuses_unmet_claim_grade_gates(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: unmet gates refuse the commission",
            metric="task_accuracy",
            claim_scope="retention",
            task_stream="segmented",
            expected_effect=0.1,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="null_frozen_lr0",
                coordinate=CONTROL_COORDINATE,
                chance=CHANCE,
                tolerance=0.2,
            ),
        )
        with pytest.raises(ValueError, match="not claim-grade"):
            run_trial(
                TrialConfig(segments=((PROBED_SEGMENT, 2), ("B", 2)), seeds=(0,)),
                preregistration=prereg,
            )

    def test_trial_refuses_a_declared_rung_cap(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: a declared rung caps the label below claim-grade",
            metric="task_accuracy",
            claim_scope="retention",
            task_stream="segmented",
            expected_effect=min_detectable_effect(5) + 0.01,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="null_frozen_lr0",
                coordinate=CONTROL_COORDINATE,
                chance=CHANCE,
                tolerance=0.2,
            ),
            declared_rung="pilot",
        )
        with pytest.raises(ValueError, match="caps the label below claim-grade"):
            run_trial(
                TrialConfig(segments=((PROBED_SEGMENT, 2), ("B", 2)), seeds=(0,)),
                preregistration=prereg,
            )

    def test_trial_enforces_the_registered_n(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: the walk must deliver the registered n",
            metric="task_accuracy",
            claim_scope="retention",
            task_stream="segmented",
            expected_effect=min_detectable_effect(5) + 0.01,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="null_frozen_lr0",
                coordinate=CONTROL_COORDINATE,
                chance=CHANCE,
                tolerance=0.2,
            ),
        )
        with pytest.raises(ValueError, match="registered design requires"):
            run_trial(
                TrialConfig(segments=((PROBED_SEGMENT, 2), ("B", 2)), seeds=(0,)),
                preregistration=prereg,
            )

    def test_trial_commissions_through_a_registered_prereg(self) -> None:
        prereg = PowerPreregistration(
            claim="lock: claim-grade prereg carries the commission",
            metric="task_accuracy",
            claim_scope="retention",
            task_stream="segmented",
            expected_effect=min_detectable_effect(5) + 0.01,
            variance_estimate=0.05,
            n_per_group=5,
            embedded_control=EmbeddedControl(
                arm="null_frozen_lr0",
                coordinate=CONTROL_COORDINATE,
                chance=CHANCE,
                tolerance=0.2,
            ),
        )
        result = run_trial(
            TrialConfig(
                segments=((PROBED_SEGMENT, 2), ("B", 2)),
                probe_episodes=2,
                seeds=(0, 1, 2, 3, 4),
            ),
            preregistration=prereg,
        )
        assert result.preregistration is prereg
        assert result.preregistration.label() == "claim_grade"
        assert result.control_verdict is not None
        assert set(result.arms) == {"null", "fast_weights", "routing", "control"}
