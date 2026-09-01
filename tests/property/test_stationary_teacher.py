"""R8.3 stationarity locks for the stationary-teacher task design (imp-54).

The legacy per-episode ``synthetic`` stream redraws its teacher every episode,
so pooled campaign deltas are capped at chance for accumulated learning
(pinned by ``test_positive_control.py::...non_stationary_by_design``). The
stationary design derives the teacher from ``(campaign_id, coordinate, seed)``
alone: same teacher across episodes, varying inputs — accumulated learning
becomes representable. The decisive lock is behavioral: a linear probe trained
on early episodes generalizes to late episodes under the stationary teacher
and cannot under the legacy stream. Legacy-stream reproduction is also pinned
so commissioned R5.1a/b artifacts stay reproducible.
"""

import pytest
import torch

from computronium.core.campaign.evaluation import (
    CALIBRATED_TEACHER_NOISE,
    _stable_seed,
    build_coordinate_system,
    episode_batch,
)
from computronium.core.campaign.stack import CampaignStack
from computronium.validation.power_preregistration import (
    EmbeddedControl,
    PowerPreregistration,
)

BATCH, DIM, CLASSES = 64, 8, 4
KEY = (
    "r83-stationarity",
    "digital/feedforward/instantaneous/null/gradient/euclidean",
    0,
)


def _stream(teacher_key: tuple[object, ...] | None, episodes: int) -> list[tuple]:
    return [
        episode_batch(
            e,
            batch_size=BATCH,
            input_dim=DIM,
            num_classes=CLASSES,
            teacher_key=teacher_key,
        )
        for e in range(episodes)
    ]


def _probe_accuracy(train: list[tuple], test: list[tuple]) -> float:
    """Fit a linear classifier on ``train`` episodes, score on ``test``."""
    torch.manual_seed(0)
    probe = torch.nn.Linear(DIM, CLASSES)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    x_train = torch.cat([x for x, _ in train])
    y_train = torch.cat([y for _, y in train])
    for _ in range(60):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(probe(x_train), y_train).backward()
        optimizer.step()
    x_test = torch.cat([x for x, _ in test])
    y_test = torch.cat([y for _, y in test])
    with torch.no_grad():
        return (probe(x_test).argmax(dim=-1) == y_test).float().mean().item()


class TestStationaryTeacherDesign:
    def test_legacy_stream_is_reproduced_exactly(self) -> None:
        """teacher_key=None matches the pre-R8.3 stream byte-for-byte."""
        generator = torch.Generator().manual_seed(1000 + 7)
        x = torch.randn(BATCH, DIM, generator=generator)
        weights = torch.randn(DIM, CLASSES, generator=generator)
        expected = (x @ weights).argmax(dim=-1)
        got_x, got_y = episode_batch(
            7, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES
        )
        assert torch.equal(got_x, x)
        assert torch.equal(got_y, expected)

    def test_inputs_vary_but_x_stream_is_untouched_by_the_key(self) -> None:
        plain_x, _ = episode_batch(
            3, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES
        )
        keyed_x, _ = episode_batch(
            3, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=KEY
        )
        assert torch.equal(plain_x, keyed_x)
        other_x, _ = episode_batch(
            4, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=KEY
        )
        assert not torch.equal(keyed_x, other_x)  # inputs keep varying

    def test_teacher_depends_on_the_full_key(self) -> None:
        """Same episode, different key components → different teacher labels."""
        _base_x, base_y = episode_batch(
            0, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=KEY
        )
        for variant in (
            (
                "r83-stationarity",
                "digital/feedforward/instantaneous/null/gradient/euclidean",
                1,
            ),
            (
                "other-campaign",
                "digital/feedforward/instantaneous/null/gradient/euclidean",
                0,
            ),
            (
                "r83-stationarity",
                "digital/recurrent/instantaneous/null/gradient/euclidean",
                0,
            ),
        ):
            _, other_y = episode_batch(
                0,
                batch_size=BATCH,
                input_dim=DIM,
                num_classes=CLASSES,
                teacher_key=variant,
            )
            assert not torch.equal(base_y, other_y), variant

    def test_teacher_key_rejected_for_parity(self) -> None:
        with pytest.raises(ValueError, match="stationary teachers"):
            episode_batch(0, task_name="parity", teacher_key=KEY)

    def test_accumulated_learning_is_representable(self) -> None:
        """Linear probe trained on early episodes generalizes to late ones.

        The same probe on the legacy (teacher-redraw) stream stays at chance —
        the imp-54 non-stationarity, preserved by design for per-episode scope.
        """
        stationary = _stream(KEY, episodes=12)
        legacy = _stream(None, episodes=12)
        late = [
            episode_batch(
                e, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=KEY
            )
            for e in range(90, 100)
        ]
        assert _probe_accuracy(stationary[:8], late) > 0.9
        assert _probe_accuracy(legacy[:8], late) < 0.35  # chance = 1/4


class TestTeacherNoiseCalibration:
    """R8.3 difficulty knob: teacher-logit noise lowers the Bayes ceiling."""

    def test_noise_free_stream_is_unchanged(self) -> None:
        noiseless, _ = episode_batch(
            5, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=KEY
        )
        explicit, _ = episode_batch(
            5,
            batch_size=BATCH,
            input_dim=DIM,
            num_classes=CLASSES,
            teacher_key=KEY,
            teacher_noise=0.0,
        )
        assert torch.equal(noiseless, explicit)

    def test_noise_changes_labels_not_inputs(self) -> None:
        x0, y0 = episode_batch(
            5, batch_size=BATCH, input_dim=DIM, num_classes=CLASSES, teacher_key=KEY
        )
        x1, y1 = episode_batch(
            5,
            batch_size=BATCH,
            input_dim=DIM,
            num_classes=CLASSES,
            teacher_key=KEY,
            teacher_noise=0.5,
        )
        assert torch.equal(x0, x1)
        assert not torch.equal(y0, y1)

    def test_calibrated_noise_keeps_oracle_accuracy_in_the_measurable_band(
        self,
    ) -> None:
        """At CALIBRATED_TEACHER_NOISE the oracle (noiseless-teacher predictor
        against noisy labels) sits well below the 1.0 ceiling and far above
        chance — accuracy axes cannot saturate (imp-36 class)."""
        agreements = []
        for episode in range(20):
            generator = torch.Generator().manual_seed(
                _stable_seed("synthetic", "teacher", *KEY)
            )
            x = torch.randn(256, DIM)
            weights = torch.randn(DIM, CLASSES, generator=generator)
            clean = (x @ weights).argmax(-1)
            noisy_logits = x @ weights + CALIBRATED_TEACHER_NOISE * torch.randn(
                256, CLASSES, generator=generator
            )
            agreements.append((clean == noisy_logits.argmax(-1)).float().mean().item())
        mean_agreement = sum(agreements) / len(agreements)
        assert 1.0 / CLASSES + 0.3 < mean_agreement < 0.95


class TestStationaryCampaignThreading:
    """R8.3/R8.5: the CampaignStack carries the task-stream design, records
    the claim label, and verifies the embedded control post-run."""

    CONTROL = "digital/feedforward/instantaneous/null/gradient/frozen"
    GRID = (
        "digital/feedforward/instantaneous/null/gradient/euclidean",
        CONTROL,
    )

    def _prereg(self) -> PowerPreregistration:
        return PowerPreregistration(
            claim="threading lock: campaign records its own claim label",
            metric="task_accuracy",
            claim_scope="per_episode_adaptation",
            task_stream="stationary",
            expected_effect=0.5,
            variance_estimate=0.05,
            n_per_group=8,
            embedded_control=EmbeddedControl(
                arm="null_frozen_lr0",
                coordinate=self.CONTROL,
                chance=0.25,
                tolerance=0.2,
            ),
            declared_rung="instrument_check",
        )

    def test_frozen_update_leaves_theta_bitwise_unchanged(self) -> None:
        joint = build_coordinate_system(self.CONTROL)
        before = {k: v.clone() for k, v in joint.geometry.params.items()}
        x, y = episode_batch(0, batch_size=8, input_dim=8, num_classes=8)
        joint.train_step(x, y)
        for name, param in joint.geometry.params.items():
            assert torch.equal(param, before[name])

    def test_campaign_records_design_and_verifies_control(self, tmp_path) -> None:
        stack = CampaignStack(tmp_path, seed=0)
        result = stack.run_campaign(
            iterations=2,
            experiments_per_iter=2,
            sampler=lambda _rng, iteration, experiment: self.GRID[
                (iteration * 2 + experiment) % len(self.GRID)
            ],
            campaign_id="r83_threading_lock",
            stationary_teacher=True,
            preregistration=self._prereg(),
        )
        records = stack.frontier_records("r83_threading_lock")
        assert records
        for record in records:
            assert record.metadata["teacher_stationary"] == pytest.approx(1.0)
        state = stack.store.get_campaign("r83_threading_lock")
        assert state is not None
        assert state.config["stationary_teacher"] is True
        assert state.config["claim_label"] == "instrument_check"
        assert result.claim_label == "instrument_check"
        assert result.embedded_control is not None
        assert result.embedded_control.verdict == "passed"
        assert not result.quarantined

    def test_legacy_campaign_metadata_flags_non_stationary(self, tmp_path) -> None:
        stack = CampaignStack(tmp_path, seed=0)
        stack.run_campaign(
            iterations=1,
            experiments_per_iter=1,
            sampler=lambda _rng, _iteration, _experiment: self.GRID[0],
            campaign_id="r83_legacy_lock",
        )
        record = stack.frontier_records("r83_legacy_lock")[0]
        assert record.metadata["teacher_stationary"] == pytest.approx(0.0)
