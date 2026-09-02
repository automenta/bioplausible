"""R8.4/R8.5 locks: power preregistration label gate + embedded controls.

Every commission declares expected effect size, variance estimate, n/group,
alpha, and stratification (imp-55); a commission below the power floor is
labeled pilot/plumbing/instrument-check, never claim-grade. Claim-grade
requires an embedded planted-effect control arm (imp-52 extension), whose
records must sit at its planted expectation — a moving or missing control
quarantines the campaign. Construct validity is gated too (imp-54):
accumulated-learning scope demands the stationary stream.
"""

import pytest

from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.resources import ResourceUsage
from computronium.validation.power_preregistration import (
    EmbeddedControl,
    PowerPreregistration,
    min_detectable_effect,
    n_for_target_power,
    verify_embedded_control,
)
from computronium.validation.preregistration import MIN_SEEDS
from computronium.validation.statistics import power_for_two_sample

CONTROL_COORD = "digital/feedforward/instantaneous/null/gradient/frozen"


def _prereg(**overrides: object) -> PowerPreregistration:
    defaults: dict[str, object] = {
        "claim": "FastWeight accumulates more than Null on the stationary stream",
        "metric": "task_accuracy",
        "claim_scope": "accumulated_learning",
        "task_stream": "stationary",
        "expected_effect": 0.8,
        "variance_estimate": 0.05,
        "n_per_group": 30,
        "embedded_control": EmbeddedControl(
            arm="null_frozen_lr0", coordinate=CONTROL_COORD, chance=0.125
        ),
    }
    defaults.update(overrides)
    return PowerPreregistration(**defaults)  # type: ignore[arg-type]


def _record(coordinate: str, accuracy: float) -> FrontierRecord:
    return FrontierRecord(
        coordinate=coordinate,
        task_name="synthetic",
        task_loss=2.0,
        task_accuracy=accuracy,
        adaptation_time=1,
        rho_jacobian=1.0,
        lyapunov_local=0.0,
        settling_time=4.0,
        basin_stability=0.5,
        resources=ResourceUsage(),
    )


class TestPowerMath:
    def test_mde_is_the_power_inverse(self) -> None:
        mde = min_detectable_effect(30)
        assert power_for_two_sample(mde, 30) == pytest.approx(0.80, abs=1e-3)

    def test_mde_decreases_with_n(self) -> None:
        assert min_detectable_effect(100) < min_detectable_effect(30)

    def test_n_for_target_power_inverts_mde(self) -> None:
        for d in (0.2, 0.5, 1.8):
            n = n_for_target_power(d)
            assert power_for_two_sample(d, n) >= 0.80
            assert power_for_two_sample(d, n - 1) < 0.80

    def test_mde_unpowerable_below_two_obs(self) -> None:
        assert min_detectable_effect(1) == float("inf")


class TestLabelGate:
    def test_fully_gated_design_is_claim_grade(self) -> None:
        assert _prereg().label() == "claim_grade"

    def test_below_power_floor_is_relabelled(self) -> None:
        weak = _prereg(expected_effect=0.1)
        assert weak.label() == "pilot"
        assert any("power floor" in r for r in weak.unmet_requirements())

    def test_missing_control_blocks_claim_grade(self) -> None:
        uncontrolled = _prereg(embedded_control=None)
        assert uncontrolled.label() == "pilot"
        assert any(
            "embedded positive control" in r for r in uncontrolled.unmet_requirements()
        )

    def test_accumulated_learning_requires_stationary_stream(self) -> None:
        legacy = _prereg(task_stream="per_episode")
        assert any("imp-54" in r for r in legacy.unmet_requirements())

    def test_retention_requires_segmented_stream(self) -> None:
        for stream in ("per_episode", "stationary"):
            prereg = _prereg(claim_scope="retention", task_stream=stream)
            assert any("segmented" in r for r in prereg.unmet_requirements()), stream

    def test_retention_on_segmented_stream_gates_normally(self) -> None:
        prereg = _prereg(claim_scope="retention", task_stream="segmented")
        assert prereg.unmet_requirements() == ()
        assert prereg.label() == "claim_grade"

    def test_accumulated_learning_rejects_segmented_stream(self) -> None:
        assert any(
            "imp-54" in r for r in _prereg(task_stream="segmented").unmet_requirements()
        )

    def test_declared_rung_caps_even_when_gates_pass(self) -> None:
        pilot = _prereg(declared_rung="pilot")
        assert pilot.unmet_requirements() == ()
        assert pilot.label() == "pilot"

    def test_require_claim_grade_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="power floor"):
            _prereg(expected_effect=0.1).require_claim_grade()


class TestEmbeddedControl:
    def test_control_at_chance_passes(self) -> None:
        records = [_record(CONTROL_COORD, 0.10), _record(CONTROL_COORD, 0.15)]
        verdict = verify_embedded_control(
            records, EmbeddedControl(arm="c", coordinate=CONTROL_COORD, chance=0.125)
        )
        assert verdict.verdict == "passed"
        assert not verdict.quarantines

    def test_moving_control_quarantines(self) -> None:
        records = [_record(CONTROL_COORD, 0.60), _record(CONTROL_COORD, 0.70)]
        verdict = verify_embedded_control(
            records, EmbeddedControl(arm="c", coordinate=CONTROL_COORD, chance=0.125)
        )
        assert verdict.verdict == "failed"
        assert verdict.quarantines

    def test_missing_control_records_quarantine(self) -> None:
        verdict = verify_embedded_control(
            [
                _record(
                    "digital/feedforward/instantaneous/fast_weights/gradient/euclidean",
                    0.5,
                )
            ],
            EmbeddedControl(arm="c", coordinate=CONTROL_COORD, chance=0.125),
        )
        assert verdict.verdict == "missing"
        assert verdict.quarantines


class TestRegistrationRoundTrip:
    def test_json_round_trip_preserves_the_gate(self) -> None:
        original = _prereg()
        revived = PowerPreregistration.from_dict(original.to_dict())
        assert revived == original
        assert revived.label() == original.label()
        assert revived.mde_metric == pytest.approx(original.mde_metric)

    def test_file_round_trip(self, tmp_path) -> None:
        path = _prereg().save(tmp_path / "prereg.json")
        assert PowerPreregistration.load(path) == _prereg()


def test_min_seeds_floor_is_respected_by_the_gate() -> None:
    thin = _prereg(n_per_group=MIN_SEEDS - 1)
    assert any("below floor" in r for r in thin.unmet_requirements())
