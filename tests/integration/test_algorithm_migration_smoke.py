"""Fast CI smoke for the algorithm-migration benchmark (Level 3.5).

Runs one tiny migration episode to verify the benchmark machinery composes,
trains, and reports — NOT a result claim. The full-parameterized suite runs
in the slow tier (`tests/integration/joint/test_benchmarks.py`).
"""

from __future__ import annotations

import pytest

COORDINATES = (
    "digital/recurrent/instantaneous/null/gradient/euclidean",
    "digital/recurrent/instantaneous/fast_weights/gradient/euclidean",
)


@pytest.mark.integration
@pytest.mark.parametrize("coordinate", COORDINATES)
def test_algorithm_migration_smoke(coordinate: str) -> None:
    """Minimal migration run: valid outputs and θ-change accounting."""
    from computronium.experiments.joint.algorithm_migration import evaluate_migration

    result = evaluate_migration(
        coordinate=coordinate,
        epochs_a0=1,
        epochs_a1=1,
        batch_size=16,
        seq_len=5,
        input_dim=8,
        device="cpu",
        seed=42,
    )

    assert result["coordinate"] == coordinate
    for key in (
        "a0_accuracy",
        "a1_accuracy",
        "a0_accuracy_after_a1",
        "migration_time",
        "theta_change",
        "resources",
    ):
        assert key in result
    assert 0 <= result["a0_accuracy"] <= 1
    assert 0 <= result["a1_accuracy"] <= 1
    assert result["theta_change"] >= 0
    assert result["migration_time"] >= 0
    # Resource vector (C compute axis) recorded via measure_suite_resources.
    assert result["resources"]["compute"] > 0


def test_algorithm_migration_rejects_bad_coordinate() -> None:
    from computronium.experiments.joint.algorithm_migration import evaluate_migration

    with pytest.raises(ValueError, match="Invalid coordinate"):
        evaluate_migration(
            coordinate="digital/feedforward",
            epochs_a0=1,
            epochs_a1=1,
            batch_size=8,
            seq_len=4,
            input_dim=8,
            device="cpu",
        )
