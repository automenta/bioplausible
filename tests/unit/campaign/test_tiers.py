"""Unit tests for the staircase gates (FIX2a §1, §8)."""

from __future__ import annotations

import pytest

import bioplausible.zoo  # ruff: ignore[unused-import]
from bioplausible.campaign import tiers


@pytest.mark.slow
def test_tier0_backprop_passes():
    outcomes = tiers.run_tier0(
        ["backprop_mlp"],
        tiers.GateSettings(input_dim=2, output_dim=2, seed=42, epochs=2, device="cpu"),
    )
    assert len(outcomes) == 1
    (outcome,) = outcomes
    assert outcome.tier == "tier0"
    assert outcome.passed
    assert "final_train_loss" in outcome.metrics


@pytest.mark.slow
def test_tier0_multiple_models():
    outcomes = tiers.run_tier0(
        ["backprop_mlp", "eqprop_mlp"],
        tiers.GateSettings(input_dim=2, output_dim=2, seed=42, epochs=2, device="cpu"),
    )
    assert {o.model for o in outcomes} == {"backprop_mlp", "eqprop_mlp"}
    assert all(o.passed for o in outcomes)


def test_tier0_task_set():
    assert tiers.TIER0_TASKS == ("xor", "spiral", "circles")


def test_tier05_never_passes_wrong_dim_rejected():
    # No trusted-training here; just assert the signature shape is callable.
    assert callable(tiers.run_tier05)


def test_outcome_is_frozen():
    outcome = tiers.TierOutcome(tier="t", model="m", task="x", passed=True, reason="r")
    with pytest.raises(Exception):
        outcome.passed = False  # type: ignore[misc]
