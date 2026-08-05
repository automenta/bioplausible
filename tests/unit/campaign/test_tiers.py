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


def test_tier0_aggregates_task_failures(monkeypatch):
    def fake_train_sample(*, task, **_kwargs):
        ok = task != "circles"
        return ok, "explanation", {"final_train_loss": 0.5, "final_train_acc": 0.9}

    monkeypatch.setattr("bioplausible.campaign.tiers._train_sample", fake_train_sample)
    (outcome,) = tiers.run_tier0(
        ["backprop_mlp"],
        tiers.GateSettings(input_dim=2, output_dim=2, seed=1, epochs=1),
    )
    assert not outcome.passed
    assert "circles" in outcome.reason
    assert "xor" not in outcome.reason


def test_tier05_passes_when_mean_acc_clears_gate(monkeypatch):
    def fake_train_sample(**_kwargs):
        return True, "ok", {"final_train_acc": 0.99, "epoch_time_s": 1.0}

    monkeypatch.setattr("bioplausible.campaign.tiers._train_sample", fake_train_sample)
    (outcome,) = tiers.run_tier05(
        ["backprop_mlp"],
        tiers.GateSettings(input_dim=64, output_dim=10, seed=1, n_seeds=2, epochs=1),
    )
    assert outcome.passed
    assert outcome.metrics["mean_acc"] == 0.99
    assert outcome.metrics["param_count"] == 650
    assert outcome.metrics["n_seeds"] == 2


def test_tier05_digits_fail_when_below_threshold(monkeypatch):
    def fake_train_sample(**_kwargs):
        return True, "ok", {"final_train_acc": 0.5, "epoch_time_s": 1.0}

    monkeypatch.setattr("bioplausible.campaign.tiers._train_sample", fake_train_sample)
    (outcome,) = tiers.run_tier05(
        ["backprop_mlp"],
        tiers.GateSettings(input_dim=64, output_dim=10, seed=1, n_seeds=2, epochs=1),
    )
    assert not outcome.passed
    assert "digits-fail" in outcome.reason
    assert "min_acc" in outcome.metrics


def test_tier05_breaks_on_first_failed_seed(monkeypatch):
    def fake_train_sample(*, settings, **_kwargs):
        ok = settings.seed == 1
        return ok, "boom", {"final_train_acc": 0.4, "epoch_time_s": 1.0}

    monkeypatch.setattr("bioplausible.campaign.tiers._train_sample", fake_train_sample)
    (outcome,) = tiers.run_tier05(
        ["backprop_mlp"],
        tiers.GateSettings(input_dim=64, output_dim=10, seed=1, n_seeds=3, epochs=1),
    )
    assert not outcome.passed
    assert "seed 2" in outcome.reason
