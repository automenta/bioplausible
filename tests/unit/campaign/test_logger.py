"""Unit tests for the JSONL experiment logger (FIX2a §4.3)."""

from __future__ import annotations

import json

import pytest

from bioplausible.campaign.logger import (
    Epoch,
    ExperimentLogger,
    GateOutcome,
    TrialEnd,
    TrialStart,
)


def test_logger_writes_typed_events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with ExperimentLogger(path) as log:
        log.log(
            TrialStart(
                trial_id=1,
                model="eqprop_mlp",
                task="xor",
                arm="mlp",
                config={"lr": 1e-3, "beta": 0.2},
                param_count=8970,
                seed=42,
            )
        )
        log.log(Epoch(trial_id=1, epoch=0, metrics={"loss": 0.5, "acc": 0.9}))
        log.log(
            TrialEnd(
                trial_id=1,
                status="completed",
                metrics={"accuracy": 0.95},
                wall_time_s=1.2,
            )
        )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(line) for line in lines]
    assert [e["kind"] for e in events] == ["trial_start", "epoch", "trial_end"]
    assert events[0]["model"] == "eqprop_mlp"
    assert events[0]["config"]["lr"] == 1e-3
    assert events[2]["status"] == "completed"


def test_logger_is_append_only(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for _ in range(2):
        with ExperimentLogger(path) as log:
            log.log(
                GateOutcome(
                    tier="tier0",
                    model="m",
                    task="spiral",
                    passed=True,
                    reason="ok",
                    metrics={},
                )
            )
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_logger_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    with ExperimentLogger(path) as log:
        log.log(
            GateOutcome(
                tier="tier0.5",
                model="m",
                task="digits",
                passed=False,
                reason="digits-fail",
                metrics={"mean_acc": 0.5},
            )
        )
    assert path.exists()


def test_unsupported_event_raises(tmp_path):
    logger = ExperimentLogger(tmp_path / "x.jsonl")
    with pytest.raises(TypeError):
        logger.log({"not": "an event"})
    logger.close()
