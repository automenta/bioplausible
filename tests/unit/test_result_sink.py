"""Tests for the universal experiment result sink (KnowledgeBase + FailureTracker)."""

import os
import tempfile

import pytest

from bioplausible.experiment.result_sink import configure, record_experiment_result


@pytest.fixture
def sink_paths():
    """Point the sink at temporary DBs and reset its cached instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = os.path.join(tmpdir, "kb.db")
        fail = os.path.join(tmpdir, "fail.db")
        configure(kb_path=kb, failure_path=fail)
        yield kb, fail
        configure()  # reset to defaults


def test_record_success_writes_knowledge_entry(sink_paths):
    """A completed probe persists a verified entry to the KnowledgeBase."""
    kb_path, _ = sink_paths
    eid = record_experiment_result(
        model="eqprop",
        task="mnist",
        config={"lr": 0.01, "max_steps": 20},
        metrics={"final_acc": 0.92, "forward_flops": 100, "backward_flops": 50,
                 "peak_memory_mb": 20.0, "wall_time_s": 3.0},
        status="completed",
        seed=1,
        epochs=5,
    )
    assert eid.startswith("EXP-")
    from bioplausible.knowledge.kb import KnowledgeBase

    kb = KnowledgeBase(db_path=kb_path, auto_embed=False)
    hits = [e for e in kb.query(model_family="eqprop") if e.id == eid]
    assert len(hits) == 1
    assert hits[0].metrics["final_acc"] == pytest.approx(0.92)


def test_record_failure_writes_failure_tracker(sink_paths):
    """A failed experiment persists a negative record to the FailureTracker."""
    _, fail_path = sink_paths
    result = record_experiment_result(
        model="pepita",
        task="mnist",
        config={"lr": 99.0},
        status="error",
        extra={"error": "diverged"},
        seed=2,
    )
    assert result.startswith("FAIL:")
    from bioplausible.execution._state import FailureTracker

    ft = FailureTracker(db_path=fail_path)
    stats = ft.get_failure_stats()
    assert stats["total_failures"] == 1
    assert "exception" in stats["by_type"]


def test_success_and_failure_use_distinct_sinks(sink_paths):
    """Successes never land in FailureTracker and failures never land in KB."""
    kb_path, fail_path = sink_paths
    record_experiment_result(model="ff", task="mnist", metrics={"accuracy": 0.5},
                             status="completed", seed=3)
    record_experiment_result(model="ff", task="mnist", status="failed", seed=4)

    from bioplausible.execution._state import FailureTracker
    from bioplausible.knowledge.kb import KnowledgeBase

    ft = FailureTracker(db_path=fail_path)
    assert ft.get_failure_stats()["total_failures"] == 1

    kb = KnowledgeBase(db_path=kb_path, auto_embed=False)
    exp = [e for e in kb.query(model_family="ff") if e.id.startswith("EXP")]
    assert len(exp) == 1  # only the success
