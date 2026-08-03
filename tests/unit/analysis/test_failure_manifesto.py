"""Sprint 2.4 — failure manifesto generator + ``biopl-failure-manifesto`` CLI."""

import sqlite3
from datetime import datetime

from bioplausible.analysis.failure_manifesto import FailureManifestoGenerator, main
from bioplausible.execution._state import FailureCategory, FailureRecord, FailureTracker


def _rec(model: str, task: str, ftype: str, epoch: int) -> FailureRecord:
    return FailureRecord(
        timestamp=datetime.now().isoformat(),
        model_name=model,
        task_name=task,
        tier="unit",
        trial_id=None,
        failure_type=ftype,
        failure_epoch=epoch,
        failure_batch=None,
        config={},
        last_metrics={},
    )


def _seed_db(path: str) -> None:
    tracker = FailureTracker(path)
    tracker.log_failure(
        _rec("eqprop_mlp", "mnist", FailureCategory.CONVERGENCE_FAILURE, 3)
    )
    tracker.log_failure(
        _rec("eqprop_mlp", "mnist", FailureCategory.GRADIENT_EXPLOSION, 7)
    )
    tracker.log_failure(
        _rec("backprop_mlp", "digits", FailureCategory.CONVERGENCE_FAILURE, 1)
    )


def test_generate_writes_markdown_with_failures(tmp_path):
    db = tmp_path / "exp.db"
    out = tmp_path / "report.md"
    _seed_db(str(db))
    gen = FailureManifestoGenerator(str(db))
    result = gen.generate(str(out))
    assert result == str(out)
    text = out.read_text()
    assert "# Failure Modes Manifesto" in text
    assert "eqprop_mlp" in text
    assert "GRADIENT_EXPLOSION" in text.capitalize() or "gradient_explosion" in text


def test_generate_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    out = tmp_path / "report.md"
    sqlite3.connect(str(db)).close()
    gen = FailureManifestoGenerator(str(db))
    gen.generate(str(out))
    assert "No failures logged yet." in out.read_text()


def test_generate_model_filter(tmp_path):
    db = tmp_path / "exp.db"
    out = tmp_path / "report.md"
    _seed_db(str(db))
    gen = FailureManifestoGenerator(str(db))
    gen.generate(str(out), model="eqprop_mlp")
    text = out.read_text()
    assert "### Scope: `eqprop_mlp`" in text
    assert "backprop_mlp" not in text


def test_cli_main_returns_zero(tmp_path):
    db = tmp_path / "exp.db"
    out = tmp_path / "report.md"
    _seed_db(str(db))
    code = main(["--db", str(db), "--model", "eqprop_mlp", "--output", str(out)])
    assert code == 0
    assert out.exists()
