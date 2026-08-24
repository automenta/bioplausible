"""Tests for ResearchSynthesizer."""

import pathlib
import sqlite3
import tempfile

import pytest

from computronium.execution.synthesizer import ResearchSynthesizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_db_path() -> str:
    """Create a temporary SQLite DB with Optuna-compatible schema + test data."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    _create_schema(tmp.name)
    _populate_test_data(tmp.name)
    yield tmp.name
    pathlib.Path(tmp.name).unlink()


def _create_schema(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name VARCHAR(512))"
    )
    conn.execute(
        "CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, study_id INTEGER, state VARCHAR(16))"
    )
    conn.execute(
        "CREATE TABLE trial_values (trial_value_id INTEGER PRIMARY KEY, trial_id INTEGER, value FLOAT)"
    )
    conn.execute(
        "CREATE TABLE trial_user_attributes (trial_user_attribute_id INTEGER PRIMARY KEY, trial_id INTEGER, key VARCHAR(512), value_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE trial_params (param_id INTEGER PRIMARY KEY, trial_id INTEGER, param_name VARCHAR(512), param_value FLOAT)"
    )
    conn.execute(
        "CREATE TABLE hyperopt_logs (log_id INTEGER PRIMARY KEY, trial_id INTEGER, param_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE failures (failure_id INTEGER PRIMARY KEY, trial_id INTEGER, model_name TEXT, task_name TEXT, failure_type TEXT, config TEXT)"
    )
    conn.execute(
        "CREATE TABLE training_trajectories (id INTEGER PRIMARY KEY, trial_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE training_checkpoints (id INTEGER PRIMARY KEY, trajectory_id INTEGER, epoch INTEGER, train_loss FLOAT, train_acc FLOAT, val_acc FLOAT, samples_seen INTEGER)"
    )
    conn.commit()
    conn.close()


def _populate_test_data(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO studies VALUES (1, 'eqprop_mnist_standard')")
    conn.execute("INSERT INTO studies VALUES (2, 'backprop_mnist_standard')")

    # Trial 1: eqprop, complete, high accuracy
    conn.execute("INSERT INTO trials VALUES (1, 1, 'COMPLETE')")
    conn.execute("INSERT INTO trials VALUES (2, 1, 'COMPLETE')")
    conn.execute("INSERT INTO trials VALUES (3, 2, 'COMPLETE')")
    conn.execute("INSERT INTO trials VALUES (4, 1, 'FAIL')")

    conn.execute("INSERT INTO trial_values VALUES (1, 1, 0.95)")
    conn.execute("INSERT INTO trial_values VALUES (2, 2, 0.92)")
    conn.execute("INSERT INTO trial_values VALUES (3, 3, 0.97)")

    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (1, 1, 'model_name', '\"eqprop\"')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (2, 1, 'task_name', '\"mnist\"')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (3, 1, 'param_count', '100000')"
    )
    conn.execute("INSERT INTO trial_user_attributes VALUES (4, 1, 'num_epochs', '50')")
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (5, 2, 'model_name', '\"eqprop\"')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (6, 2, 'task_name', '\"mnist\"')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (7, 2, 'param_count', '200000')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (8, 3, 'model_name', '\"backprop\"')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (9, 3, 'task_name', '\"mnist\"')"
    )
    conn.execute(
        "INSERT INTO trial_user_attributes VALUES (10, 3, 'param_count', '50000')"
    )

    conn.execute("INSERT INTO trial_params VALUES (1, 1, 'lr', 0.001)")
    conn.execute("INSERT INTO trial_params VALUES (2, 1, 'hidden_dim', 256)")
    conn.execute("INSERT INTO trial_params VALUES (3, 1, 'num_layers', 2)")
    conn.execute("INSERT INTO trial_params VALUES (4, 2, 'lr', 0.0005)")
    conn.execute("INSERT INTO trial_params VALUES (5, 2, 'hidden_dim', 128)")
    conn.execute("INSERT INTO trial_params VALUES (6, 2, 'num_layers', 3)")
    conn.execute("INSERT INTO trial_params VALUES (7, 3, 'lr', 0.01)")
    conn.execute("INSERT INTO trial_params VALUES (8, 3, 'hidden_dim', 512)")
    conn.execute("INSERT INTO trial_params VALUES (9, 3, 'num_layers', 4)")

    conn.execute("INSERT INTO hyperopt_logs VALUES (1, 1, 100000)")
    conn.execute("INSERT INTO hyperopt_logs VALUES (2, 2, 200000)")
    conn.execute("INSERT INTO hyperopt_logs VALUES (3, 3, 50000)")

    conn.execute(
        "INSERT INTO failures VALUES (1, 4, 'eqprop', 'mnist', 'DIVERGED', '{\"lr\": 0.1}')"
    )

    conn.execute("INSERT INTO training_trajectories VALUES (1, 1)")
    conn.execute(
        "INSERT INTO training_checkpoints VALUES (1, 1, 1, 0.5, 0.8, 0.85, 1000)"
    )
    conn.execute(
        "INSERT INTO training_checkpoints VALUES (2, 1, 5, 0.1, 0.95, 0.95, 5000)"
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_creation(synth_db_path: str) -> None:
    """ResearchSynthesizer accepts a db_path."""
    synth = ResearchSynthesizer(synth_db_path)
    assert synth.db_path == synth_db_path


# ---------------------------------------------------------------------------
# synthesize_full_report
# ---------------------------------------------------------------------------


def test_synthesize_full_report_returns_all_keys(synth_db_path: str) -> None:
    """synthesize_full_report returns a dict with all expected keys."""
    synth = ResearchSynthesizer(synth_db_path)
    report = synth.synthesize_full_report()
    expected_keys = [
        "cross_algorithm_insights",
        "task_specific_winners",
        "efficiency_analysis",
        "backprop_gap_analysis",
        "ablation_analysis",
        "statistical_significance",
        "failure_analysis",
        "quick_wins",
        "research_gaps",
    ]
    for key in expected_keys:
        assert key in report, f"Missing key: {key}"


def test_synthesize_full_report_cross_algorithm(synth_db_path: str) -> None:
    """Cross-algorithm insights includes rankings."""
    synth = ResearchSynthesizer(synth_db_path)
    report = synth.synthesize_full_report()
    cross = report["cross_algorithm_insights"]
    assert "rankings" in cross
    assert len(cross["rankings"]) >= 1


def test_synthesize_full_report_task_winners(synth_db_path: str) -> None:
    """Task-specific winners includes mnist."""
    synth = ResearchSynthesizer(synth_db_path)
    report = synth.synthesize_full_report()
    winners = report["task_specific_winners"]
    assert "mnist" in winners


def test_synthesize_full_report_failure_analysis(synth_db_path: str) -> None:
    """Failure analysis includes counts and patterns."""
    synth = ResearchSynthesizer(synth_db_path)
    report = synth.synthesize_full_report()
    failures = report["failure_analysis"]
    assert isinstance(failures, dict)
    assert "counts" in failures
    assert failures["counts"].get("DIVERGED", 0) >= 1


def test_synthesize_full_report_research_gaps(synth_db_path: str) -> None:
    """Research gaps returns a non-empty list."""
    synth = ResearchSynthesizer(synth_db_path)
    report = synth.synthesize_full_report()
    assert len(report["research_gaps"]) >= 1


# ---------------------------------------------------------------------------
# _get_trials_df
# ---------------------------------------------------------------------------


def test_get_trials_df_returns_dataframe(synth_db_path: str) -> None:
    """_get_trials_df returns a DataFrame with expected columns."""
    synth = ResearchSynthesizer(synth_db_path)
    import sqlite3

    conn = sqlite3.connect(synth_db_path)
    df = synth._get_trials_df(conn)
    conn.close()

    assert not df.empty
    assert "trial_id" in df.columns
    assert "model_name" in df.columns
    assert "accuracy" in df.columns


def test_get_trials_df_filters_incomplete(synth_db_path: str) -> None:
    """_get_trials_df only includes COMPLETE trials."""
    synth = ResearchSynthesizer(synth_db_path)
    import sqlite3

    conn = sqlite3.connect(synth_db_path)
    df = synth._get_trials_df(conn)
    conn.close()

    # Trial 4 is FAIL, should not appear
    assert 4 not in df["trial_id"].values


# ---------------------------------------------------------------------------
# _get_trials_df with empty DB
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_db_path() -> str:
    """A database with correct schema but no data."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    _create_schema(tmp.name)
    yield tmp.name
    pathlib.Path(tmp.name).unlink()


def test_get_trials_df_empty(empty_db_path: str) -> None:
    """_get_trials_df returns empty DataFrame for empty DB."""
    synth = ResearchSynthesizer(empty_db_path)
    import sqlite3

    conn = sqlite3.connect(empty_db_path)
    df = synth._get_trials_df(conn)
    conn.close()
    assert df.empty


def test_synthesize_full_report_empty_db(empty_db_path: str) -> None:
    """synthesize_full_report handles empty DB without crashing."""
    synth = ResearchSynthesizer(empty_db_path)
    report = synth.synthesize_full_report()
    # Should not raise, returns dict with all keys
    assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# _estimate_param_count
# ---------------------------------------------------------------------------


def test_estimate_param_count_basic() -> None:
    """_estimate_param_count computes reasonable value."""
    synth = ResearchSynthesizer(":memory:")

    class FakeRow(dict):
        hidden_dim = 256
        num_layers = 3

    row = FakeRow({"hidden_dim": 256, "num_layers": 3})
    count = synth._estimate_param_count(row)
    # l * (h * h) + (h * 10) = 3 * (256*256) + (256*10) = 196608 + 2560 = 199168
    assert count == 199168


# ---------------------------------------------------------------------------
# Declarative __all__
# ---------------------------------------------------------------------------


def test_module_all() -> None:
    """Module exports ResearchSynthesizer."""
    from computronium.execution import synthesizer

    assert hasattr(synthesizer, "ResearchSynthesizer")
