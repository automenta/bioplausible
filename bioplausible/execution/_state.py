"""State management for AutoScientist execution.

Consolidates database-adjacent state tracking:
- ExperimentState: Progress queries and Optuna study management
- DecisionLogger: Auditable scientific decision trail
- FailureTracker: Failure logging and pattern analysis
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    from collections.abc import Iterator

from bioplausible.hyperopt.storage import HyperoptStorage

__all__ = [
    "DecisionLogger",
    "ExperimentState",
    "FailureCategory",
    "FailureRecord",
    "FailureTracker",
    "logger",
]
logger = logging.getLogger(__name__)


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Yield a committed sqlite3 connection with ``Row`` row factory.

    Commits on success, rolls back on exception, always closes the
    connection — the single safe entry point for all SQLite access.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


# =============================================================================
# Failure Tracking
# =============================================================================


class FailureCategory(StrEnum):
    CONVERGENCE_FAILURE = "convergence_failure"
    GRADIENT_EXPLOSION = "gradient_explosion"
    SETTLING_DIVERGENCE = "settling_divergence"
    SPECTRAL_INSTABILITY = "spectral_instability"
    MEMORY_OOM = "memory_oom"
    TASK_INCOMPATIBILITY = "task_incompatibility"
    SLOW_CONVERGENCE = "slow_convergence"
    NEGATIVE_TRANSFER = "negative_transfer"
    GOODNESS_COLLAPSE = "goodness_collapse"
    SPIKE_SILENCING = "spike_silencing"


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Record of a single training failure."""

    timestamp: str
    model_name: str
    task_name: str
    tier: str
    trial_id: int | None
    failure_type: str
    failure_epoch: int | None
    failure_batch: int | None
    config: dict[str, object]
    last_metrics: dict[str, object]
    stack_trace: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FailureTracker:
    """Tracks and analyzes training failures persisted to SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with _connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    trial_id INTEGER,
                    failure_type TEXT NOT NULL,
                    failure_epoch INTEGER,
                    failure_batch INTEGER,
                    config TEXT NOT NULL,
                    last_metrics TEXT NOT NULL,
                    stack_trace TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_model ON failures(model_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_type ON failures(failure_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_timestamp"
                " ON failures(timestamp)"
            )

    def log_failure(self, record: FailureRecord) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO failures
                (timestamp, model_name, task_name, tier, trial_id,
                 failure_type, failure_epoch, failure_batch,
                 config, last_metrics, stack_trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.timestamp,
                    record.model_name,
                    record.task_name,
                    record.tier,
                    record.trial_id,
                    record.failure_type,
                    record.failure_epoch,
                    record.failure_batch,
                    json.dumps(record.config),
                    json.dumps(record.last_metrics),
                    record.stack_trace,
                ),
            )
            logger.info(
                "Logged %s failure for %s", record.failure_type, record.model_name
            )

    def get_failure_stats(self, hours: int | None = None) -> dict[str, object]:
        with _connect(self.db_path) as conn:
            cursor = conn.cursor()
            where_clause = ""
            if hours:
                cutoff = datetime.now().isoformat()[:19]
                where_clause = (
                    f"WHERE timestamp >= datetime('{cutoff}', '-{hours} hours')"
                )

            cursor.execute(f"""
                SELECT failure_type, COUNT(*) as count
                FROM failures {where_clause}
                GROUP BY failure_type ORDER BY count DESC
            """)
            by_type = dict(cursor.fetchall())

            cursor.execute(f"""
                SELECT model_name, COUNT(*) as count
                FROM failures {where_clause}
                GROUP BY model_name ORDER BY count DESC LIMIT 10
            """)
            by_model = dict(cursor.fetchall())

            cursor.execute(f"""
                SELECT task_name, COUNT(*) as count
                FROM failures {where_clause}
                GROUP BY task_name ORDER BY count DESC
            """)
            by_task = dict(cursor.fetchall())

            cursor.execute(f"SELECT COUNT(*) FROM failures {where_clause}")
            total_failures = cursor.fetchone()[0]

            return {
                "total_failures": total_failures,
                "by_type": by_type,
                "by_model": by_model,
                "by_task": by_task,
                "time_window_hours": hours,
            }

    def get_recent_failures(self, limit: int = 50) -> list[FailureRecord]:
        with _connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, model_name, task_name, tier, trial_id,
                       failure_type, failure_epoch, failure_batch,
                       config, last_metrics, stack_trace
                FROM failures ORDER BY id DESC LIMIT ?
            """,
                (limit,),
            )
            records = []
            for row in cursor.fetchall():
                records.append(
                    FailureRecord(
                        timestamp=row[0],
                        model_name=row[1],
                        task_name=row[2],
                        tier=row[3],
                        trial_id=row[4],
                        failure_type=row[5],
                        failure_epoch=row[6],
                        failure_batch=row[7],
                        config=json.loads(row[8]),
                        last_metrics=json.loads(row[9]),
                        stack_trace=row[10],
                    )
                )
            return records

    def analyze_failure_patterns(self) -> dict[str, object]:
        stats = self.get_failure_stats()
        recommendations: list[dict[str, object]] = []

        nan_count = stats["by_type"].get("grad_nan", 0) + stats["by_type"].get(
            "loss_nan_or_inf", 0
        )
        pct_nan = (
            nan_count / stats["total_failures"] if stats["total_failures"] > 0 else 0
        )

        if pct_nan > 0.3:
            high_lr_risk = self._check_hyperparam_correlation("lr", "grad_nan")
            msg = "Reduce learning rate ranges"
            if high_lr_risk:
                msg += f" (High LR detected in failures: mean={high_lr_risk:.2e})"
            recommendations.append({
                "issue": "High NaN failure rate",
                "severity": "critical",
                "suggestion": msg,
                "affected_models": list(stats["by_model"].keys())[:3],
            })

        oom_count = stats["by_type"].get("oom", 0)
        if oom_count > 5:
            recommendations.append({
                "issue": "Out of memory errors",
                "severity": "high",
                "suggestion": "Reduce batch size or model size",
                "count": oom_count,
            })

        timeout_count = stats["by_type"].get("timeout", 0)
        if timeout_count > 3:
            recommendations.append({
                "issue": "Frequent timeouts",
                "severity": "high",
                "suggestion": "Reduce model size or iterations",
                "count": timeout_count,
                "affected_models": list(stats["by_model"].keys())[:3],
            })

        recommendations.extend(self._detect_divergence_signatures())

        return {
            "stats": stats,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _check_hyperparam_correlation(
        self, param: str, failure_type: str
    ) -> float | None:
        try:
            with _connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT config FROM failures WHERE failure_type=?",
                    (failure_type,),
                )
                failed_vals = []
                for row in cursor.fetchall():
                    try:
                        cfg = json.loads(row[0])
                        if param in cfg:
                            failed_vals.append(float(cfg[param]))
                    except ValueError, TypeError, json.JSONDecodeError:
                        pass
                if not failed_vals:
                    return None
                return sum(failed_vals) / len(failed_vals)
        except (sqlite3.Error, OSError) as e:
            logger.warning("Correlation check failed: %s", e)
            return None

    def _detect_divergence_signatures(self) -> list[dict[str, object]]:
        recs = []
        try:
            with _connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM failures WHERE failure_epoch < 2")
                res = cursor.fetchone()
                early_fails = res[0] if res else 0

                cursor.execute("SELECT COUNT(*) FROM failures")
                res = cursor.fetchone()
                total = res[0] if res else 0

                if total > 0 and (early_fails / total) > 0.5:
                    recs.append({
                        "issue": "Early Training Instability",
                        "severity": "high",
                        "suggestion": "Check initialization or reduce initial LR",
                        "details": (
                            f"{early_fails}/{total} failures occurred in first 2 epochs"
                        ),
                    })
        except (sqlite3.Error, OSError) as e:
            logger.warning("Divergence check failed: %s", e)
        return recs


# =============================================================================
# Decision Logger
# =============================================================================


class DecisionLogger:
    """Logs high-level scientific decisions to a persistent SQLite database."""

    def __init__(self, db_path: str = "bioplausible.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        try:
            with _connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS decision_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        event_type TEXT,
                        description TEXT,
                        metadata TEXT
                    )
                """)
        except sqlite3.Error:
            logger.exception("Failed to init decision log DB")

    def log_decision(
        self,
        event_type: str,
        description: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            meta_json = json.dumps(metadata) if metadata else "{}"
            with _connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO decision_log"
                    " (timestamp, event_type, description, metadata)"
                    " VALUES (?, ?, ?, ?)",
                    (time.time(), event_type, description, meta_json),
                )
            logger.info("Decision Logged: [%s] %s", event_type, description)
        except sqlite3.Error:
            logger.exception("Failed to log decision")
        except (OSError, ValueError, TypeError) as e:
            logger.error("Unexpected error logging decision: %s", e, exc_info=True)

    def get_log(self, limit: int = 1000) -> list[dict[str, object]]:
        entries = []
        try:
            with _connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM decision_log ORDER BY timestamp ASC LIMIT ?",
                    (limit,),
                )
                for row in cursor.fetchall():
                    entries.append({
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "date_str": datetime.fromtimestamp(row["timestamp"]).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "event_type": row["event_type"],
                        "description": row["description"],
                        "metadata": json.loads(row["metadata"]),
                    })
        except sqlite3.Error:
            logger.exception("Failed to read decision log")
        except (OSError, ValueError, TypeError) as e:
            logger.error("Unexpected error reading decision log: %s", e, exc_info=True)
        return entries


# =============================================================================
# Experiment State
# =============================================================================


class ExperimentState:
    """Analyzes the current state of research by querying the database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.storage = HyperoptStorage(db_path)
        self.failure_tracker = FailureTracker(db_path)

    def get_failure_analysis(self) -> dict[str, object]:
        return self.failure_tracker.analyze_failure_patterns()

    def get_progress(self) -> dict[str, dict[str, dict[str, object]]]:
        trials = self.storage.get_all_trials()
        progress: dict[str, dict[str, dict[str, object]]] = {}

        for t in trials:
            if t.status != "completed":
                continue
            model = t.model_name
            task = t.config.get("task")
            tier_val = t.config.get("tier")

            if not tier_val:
                epochs = t.config.get("epochs")
                if epochs:
                    if epochs <= 3:
                        tier_val = "smoke"
                    elif epochs <= 7:
                        tier_val = "shallow"
                    elif epochs <= 15:
                        tier_val = "standard"
                    else:
                        tier_val = "deep"

            if not task or not tier_val:
                continue

            if model not in progress:
                progress[model] = {}
            if task not in progress[model]:
                progress[model][task] = {}
            if tier_val not in progress[model][task]:
                progress[model][task][tier_val] = {
                    "count": 0,
                    "best_acc": -1.0,
                    "trials": [],
                    "last_run_ts": 0.0,
                }

            entry = progress[model][task][tier_val]
            entry["count"] += 1
            entry["trials"].append(t)
            entry["best_acc"] = max(entry["best_acc"], t.accuracy)

        return progress

    def get_optuna_study(self, study_name: str) -> optuna.Study:
        return optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{self.db_path}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(),
        )

    def get_recent_tasks(self, limit: int = 10) -> list[str]:
        try:
            cursor = self.storage.conn.cursor()
            cursor.execute(
                "SELECT config_json FROM hyperopt_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            recent_tasks = []
            for row in cursor.fetchall():
                try:
                    config = json.loads(row[0])
                    if "task" in config:
                        recent_tasks.append(config["task"])
                except ValueError, TypeError:
                    logger.warning("Failed to deserialize recent task entry")
            return recent_tasks
        except sqlite3.Error, OSError, ValueError:
            logger.exception("Error fetching recent tasks")
            return []

    def get_recent_models(self, limit: int = 10) -> list[str]:
        try:
            cursor = self.storage.conn.cursor()
            cursor.execute(
                "SELECT model_name FROM hyperopt_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error, OSError, ValueError:
            logger.exception("Error fetching recent models")
            return []

    def get_fragile_models(
        self, acc_threshold: float = 0.80, robust_threshold: float = 0.40
    ) -> dict[str, object]:
        fragile_models: dict[str, object] = {}
        try:
            cursor = self.storage.conn.cursor()
            cursor.execute(
                """
                SELECT t.model_name,
                       AVG(t.accuracy) as avg_acc,
                       AVG(CASE WHEN ua.key = 'robustness_score'
                           THEN CAST(ua.value_json as REAL) END) as avg_rob
                FROM hyperopt_logs t
                JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
                WHERE t.status = 'completed'
                GROUP BY t.model_name
                HAVING avg_acc > ? AND avg_rob < ? AND avg_rob > 0
            """,
                (acc_threshold, robust_threshold),
            )
            for row in cursor.fetchall():
                fragile_models[row["model_name"]] = row["avg_rob"]
        except sqlite3.Error, OSError, ValueError:
            logger.warning("Failed to query fragile models")
        return fragile_models

    def close(self) -> None:
        self.storage.close()
