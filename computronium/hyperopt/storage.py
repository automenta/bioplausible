"""
SQLite Storage Backend

Persists trials, configurations, and results to a SQLite database.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from computronium.core.logging import get_logger
from computronium.core.training_state import TRAINING_CHECKPOINTS_DDL, EpochCheckpoint

from .metrics import TrialMetrics

logger = get_logger()

__all__ = [
    "HyperoptStorage",
    "logger",
    "list_studies",
]


class HyperoptStorage:
    """Storage backend for hyperparameter optimization trials."""

    def __init__(self, db_path: str = "results/hyperopt.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable Write-Ahead Logging for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")

        cursor = self.conn.cursor()

        # Trials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperopt_logs (
                trial_id INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                epochs_completed INTEGER DEFAULT 0,
                final_loss REAL,
                accuracy REAL,
                perplexity REAL,
                iteration_time REAL,
                param_count REAL,
                is_pareto INTEGER DEFAULT 0
            )
        """)

        # Training trajectories table (Scientist++ Phase 2)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                convergence_epoch INTEGER,
                converged BOOLEAN,
                overfitting_detected BOOLEAN,
                unstable BOOLEAN,
                FOREIGN KEY (trial_id) REFERENCES hyperopt_logs(trial_id)
            )
        """)

        # Training checkpoints table
        cursor.execute(TRAINING_CHECKPOINTS_DDL)

        # Indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_trajectory"
            " ON training_checkpoints(trajectory_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_epoch"
            " ON training_checkpoints(epoch);"
        )

        self.conn.commit()

    def create_trial(
        self,
        model_name: str,
        config: dict[str, object],
    ) -> int:
        """
        Create a new trial log.

        Args:
            model_name: Name of the model
            config: Configuration dictionary
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO hyperopt_logs (model_name, config_json, status, timestamp)
            VALUES (?, ?, ?, ?)
        """,
            (model_name, json.dumps(config), "pending", datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_trial(
        self,
        trial_id: int,
        status: str = None,
        epochs_completed: int = None,
        final_loss: float = None,
        accuracy: float = None,
        perplexity: float = None,
        iteration_time: float = None,
        param_count: float = None,
    ):
        """Update trial with results."""
        updates = []
        values = []

        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if epochs_completed is not None:
            updates.append("epochs_completed = ?")
            values.append(epochs_completed)
        if final_loss is not None:
            updates.append("final_loss = ?")
            values.append(final_loss)
        if accuracy is not None:
            updates.append("accuracy = ?")
            values.append(accuracy)
        if perplexity is not None:
            updates.append("perplexity = ?")
            values.append(perplexity)
        if iteration_time is not None:
            updates.append("iteration_time = ?")
            values.append(iteration_time)
        if param_count is not None:
            updates.append("param_count = ?")
            values.append(param_count)

        if updates:
            values.append(trial_id)
            set_clause = ", ".join(updates)
            query = "UPDATE hyperopt_logs SET " + set_clause + " WHERE trial_id = ?"
            self.conn.execute(query, values)
            self.conn.commit()

    def log_epoch(
        self,
        trial_id: int,
        epoch: int,
        loss: float,
        accuracy: float,
        perplexity: float,
        time: float,
        *,
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        grad_norm_mean: float = 0.0,
        grad_norm_std: float = 0.0,
        weight_norm: float = 0.0,
        learning_rate: float = 0.0,
        train_val_gap: float = 0.0,
        test_acc: float | None = None,
        reward: float | None = None,
        wall_time_seconds: float = 0.0,
        total_flops: int | None = None,
        samples_seen: int = 0,
    ) -> None:
        """Log metrics for a specific epoch using EpochCheckpoint schema.

        Args:
            trial_id: Trial identifier.
            epoch: Epoch number.
            loss: Training loss (maps to val_loss for backward compatibility).
            accuracy: Validation accuracy (maps to val_acc for backward compatibility).
            perplexity: Perplexity metric.
            time: Time for this epoch.
            train_acc: Training accuracy (new, defaults to 0.0).
            val_acc: Validation accuracy (new, defaults to accuracy if provided).
            train_loss: Training loss (new, defaults to loss).
            val_loss: Validation loss (new, defaults to loss).
            grad_norm_mean: Mean gradient norm.
            grad_norm_std: Std of gradient norm.
            weight_norm: Weight norm.
            learning_rate: Current learning rate.
            train_val_gap: Train-val accuracy gap.
            test_acc: Test accuracy (optional).
            reward: Reward for RL tasks (optional).
            wall_time_seconds: Cumulative wall time.
            total_flops: Estimated FLOPs.
            samples_seen: Total samples processed.
        """
        train_acc_eff = train_acc or accuracy
        val_acc_eff = val_acc or accuracy
        ckpt = EpochCheckpoint(
            epoch=epoch,
            train_acc=train_acc_eff,
            val_acc=val_acc_eff,
            train_loss=train_loss or loss,
            val_loss=val_loss or loss,
            grad_norm_mean=grad_norm_mean,
            grad_norm_std=grad_norm_std,
            weight_norm=weight_norm,
            learning_rate=learning_rate,
            train_val_gap=train_val_gap or (train_acc_eff - val_acc_eff),
            test_acc=test_acc,
            perplexity=perplexity or None,
            reward=reward,
            wall_time_seconds=wall_time_seconds or time,
            total_flops=total_flops,
            samples_seen=samples_seen,
        )

        self.conn.execute(
            """
            INSERT INTO training_checkpoints (
                trial_id, trajectory_id, epoch, train_acc, val_acc, train_loss,
                val_loss, grad_norm_mean, grad_norm_std, weight_norm,
                learning_rate, train_val_gap, test_acc, perplexity, reward,
                wall_time_seconds, total_flops, samples_seen
            ) VALUES (?, -1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trial_id,
                ckpt.epoch,
                ckpt.train_acc,
                ckpt.val_acc,
                ckpt.train_loss,
                ckpt.val_loss,
                ckpt.grad_norm_mean,
                ckpt.grad_norm_std,
                ckpt.weight_norm,
                ckpt.learning_rate,
                ckpt.train_val_gap,
                ckpt.test_acc,
                ckpt.perplexity,
                ckpt.reward,
                ckpt.wall_time_seconds,
                ckpt.total_flops,
                ckpt.samples_seen,
            ),
        )
        self.conn.commit()

    def get_trial(self, trial_id: int) -> TrialMetrics | None:
        """Retrieve a trial by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM hyperopt_logs WHERE trial_id = ?", (trial_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return TrialMetrics(
            trial_id=row["trial_id"],
            model_name=row["model_name"],
            config=json.loads(row["config_json"]),
            accuracy=row["accuracy"] or 0.0,
            perplexity=row["perplexity"] or 10.0,
            iteration_time=row["iteration_time"] or 1.0,
            param_count=row["param_count"] or 1.0,
            epochs_completed=row["epochs_completed"] or 0,
            final_loss=row["final_loss"] or 10.0,
            status=row["status"],
        )

    def get_all_trials(
        self, model_name: str = None, status: str = None
    ) -> list[TrialMetrics]:
        """Retrieve all trials, optionally filtered."""
        query = "SELECT * FROM hyperopt_logs WHERE 1=1"
        params = []

        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        if status is not None:
            query += " AND status = ?"
            params.append(status)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        trials = []
        for row in rows:
            trials.append(
                TrialMetrics(
                    trial_id=row["trial_id"],
                    model_name=row["model_name"],
                    config=json.loads(row["config_json"]),
                    accuracy=row["accuracy"] or 0.0,
                    perplexity=row["perplexity"] or 10.0,
                    iteration_time=row["iteration_time"] or 1.0,
                    param_count=row["param_count"] or 1.0,
                    epochs_completed=row["epochs_completed"] or 0,
                    final_loss=row["final_loss"] or 10.0,
                    status=row["status"],
                )
            )

        return trials

    def mark_pareto_frontier(self, trial_ids: list[int]):
        """Mark trials as being on the Pareto frontier."""
        # Clear previous frontier
        self.conn.execute("UPDATE hyperopt_logs SET is_pareto = 0")

        # Mark new frontier
        if trial_ids:
            placeholders = ",".join("?" * len(trial_ids))
            self.conn.execute(
                f"UPDATE hyperopt_logs SET is_pareto = 1"
                f" WHERE trial_id IN ({placeholders})",
                trial_ids,
            )

        self.conn.commit()

    def clear_all_trials(self):
        """Clear all trials and associated checkpoints from the database."""
        cursor = self.conn.cursor()

        # Clear derived rows first (due to foreign key constraints)
        cursor.execute("DELETE FROM training_checkpoints")
        cursor.execute("DELETE FROM training_trajectories")

        # Clear trials
        cursor.execute("DELETE FROM hyperopt_logs")

        self.conn.commit()

    def save_trajectory(self, trajectory):
        """
        Save a full TrainingTrajectory and its checkpoints.

        Args:
            trajectory: TrainingTrajectory object
                (from computronium.execution.training_dynamics)
        """
        try:
            cursor = self.conn.cursor()

            # Insert Trajectory
            cursor.execute(
                """
                INSERT INTO training_trajectories (
                    trial_id, model_name, task_name, config_json,
                    convergence_epoch, converged, overfitting_detected, unstable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trajectory.trial_id,
                    trajectory.model_name,
                    trajectory.task_name,
                    json.dumps(trajectory.config),
                    trajectory.convergence_epoch,
                    int(trajectory.converged),
                    int(trajectory.overfitting_detected),
                    int(trajectory.unstable),
                ),
            )

            trajectory_id = cursor.lastrowid

            # Bulk Insert Checkpoints
            checkpoints_data = []
            for ckpt in trajectory.checkpoints:
                checkpoints_data.append((
                    trajectory_id,
                    ckpt.epoch,
                    ckpt.train_acc,
                    ckpt.val_acc,
                    ckpt.test_acc,
                    ckpt.train_loss,
                    ckpt.val_loss,
                    ckpt.grad_norm_mean,
                    ckpt.grad_norm_std,
                    ckpt.weight_norm,
                    ckpt.learning_rate,
                    ckpt.train_val_gap,
                    ckpt.perplexity,
                    ckpt.reward,
                    ckpt.wall_time_seconds,
                    ckpt.total_flops,
                    ckpt.samples_seen,
                ))

            cursor.executemany(
                """
                INSERT INTO training_checkpoints (
                    trajectory_id, epoch, train_acc, val_acc, test_acc,
                    train_loss, val_loss, grad_norm_mean, grad_norm_std,
                    weight_norm, learning_rate, train_val_gap, perplexity,
                    reward, wall_time_seconds, total_flops, samples_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                checkpoints_data,
            )

            self.conn.commit()

        except sqlite3.Error:
            logger.exception("Database error saving trajectory")
            # Don't crash training if logging fails, but maybe re-raise?
            # For now, just log error.

    def get_all_trajectories(self):
        """
        Retrieve all training trajectories with their checkpoints.
        Returns: List[TrainingTrajectory]
        """
        from computronium.core.training_state import (
            EpochCheckpoint,
            TrainingTrajectory,
        )

        cursor = self.conn.cursor()

        # 1. Get all trajectories
        cursor.execute("""
            SELECT id, trial_id, model_name, task_name, config_json,
                   convergence_epoch, converged, overfitting_detected, unstable
            FROM training_trajectories
        """)
        rows = cursor.fetchall()

        trajectories = []
        for row in rows:
            traj_id = row["id"]

            cursor.execute(
                """
                SELECT *
                FROM training_checkpoints
                WHERE trajectory_id = ?
                ORDER BY epoch ASC
            """,
                (traj_id,),
            )
            ckpt_rows = cursor.fetchall()

            checkpoints = []
            for cr in ckpt_rows:
                checkpoints.append(
                    EpochCheckpoint(
                        epoch=cr["epoch"],
                        train_acc=cr["train_acc"],
                        val_acc=cr["val_acc"],
                        test_acc=cr["test_acc"],
                        train_loss=cr["train_loss"],
                        val_loss=cr["val_loss"],
                        grad_norm_mean=cr["grad_norm_mean"],
                        grad_norm_std=cr["grad_norm_std"],
                        weight_norm=cr["weight_norm"],
                        learning_rate=cr["learning_rate"],
                        train_val_gap=cr["train_val_gap"],
                        perplexity=cr["perplexity"],
                        reward=cr["reward"],
                        wall_time_seconds=cr["wall_time_seconds"],
                        total_flops=cr["total_flops"],
                        samples_seen=cr["samples_seen"],
                    )
                )

            traj = TrainingTrajectory(
                trial_id=row["trial_id"],
                model_name=row["model_name"],
                task_name=row["task_name"],
                config=json.loads(row["config_json"]),
                checkpoints=checkpoints,
            )
            # Set computed/stored fields
            traj.convergence_epoch = row["convergence_epoch"]
            traj.converged = bool(row["converged"])
            traj.overfitting_detected = bool(row["overfitting_detected"])
            traj.unstable = bool(row["unstable"])

            trajectories.append(traj)

        return trajectories

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def list_studies(storage_url: str) -> list[str]:
    """List all Optuna study names in the given storage."""
    import optuna

    storage = optuna.storages.RDBStorage(url=storage_url)
    return storage.get_all_study_names()
