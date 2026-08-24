"""Shared training-state containers (REFACTOR.md §4).

Unifies the epoch-level checkpoint/trajectory types previously scattered
across ``computronium.execution.training_dynamics`` (``TrainingCheckpoint``,
``TrainingTrajectory``) and ``computronium.hyperopt.storage`` (trial epoch
metrics). ``EpochCheckpoint`` is the canonical epoch-level record; both the
execution path and the hyperopt persistence layer build on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TRAINING_CHECKPOINTS_DDL",
    "EpochCheckpoint",
    "TrainingTrajectory",
]

TRAINING_CHECKPOINTS_DDL = """
    CREATE TABLE IF NOT EXISTS training_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trial_id INTEGER,
        trajectory_id INTEGER NOT NULL DEFAULT -1,
        epoch INTEGER NOT NULL,
        train_acc REAL,
        val_acc REAL,
        test_acc REAL,
        train_loss REAL,
        val_loss REAL,
        grad_norm_mean REAL,
        grad_norm_std REAL,
        weight_norm REAL,
        learning_rate REAL,
        train_val_gap REAL,
        perplexity REAL,
        reward REAL,
        wall_time_seconds REAL,
        total_flops INTEGER,
        samples_seen INTEGER DEFAULT 0,
        FOREIGN KEY (trajectory_id) REFERENCES training_trajectories(id)
    )
"""


@dataclass(frozen=True, slots=True)
class EpochCheckpoint:
    """Metrics captured at a single checkpoint during training.

    Attributes:
        epoch: Epoch number.
        train_acc: Training accuracy.
        val_acc: Validation accuracy.
        train_loss: Training loss.
        val_loss: Validation loss.
        grad_norm_mean: Mean gradient norm.
        grad_norm_std: Standard deviation of gradient norm.
        weight_norm: L2 norm of weights.
        learning_rate: Current learning rate.
        train_val_gap: Difference between training and validation accuracy.
        test_acc: Test accuracy (optional).
        perplexity: Perplexity for LM tasks (optional).
        reward: Reward for RL tasks (optional).
        wall_time_seconds: Cumulative training time.
        total_flops: Estimated total FLOPs used.
        samples_seen: Total samples processed.
    """

    epoch: int
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float

    # Training dynamics
    grad_norm_mean: float = 0.0
    grad_norm_std: float = 0.0
    weight_norm: float = 0.0
    learning_rate: float = 0.0  # May decay over time

    # Overfitting indicators
    train_val_gap: float = 0.0  # train_acc - val_acc

    # Task-specific metrics
    test_acc: float | None = None
    perplexity: float | None = None  # For LM tasks
    reward: float | None = None  # For RL tasks

    # Efficiency metrics
    wall_time_seconds: float = 0.0
    total_flops: int | None = None
    samples_seen: int = 0


@dataclass(slots=True)
class TrainingTrajectory:
    """Complete training history for one trial.

    Attributes:
        trial_id: Unique trial identifier.
        model_name: Name of the model.
        task_name: Name of the task.
        config: Hyperparameter configuration.
        checkpoints: List of recorded checkpoints.
        convergence_epoch: Epoch where convergence was detected.
        converged: Whether the training converged successfully.
        overfitting_detected: Whether overfitting was detected.
        unstable: Whether training was unstable (high variance).
    """

    trial_id: int
    model_name: str
    task_name: str
    config: dict[str, object]
    checkpoints: list[EpochCheckpoint] = field(default_factory=list)

    # Derived metrics (computed from checkpoints)
    convergence_epoch: int | None = None  # Epoch where improvement plateaus
    converged: bool = False
    overfitting_detected: bool = False
    unstable: bool = False  # Large loss variance

    def compute_convergence_speed(self) -> float:
        """Calculate epochs to reach 90% of final accuracy.

        Returns:
            float: Epoch number, or infinity if never reached.
        """
        if not self.checkpoints:
            return float("inf")

        final_acc = self.checkpoints[-1].val_acc
        target_acc = 0.9 * final_acc

        for ckpt in self.checkpoints:
            # We assume checkpoints are sorted by epoch
            if ckpt.val_acc >= target_acc:
                return float(ckpt.epoch)

        return float("inf")

    def compute_sample_efficiency(self) -> float:
        """Compute Area Under the Learning Curve (AUC).

        Higher AUC indicates faster learning with fewer samples.

        Returns:
            float: Normalized AUC score.
        """
        if not self.checkpoints:
            return 0.0

        epochs = [c.epoch for c in self.checkpoints]
        accs = [c.val_acc for c in self.checkpoints]

        if epochs[-1] == 0:
            return 0.0

        # Manual trapezoidal integration (faster and avoids version-dependent API)
        area = 0.0
        for i in range(len(epochs) - 1):
            width = epochs[i + 1] - epochs[i]
            height = (accs[i + 1] + accs[i]) / 2.0
            area += width * height

        return float(area) / epochs[-1]

    def detect_overfitting(self, threshold: float = 0.1) -> bool:
        """Check if training/validation gap exceeds threshold.

        Args:
            threshold: Gap threshold (default 0.1).

        Returns:
            bool: True if overfitting detected.
        """
        min_checkpoints = 2
        if len(self.checkpoints) < min_checkpoints:
            return False

        # Check last 3 checkpoints or fewer if not available
        check_count = min(3, len(self.checkpoints))
        recent_gaps = [c.train_val_gap for c in self.checkpoints[-check_count:]]
        return any(gap > threshold for gap in recent_gaps)
