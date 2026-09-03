"""
Training Dynamics Analysis.

Captures detailed metrics during training (gradients, weight norms, loss curves)
to analyze convergence behavior, detect overfitting, and measure sample efficiency.
"""

from typing import TYPE_CHECKING

import numpy as np

from computronium.core.logging import get_logger
from computronium.core.training_state import EpochCheckpoint as TrainingCheckpoint
from computronium.core.training_state import TrainingTrajectory

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger()

__all__ = [
    "ContinuousTrainingSchedule",
    "TrainingCheckpoint",
    "TrainingTrajectory",
    "logger",
]


class ContinuousTrainingSchedule:
    """
    Manages progressive training with adaptive checkpointing.
    """

    # Standard checkpoints (logarithmic-ish scale)
    DEFAULT_CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]  # ruff: ignore[mutable-class-default]

    def __init__(self, max_epochs: int = 100, enable_pruning: bool = True) -> None:
        """
        Initialize the training schedule.

        Args:
            max_epochs: Maximum epochs to train.
            enable_pruning: Whether to allow early stopping/pruning.
        """
        self.max_epochs = max_epochs
        self.enable_pruning = enable_pruning
        # Filter checkpoints that are beyond max_epochs
        self.checkpoints = [c for c in self.DEFAULT_CHECKPOINTS if c < max_epochs]
        # Always include the final epoch
        if not self.checkpoints or self.checkpoints[-1] != max_epochs:
            self.checkpoints.append(max_epochs)

    def train_with_checkpoints(
        self,
        trainer: object,  # The training object (e.g. from computronium.training.Trainer)
        trial_id: int,
        model_name: str,
        task_name: str,
        config: dict[str, object],
        optuna_trial: object | None = None,
        pruning_callback: Callable[[int, int, dict[str, float]], bool] | None = None,
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    ) -> TrainingTrajectory:
        """
        Train model with periodic evaluation at checkpoints.

        Args:
            trainer: Object with train_epoch() method returning metrics.
            trial_id: ID of the trial.
            model_name: Name of the model.
            task_name: Name of the task.
            config: Configuration dict.
            optuna_trial: Optional Optuna trial object.
            pruning_callback: Optional callback(trial_id, epoch, metrics) -> bool.
            on_epoch_end: Optional callback(epoch, metrics) -> None.

        Returns:
            Completed TrainingTrajectory.
        """
        trajectory = TrainingTrajectory(
            trial_id=trial_id,
            model_name=model_name,
            task_name=task_name,
            config=config,
            checkpoints=[],
        )

        current_epoch = 0
        cumulative_time = 0.0

        for target_epoch in self.checkpoints:
            epochs_to_run = target_epoch - current_epoch

            if epochs_to_run <= 0:
                continue

            chunk_metrics: list[dict[str, float]] = []
            for _ in range(epochs_to_run):
                # Run one epoch
                m = trainer.train_epoch()
                chunk_metrics.append(m)
                cumulative_time += m.get("time", 0.0)
                current_epoch += 1

                if on_epoch_end:
                    on_epoch_end(current_epoch, m)

            # Use metrics from the LAST epoch of this chunk for the checkpoint
            last_metrics = chunk_metrics[-1]

            # Compute train_val_gap
            t_acc = last_metrics.get("train_acc", 0.0)
            # Standard trainer uses 'accuracy' for validation
            v_acc = last_metrics.get("accuracy", 0.0)
            if "val_acc" in last_metrics:
                v_acc = last_metrics["val_acc"]

            gap = t_acc - v_acc

            # Placeholder for grad norms if not provided
            g_norm = last_metrics.get("grad_norm", 0.0)
            w_norm = last_metrics.get("weight_norm", 0.0)

            ckpt = TrainingCheckpoint(
                epoch=target_epoch,
                train_acc=t_acc,
                val_acc=v_acc,
                train_loss=last_metrics.get(
                    "train_loss", last_metrics.get("loss", 0.0)
                ),
                val_loss=last_metrics.get(
                    "val_loss", last_metrics.get("loss", 0.0)
                ),  # Sometimes same if no separate val set
                grad_norm_mean=g_norm,
                grad_norm_std=0.0,  # Would need collection during epoch
                weight_norm=w_norm,
                learning_rate=config.get("lr", 0.0),  # Simplification
                train_val_gap=gap,
                perplexity=last_metrics.get("perplexity"),  # type: ignore[unknown]
                reward=last_metrics.get("reward"),  # type: ignore[unknown]
                wall_time_seconds=cumulative_time,
                samples_seen=int(last_metrics.get("samples_seen", 0)),
            )

            trajectory.checkpoints.append(ckpt)

            # Pruning integration
            if optuna_trial and self.enable_pruning:
                optuna_trial.report(v_acc, target_epoch)
                if optuna_trial.should_prune():
                    trajectory.converged = False
                    logger.info(
                        "[PRUNE]  Trial %s PRUNED at epoch %s", trial_id, target_epoch
                    )
                    break

            elif pruning_callback and self.enable_pruning:
                # Use generic callback if provided
                if pruning_callback(trial_id, target_epoch, last_metrics):
                    trajectory.converged = False
                    logger.info(
                        "[PRUNE]  Trial %s PRUNED at epoch %s", trial_id, target_epoch
                    )
                    break

        # Post-training analysis
        trajectory.convergence_epoch = self._find_convergence(trajectory)
        trajectory.overfitting_detected = trajectory.detect_overfitting()
        trajectory.unstable = self._check_stability(trajectory)
        trajectory.converged = not trajectory.unstable

        return trajectory

    def _find_convergence(self, trajectory: TrainingTrajectory) -> int | None:
        """
        Detect convergence point (where improvement plateaus).

        Args:
            trajectory: The training trajectory to analyze.

        Returns:
            Optional[int]: The epoch where convergence occurred, or None.
        """
        if len(trajectory.checkpoints) < 3:
            return None

        window_size = 3
        improvement_threshold = 0.01

        for i in range(len(trajectory.checkpoints) - window_size + 1):
            window = trajectory.checkpoints[i : i + window_size]
            improvement = window[-1].val_acc - window[0].val_acc

            # If improvement over the window is very small
            if improvement < improvement_threshold:
                return window[0].epoch

        return None  # Still improving

    def _check_stability(self, trajectory: TrainingTrajectory) -> bool:
        """
        Check if training is unstable (high loss variance).

        Args:
            trajectory: The training trajectory to analyze.

        Returns:
            bool: True if unstable.
        """
        if len(trajectory.checkpoints) < 5:
            return False

        recent_losses = [c.train_loss for c in trajectory.checkpoints[-5:]]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)

        # Avoid division by zero
        if loss_mean == 0:
            return False

        return (loss_std / loss_mean) > 0.5
