"""
Lifecycle management for experiment execution.

Groups: promotion (``PromotionGate``), archiving (``ExperimentArchiver``),
checkpointing (``CheckpointManager``), and curriculum (``CurriculumManager``).
"""

import json
import shutil
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from bioplausible.core.checkpoint import save_checkpoint
from bioplausible.core.logging import get_logger
from bioplausible.core.training_state import TRAINING_CHECKPOINTS_DDL

__all__ = [
    "ARTIFACTS_DIR",
    "PROMOTION_THRESHOLDS",
    "CheckpointManager",
    "CheckpointRecord",
    "CurriculumManager",
    "ExperimentArchiver",
    "PromotionGate",
    "logger",
]

logger = get_logger("Lifecycle")

# ---------------------------------------------------------------------------
# Promotion — promotion.py
# ---------------------------------------------------------------------------

# Minimum success criteria for each task
PROMOTION_THRESHOLDS: dict[str, dict[str, float]] = {
    "char_ngram": {"accuracy": 0.95},  # Should be trivial
    "digits": {"accuracy": 0.90},  # Tiny, should be easy
    "usps": {"accuracy": 0.85},
    "kmnist": {"accuracy": 0.80},
    "mnist": {"accuracy": 0.85},  # Good baseline
    "fashion_mnist": {"accuracy": 0.75},
    "svhn": {"accuracy": 0.60},  # Noisier than F-MNIST
    "cifar10": {"accuracy": 0.45},  # Harder
    "cifar100": {"accuracy": 0.20},  # Very Hard (100 classes)
    "pendulum": {"reward": -200.0},  # "Solved" is roughly -200
    "cartpole": {"reward": 100.0},  # Basic balancing
    "acrobot": {"reward": -100.0},
}


class PromotionGate:
    """Checks if model performance warrants promotion."""

    @staticmethod
    def check_promotion(task_name: str, metrics: dict[str, object]) -> bool:
        """
        Check if metrics satisfy promotion criteria for task.

        Args:
            task_name: The name of the task.
            metrics: Dictionary of performance metrics (e.g., {'accuracy': 0.95}).

        Returns:
            bool: True if promotion criteria are met, False otherwise.
        """
        thresholds = PROMOTION_THRESHOLDS.get(task_name)
        if not thresholds:
            return True  # No barrier

        acc = metrics.get("accuracy")
        rew = metrics.get("reward")

        # Check Accuracy
        if "accuracy" in thresholds:
            if acc is None or acc < thresholds["accuracy"]:
                return False

        # Check Reward
        if "reward" in thresholds:
            if rew is None or rew < thresholds["reward"]:
                return False

        # Check Efficiency (if available)
        if (
            "time" in metrics
            and metrics["time"] > 0
            and (task_name in ["digits", "mnist"] and metrics["time"] > 600.0)
        ):  # > 10 mins for MNIST is bad
            return False

        return True

    @staticmethod
    def get_threshold_desc(task_name: str) -> str:
        """
        Get human readable description of promotion thresholds.

        Args:
            task_name: The task name.

        Returns:
            str: Description of thresholds (e.g., "accuracy > 0.95").
        """
        t = PROMOTION_THRESHOLDS.get(task_name, {})
        return ", ".join([f"{k} > {v}" for k, v in t.items()])


# ---------------------------------------------------------------------------
# Archiver — archiver.py
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")


class ExperimentArchiver:
    """
    Manages the creation and storage of experiment artifacts.
    """

    def __init__(self, base_dir: Path = ARTIFACTS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def archive_trial(
        self,
        trial_id: int,
        model: torch.nn.Module,
        config: dict[str, object],
        metrics: dict[str, object],
        extra_files: dict[str, str] | None = None,
    ) -> str | None:
        """
        Creates a ZIP archive for a specific trial.

        Args:
            trial_id: Unique trial ID.
            model: The PyTorch model to save.
            config: Configuration dictionary.
            metrics: Final metrics dictionary.
            extra_files: Optional dictionary mapping filename -> content (str).

        Returns:
            Path to the created ZIP file, or None if failed.
        """
        try:
            trial_name = f"trial_{trial_id}_{config.get('model', 'unknown')}"
            trial_dir = self.base_dir / trial_name
            trial_dir.mkdir(exist_ok=True)

            checkpoint_path = trial_dir / "model.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "metrics": dict(metrics),
                },
            )

            with Path(trial_dir / "config.json").open("w") as f:
                json.dump(config, f, indent=2)

            with Path(trial_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)

            repro_script = self._generate_reproduction_script(
                config.get("model", "unknown"), config, metrics
            )
            with Path(trial_dir / "reproduce.py").open("w") as f:
                f.write(repro_script)

            if extra_files:
                for fname, content in extra_files.items():
                    with Path(trial_dir / fname).open("w") as f:
                        f.write(content)

            zip_path = self.base_dir / f"{trial_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in trial_dir.iterdir():
                    zf.write(file, file.name)

            shutil.rmtree(trial_dir)

            logger.info("Archived trial %s to %s", trial_id, zip_path)
            return str(zip_path)

        except Exception:  # broad: best-effort
            logger.exception("Failed to archive trial %s", trial_id)
            return None

    def _generate_reproduction_script(
        self, model_name: str, config: dict[str, object], metrics: dict[str, object]
    ) -> str:
        """Generate a standalone reproduction script for this trial."""
        config_repr = repr(config)
        script = f'''"""
Standalone Reproduction Script
Generated by AutoScientist
Model: {model_name}
Original Accuracy: {metrics.get("accuracy", 0.0):.4f}
"""

import torch
import json
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.domains import create_task
from bioplausible.core.utils.device import get_device

def reproduce():
    config = {config_repr}

    logger.info("Reproducing model: %s", config['model'])
    device = str(get_device())

    task_name = config.get("task", "mnist")
    logger.info("Loading task: %s...", task_name)
    task = create_task(task_name, device=device, quick_mode=False)
    task.setup()

    logger.info("Creating model...")
    from bioplausible.core.construction import construct_model

    model_cls = Registry.get(ComponentCategory.MODEL, config["model"])
    model = construct_model(
        model_cls,
        {
            "hidden_dim": config.get("hidden_dim", 128),
            "num_layers": config.get("num_layers", 4),
        },
        input_dim=task.input_dim,
        output_dim=task.output_dim,
        model_name=config["model"],
    ).to(device)

    if "beta" in config and hasattr(model, "beta"):
        model.beta = config["beta"]

    logger.info("Starting training...")
    trainer = task.create_trainer(
        model,
        lr=config.get("lr", 0.001),
        steps=config.get("steps", 20),
        batches_per_epoch=200,
        eval_batches=50
    )

    epochs = config.get("epochs", 5)
    for epoch in range(epochs):
        metrics = trainer.train_epoch()
        acc = metrics.get('accuracy', 0.0)
        loss = metrics['loss']
        logger.info(
            "Epoch %d/%d: Acc=%.4f Loss=%.4f", epoch + 1, epochs, acc, loss
        )

if __name__ == "__main__":
    reproduce()
'''
        return script


# ---------------------------------------------------------------------------
# Checkpoint Manager — checkpoint_manager.py
# ---------------------------------------------------------------------------


@dataclass
class CheckpointRecord:
    epoch: int
    step: int
    metrics: dict[str, float]

    def to_dict(self):
        return asdict(self)


class CheckpointManager:
    """
    Manages saving checkpoints to the database.
    Lightweight synchronous SQLite-based checkpointing.
    """

    def __init__(self, db_path: str, trial_id: int):
        self.db_path = db_path
        self.trial_id = trial_id
        self.buffer = []
        self.buffer_size = 5  # Flush every 5 calls

    def log_metric(self, epoch: int, step: int, metrics: dict[str, float]):
        """Buffer a metric record."""
        record = CheckpointRecord(epoch, step, metrics)
        self.buffer.append(record)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffer to DB."""
        if not self.buffer:
            return

        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            data = []
            for r in self.buffer:
                train_acc = r.metrics.get(
                    "training_accuracy", r.metrics.get("train_acc", 0.0)
                )
                val_acc = r.metrics.get("accuracy", r.metrics.get("val_acc", 0.0))
                train_loss = r.metrics.get("loss", r.metrics.get("train_loss", 0.0))
                val_loss = r.metrics.get("val_loss", 0.0)
                perplexity = r.metrics.get("perplexity", 0.0)
                samples_seen = r.metrics.get("samples_seen", 0)
                timestamp = r.metrics.get("timestamp", 0.0)

                conn.execute(TRAINING_CHECKPOINTS_DDL)

                data.append((
                    self.trial_id,
                    r.epoch,
                    train_acc,
                    val_acc,
                    train_loss,
                    val_loss,
                    samples_seen,
                    perplexity,
                    timestamp,
                ))

            conn.executemany(
                """
                INSERT OR REPLACE INTO training_checkpoints
                (trial_id, trajectory_id, epoch, train_acc, val_acc,
                 train_loss, val_loss, samples_seen, perplexity, wall_time_seconds)
                VALUES (?, -1, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                data,
            )
            conn.commit()
            self.buffer = []
        except (sqlite3.Error, OSError, RuntimeError) as e:
            logger.warning("Failed to flush checkpoints: %s", e)
        finally:
            conn.close()

    def close(self):
        self.flush()


# ---------------------------------------------------------------------------
# Curriculum — curriculum.py
# ---------------------------------------------------------------------------


class CurriculumManager:
    """
    Defines task tracks and progressions.

    Attributes:
        TRACKS (Dict[str, List[str]]): Mapping of track names to ordered task lists.
    """

    TRACKS: dict[str, list[str]] = {
        "vision": [
            "digits",
            "usps",
            "kmnist",
            "mnist",
            "fashion_mnist",
            "svhn",
            "cifar10",
            "cifar100",
        ],
        "lm": ["char_ngram", "tiny_shakespeare"],
        "rl": ["cartpole", "pendulum", "acrobot"],
    }

    def __init__(self) -> None:
        """Initialize the Curriculum Manager."""

    def get_next_task(
        self, model_family: str, current_task: str, success: bool
    ) -> str | None:
        """
        Suggest next task based on current outcome.

        Args:
            model_family: The family of the model (unused currently but reserved).
            current_task: The name of the task just completed.
            success: Whether the task was completed successfully.

        Returns:
            The name of the next task, or None if no progression is available.
        """
        track = None
        for t_name, t_list in self.TRACKS.items():
            if current_task in t_list:
                track = t_list
                break

        if not track:
            return None

        try:
            curr_idx = track.index(current_task)
        except ValueError:
            return None

        if success:
            if curr_idx + 1 < len(track):
                return track[curr_idx + 1]
            else:
                return "completed_track"
        return None

    def get_initial_task(self, model_family: str) -> str:
        """
        Get starting task for a model family.

        Args:
            model_family: The type of model (e.g., 'transformer', 'mlp').

        Returns:
            The name of the initial task for this model type.
        """
        family = model_family.lower()
        if "transformer" in family or "lm" in family or "language" in family:
            return "char_ngram"
        elif "rl" in family or "control" in family:
            return "cartpole"
        else:
            return "digits"
