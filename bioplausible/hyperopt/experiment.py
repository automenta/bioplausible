"""
Experiment Runner

Executes hyperparameter optimization trials and collects metrics.
"""

import contextlib
import io
import logging
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.domains import create_task
from bioplausible.execution._guards import SafetyConfig
from bioplausible.execution._lifecycle import CheckpointManager, ExperimentArchiver
from bioplausible.execution._state import FailureRecord, FailureTracker
from bioplausible.execution.dashboard import DASHBOARD
from bioplausible.execution.monitoring import InterferenceMonitor
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.tracking import ExperimentTracker
from bioplausible.zoo import load_weights

__all__ = [
    "TrialRunner",
    "logger",
    "run_single_trial_task",
]


class TrialRunner:
    """Runs individual hyperparameter optimization trials."""

    def __init__(
        self,
        storage: HyperoptStorage = None,
        device: str = "auto",
        task: str = "shakespeare",
        quick_mode: bool = True,
        checkpoint_db_path: str = None,
        task_kwargs: dict = None,
        timeout: float = 3600.0,
        epochs: int = 3,
    ):
        self.storage = storage or HyperoptStorage()
        self.checkpoint_db_path = checkpoint_db_path
        self.device = self._select_device(device)
        self.task_name = task
        self.quick_mode = quick_mode
        self.epochs = epochs
        self.task_kwargs = task_kwargs or {}
        self.timeout = timeout

        # Initialize Task abstraction
        self._setup_task()

    def _select_device(self, device: str) -> str:
        """Resolve 'auto' device selection."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_task(self):
        """Initialize and setup the task object."""
        self.task_obj = create_task(
            self.task_name, self.device, self.quick_mode, **self.task_kwargs
        )
        self.task_obj.setup()
        self.input_dim = self.task_obj.input_dim
        self.output_dim = self.task_obj.output_dim

    def _load_transfer_weights(
        self, transfer_from: int, model: torch.nn.Module, config: dict
    ):
        """Helper to find and load weights from a previous trial."""
        import zipfile
        from pathlib import Path

        artifact_dir = Path("artifacts")
        if not artifact_dir.exists():
            logger.warning("Artifacts directory not found.")
            return

        def _load_state_dict(path: Path, model: torch.nn.Module, freeze_layers: bool):
            load_weights(
                model,
                str(path),
                device=self.device,
                strict=False,
                freeze_layers=freeze_layers,
            )

        try:
            # Find matching zip or dir
            for item in artifact_dir.iterdir():
                # Expected format: trial_{id}_{model_name}
                if item.name.startswith(f"trial_{transfer_from}_"):
                    freeze_layers = config.get("freeze_layers", False)
                    if item.is_dir():
                        found_path = item / "model.pt"
                        if found_path.exists():
                            _load_state_dict(found_path, model, freeze_layers)
                            return
                    elif item.suffix == ".zip":
                        # Unzip to temp context
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            with zipfile.ZipFile(item, "r") as zip_ref:
                                zip_ref.extract("model.pt", temp_path)
                                found_path = temp_path / "model.pt"
                                if found_path.exists():
                                    _load_state_dict(found_path, model, freeze_layers)
                                    return
                    break

            logger.warning("Could not find artifact for trial %s", transfer_from)

        except OSError, RuntimeError, ValueError, KeyError:
            logger.exception("Error loading transfer weights")

    def run_trial(self, trial_id: int, pruning_callback=None) -> bool:
        """Run a single trial and record results."""
        trial = self.storage.get_trial(trial_id)
        if not trial:
            logger.warning("Trial %s not found", trial_id)
            return False

        self.storage.update_trial(trial_id, status="running")

        tracker = ExperimentTracker(
            project="bioplausible",
            name=f"trial_{trial_id}_{trial.model_name}",
            config=trial.config,
        )

        try:
            # 1. Create Model and Trainer
            model, trainer = self._create_model_and_trainer(trial, tracker)

            # 2. Setup Training (Schedule, Monitoring, Checkpointing)
            from bioplausible.execution.training_dynamics import (
                ContinuousTrainingSchedule,
            )

            schedule = ContinuousTrainingSchedule(
                max_epochs=self.epochs, enable_pruning=True
            )
            # Disable monitor in quick mode to prevent test flakiness
            monitor = (
                InterferenceMonitor(threshold_cpu=20.0, sustain_duration=5.0)
                if not self.quick_mode
                else None
            )
            checkpoint_manager = None
            if self.checkpoint_db_path:
                try:
                    checkpoint_manager = CheckpointManager(
                        self.checkpoint_db_path, trial_id
                    )
                except (OSError, ValueError, RuntimeError, TypeError) as e:
                    logger.warning("Failed to init CheckpointManager: %s", e)

            # 3. Define Callbacks
            epoch_times = []
            start_time = time.time()

            def on_epoch_end_callback(epoch, metrics):
                # Timeout Check
                if time.time() - start_time > self.timeout:
                    logger.warning(
                        "Trial %s exceeded timeout (%ss). Stopping.",
                        trial_id,
                        self.timeout,
                    )
                    raise TimeoutError(f"Trial exceeded {self.timeout}s limit.")

                self.storage.log_epoch(
                    trial_id,
                    epoch - 1,
                    metrics["loss"],
                    metrics.get("accuracy", 0.0),
                    metrics.get("perplexity", 0.0),
                    metrics["time"],
                )
                epoch_times.append(metrics["time"])
                if checkpoint_manager:
                    checkpoint_manager.log_metric(epoch, 0, metrics)
                DASHBOARD.update_progress(epoch, self.epochs, metrics)

            def wrapped_pruning_callback(tid, epoch, m):
                if pruning_callback and pruning_callback(tid, epoch, m):
                    self.storage.update_trial(trial_id, status="pruned")
                    if monitor:
                        monitor.stop()
                    return True
                return False

            # 4. Execute Training Loop
            if monitor:
                monitor.start()

            trajectory = schedule.train_with_checkpoints(
                trainer=trainer,
                trial_id=trial_id,
                model_name=trial.model_name,
                task_name=self.task_name,
                config=trial.config,
                optuna_trial=None,
                pruning_callback=wrapped_pruning_callback,
                on_epoch_end=on_epoch_end_callback,
            )

            if monitor:
                monitor.stop()

            # 5. Finalize and Save
            if checkpoint_manager:
                checkpoint_manager.close()

            return self._finalize_trial(
                trial_id,
                trial,
                trajectory,
                monitor,
                epoch_times,
                model,
                trainer,
                config=trial.config,
            )

        except Exception as exc:  # noqa: BLE001  # broad: a failing trial must not stop the loop
            logger.warning("Trial %s failed: %s: %s", trial_id, type(exc).__name__, exc)
            self.storage.update_trial(trial_id, status="failed")
            return False
        finally:
            if "monitor" in locals() and monitor:
                monitor.stop()
            tracker.finish()

            # Robust Cleanup
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _create_model_and_trainer(self, trial, tracker):
        """Instantiate model and trainer based on trial config."""
        config = trial.config
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 4)

        model_cls = Registry.get(ComponentCategory.MODEL, trial.model_name)

        # Use the model's build classmethod if available (handles parameter mapping)
        if hasattr(model_cls, "build") and callable(getattr(model_cls, "build")):
            from bioplausible.zoo import get_model_spec

            spec = get_model_spec(trial.model_name)
            # Remove keys already passed explicitly to avoid duplicate kwargs.
            build_kwargs = dict(config)
            for _k in (
                "hidden_dim",
                "num_layers",
                "input_dim",
                "output_dim",
                "device",
                "task_type",
            ):
                build_kwargs.pop(_k, None)
            model = model_cls.build(
                spec=spec,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=self.device,
                task_type=self.task_name,
                **build_kwargs,
            )
        else:
            # Fallback to direct constructor call
            model = model_cls(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ).to(self.device)

        transfer_from = config.get("transfer_from")
        if transfer_from:
            logger.info(
                "Initializing Transfer Learning from Trial %s...", transfer_from
            )
            self._load_transfer_weights(transfer_from, model, config)

        meta = Registry.get_metadata(ComponentCategory.MODEL, trial.model_name)
        lr = config.get("lr", meta.typical_lr_range[0])
        beta = config.get("beta")
        steps = config.get("steps")

        trainer_kwargs = config.copy()
        # Clean config for kwargs
        for key in [
            "lr",
            "steps",
            "batches_per_epoch",
            "eval_batches",
            "model",
            "task",
            "tier",
            "job_id",
            "fold",
            "data_fraction",
            "is_verification",
            "verified_trial_id",
        ]:
            if key in trainer_kwargs:
                del trainer_kwargs[key]

        if "scheduler" in config:
            trainer_kwargs["scheduler_type"] = config["scheduler"]
            trainer_kwargs["scheduler_kwargs"] = config.get("scheduler_kwargs", {})

        # Resolve a string ``optimizer`` (from the search space: adam/sgd/...) to
        # an actual Optimizer instance. ``CoreTrainer.from_task`` assigns the
        # object verbatim to ``trainer.optimizer`` (never resolving a name), so a
        # bare string would later crash ``_bptt_step`` with
        # ``AttributeError: 'str' object has no attribute 'zero_grad'``.
        optimizer_name = config.get("optimizer")
        if isinstance(optimizer_name, str):
            opt_cls = getattr(torch.optim, optimizer_name, torch.optim.Adam)
            optimizer = opt_cls(
                model.parameters(),
                lr=lr,
                weight_decay=float(config.get("weight_decay", 0.0)),
                **config.get("optimizer_kwargs", {}),
            )
            trainer_kwargs["optimizer"] = optimizer

        safety_config = SafetyConfig(
            max_grad_norm=config.get("grad_clip", 10.0),
            nan_check_frequency=10,
            max_nan_retries=3,
        )

        trainer = self.task_obj.create_trainer(
            model,
            lr=lr,
            steps=steps if steps else 20,
            batches_per_epoch=200 if not self.quick_mode else 5,
            eval_batches=50 if not self.quick_mode else 2,
            tracker=tracker,
            safety_config=safety_config,
            **trainer_kwargs,
        )

        # ``model.config`` is a *frozen* ``ModelConfig`` (slots+dataclass), so
        # direct assignment raises ``FrozenInstanceError``. Use ``object.__setattr__``
        # to bypass the frozen guard, and additionally set the live attribute that
        # the training loop actually reads (``model.beta`` / ``model.max_steps``).
        if beta is not None:
            config_obj = getattr(model, "config", None)
            if config_obj is not None and hasattr(config_obj, "beta"):
                try:
                    object.__setattr__(config_obj, "beta", beta)
                except AttributeError, TypeError:
                    pass
            if hasattr(model, "beta"):
                if isinstance(model.beta, torch.Tensor):
                    model.beta.fill_(beta)
                else:
                    model.beta = beta

        if steps is not None and hasattr(model, "max_steps"):
            model.max_steps = int(steps)

        return model, trainer

    def _finalize_trial(
        self, trial_id, trial, trajectory, monitor, epoch_times, model, trainer, config
    ):
        """Process results, update storage, and archive artifacts."""
        self.storage.save_trajectory(trajectory)

        if trajectory.checkpoints and trajectory.checkpoints[-1].epoch < self.epochs:
            return False  # Pruned

        if monitor and monitor.check_interference():
            logger.warning("INTERFERENCE DETECTED: Rejecting trial results.")
            self.storage.update_trial(trial_id, status="failed")
            return False

        if not trajectory.checkpoints:
            logger.warning("No checkpoints found. Marking trial as failed.")
            self.storage.update_trial(trial_id, status="failed")
            return False

        last_ckpt = trajectory.checkpoints[-1]

        # Calculate avg iteration time
        divisor = (
            trainer.episodes_per_epoch
            if hasattr(trainer, "episodes_per_epoch")
            else (
                trainer.batches_per_epoch
                if hasattr(trainer, "batches_per_epoch")
                else 1
            )
        )
        avg_iter_time = np.mean(epoch_times) / divisor if epoch_times else 0.0

        # Store raw parameter count (not millions)
        param_count = sum(p.numel() for p in model.parameters())

        self.storage.update_trial(
            trial_id,
            status="completed",
            epochs_completed=self.epochs,
            final_loss=last_ckpt.train_loss,
            accuracy=last_ckpt.val_acc,
            perplexity=last_ckpt.perplexity if last_ckpt.perplexity else 0.0,
            iteration_time=avg_iter_time,
            param_count=param_count,
        )

        if config.get("save_artifacts"):
            logger.info("Archiving artifacts...")
            archiver = ExperimentArchiver()
            final_metrics = {
                "loss": last_ckpt.train_loss,
                "accuracy": last_ckpt.val_acc,
                "perplexity": last_ckpt.perplexity,
            }
            archiver.archive_trial(
                trial_id=trial_id, model=model, config=config, metrics=final_metrics
            )

        logger.info("Trial %s completed successfully!", trial_id)
        return True


def run_single_trial_task(
    task: str,
    model_name: str,
    config: dict[str, object],
    storage_path: str | None = None,
    quick_mode: bool = True,
    verbose: bool = False,
) -> dict[str, float] | None:
    """
    Execute a single trial for a given task and model configuration.
    Wraps TrialRunner with storage and failure tracking.
    """
    temp_dir = None

    if storage_path is None:
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "worker_temp.db"
    else:
        db_path = Path(storage_path)

    storage = None
    failure_tracker = FailureTracker(str(db_path))

    try:
        storage = HyperoptStorage(str(db_path))

        # Create trial entry
        trial_id = storage.create_trial(model_name, config)

        # Log basic config info
        tier = config.get("tier", "unknown")
        epochs = config.get("epochs", "?")
        logger.info(
            "[Trial %s] Task: %s | Model: %s | Tier: %s | Epochs: %s",
            trial_id,
            task,
            model_name,
            tier,
            epochs,
        )

        # Extract task kwargs
        task_kwargs = {}
        if "fold" in config:
            task_kwargs["fold"] = config["fold"]
        if "data_fraction" in config:
            task_kwargs["data_fraction"] = config["data_fraction"]

        # Create runner
        timeout = config.get("timeout", 3600.0)
        runner = TrialRunner(
            storage=storage,
            device="auto",
            task=task,
            quick_mode=quick_mode,
            checkpoint_db_path=str(db_path),
            task_kwargs=task_kwargs,
            timeout=timeout,
        )

        # Override epochs if present
        if "epochs" in config:
            runner.epochs = int(config["epochs"])

        # Run training
        if verbose:
            success = runner.run_trial(trial_id)
        else:
            # Suppress output but keep stderr for errors
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                success = runner.run_trial(trial_id)

        if success:
            trial = storage.get_trial(trial_id)
            metrics = {
                "trial_id": trial_id,  # DB PK
                "accuracy": trial.accuracy,
                "loss": trial.final_loss,
                "perplexity": trial.perplexity,
                "time": trial.iteration_time,
                "param_count": trial.param_count,  # In millions
            }
            return metrics
        else:
            if verbose:
                logger.warning("Trial %s returned success=False", trial_id)

            # Log logical failure (e.g. NaN, divergence)
            failure_tracker.log_failure(
                FailureRecord(
                    timestamp=datetime.now().isoformat(),
                    model_name=model_name,
                    task_name=task,
                    tier=config.get("tier", "unknown"),
                    trial_id=trial_id,
                    failure_type="training_failed",
                    failure_epoch=config.get("epochs", 0),  # approx
                    failure_batch=None,
                    config=config,
                    last_metrics={},
                )
            )
            return None

    except TimeoutError as e:
        logger.exception("Timeout Error")
        failure_tracker.log_failure(
            FailureRecord(
                timestamp=datetime.now().isoformat(),
                model_name=model_name,
                task_name=task,
                tier=config.get("tier", "unknown"),
                trial_id=config.get("job_id"),
                failure_type="timeout",
                failure_epoch=None,
                failure_batch=None,
                config=config,
                last_metrics={},
                stack_trace=str(e),
            )
        )
        return None

    except Exception:  # noqa: BLE001  # broad: top-level executor safety net
        logger.exception("Execution Error")
        if verbose:
            traceback.print_exc()

        # Log exception failure
        failure_tracker.log_failure(
            FailureRecord(
                timestamp=datetime.now().isoformat(),
                model_name=model_name,
                task_name=task,
                tier=config.get("tier", "unknown"),
                trial_id=config.get("job_id"),  # might be None
                failure_type="exception",
                failure_epoch=None,
                failure_batch=None,
                config=config,
                last_metrics={},
                stack_trace=traceback.format_exc(),
            )
        )
        return None
    finally:
        if storage:
            storage.close()

        # Cleanup
        if verbose:
            logger.info("Cleaning up trial resources...")

        # Explicitly break references
        if "runner" in locals():
            del runner
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            logger.info("Cleanup complete.")

        if temp_dir:
            shutil.rmtree(temp_dir)
