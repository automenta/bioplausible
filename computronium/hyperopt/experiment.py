"""
Experiment Runner

Executes hyperparameter optimization trials and collects metrics.
"""

import contextlib
import io
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from computronium.core.construction import construct_model
from computronium.core.logging import get_logger
from computronium.core.utils.device import get_device
from computronium.execution._guards import SafetyConfig
from computronium.execution._lifecycle import CheckpointManager, ExperimentArchiver
from computronium.execution.events import EventSink, NullEventSink
from computronium.execution.monitoring import InterferenceMonitor
from computronium.hyperopt.storage import HyperoptStorage
from computronium.tracking import ExperimentTracker
from computronium.utils import count_parameters

logger = get_logger()


__all__ = [
    "TrialRunner",
    "logger",
    "run_single_trial_task",
]


class TrialRunner:
    """Runs individual hyperparameter optimization trials."""

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        storage: HyperoptStorage = None,
        device: str = "auto",
        task: str = "shakespeare",
        quick_mode: bool = True,
        checkpoint_db_path: str | None = None,
        task_kwargs: dict | None = None,
        timeout: float = 3600.0,
        epochs: int = 3,
        event_sink: EventSink | None = None,
        *,
        model_cls: object | None = None,
    ):
        self.storage = storage or HyperoptStorage()
        self.checkpoint_db_path = checkpoint_db_path
        self.device = self._select_device(device)
        self.task_name = task
        self.quick_mode = quick_mode
        self.epochs = epochs
        self.task_kwargs = task_kwargs or {}
        self.timeout = timeout
        self.model_cls = model_cls
        self._events: EventSink = (
            event_sink if event_sink is not None else NullEventSink()
        )

        # Initialize Task abstraction
        self._setup_task()

    def _select_device(self, device: str) -> str:
        """Resolve 'auto' device selection."""
        if device == "auto":
            return str(get_device())
        return device

    def _setup_task(self):
        """Initialize and setup the task object via unified resolution."""
        from computronium.config.unified import DataConfig
        from computronium.domains.registry import resolve_task_from_data_config

        data_config = DataConfig(
            name=self.task_name,
            task=self.task_name,
            batch_size=64,
            val_batch_size=None,
            num_workers=4,
            seq_len=self.task_kwargs.get("seq_len", 64),
            augment=False,
            data_fraction=self.task_kwargs.get("data_fraction", 1.0),
            data_kwargs=self.task_kwargs,
        )

        self.task_obj = resolve_task_from_data_config(data_config, device=self.device)
        self.input_dim = self.task_obj.input_dim
        self.output_dim = self.task_obj.output_dim

    def run_trial(self, trial_id: int, pruning_callback=None) -> bool:  # ruff: ignore[complex-structure, too-many-statements]
        """Run a single trial and record results."""
        trial = self.storage.get_trial(trial_id)
        if not trial:
            logger.warning("Trial %s not found", trial_id)
            return False

        self.storage.update_trial(trial_id, status="running")

        tracker = ExperimentTracker(
            project="computronium",
            name=f"trial_{trial_id}_{trial.model_name}",
            config=trial.config,
        )

        try:  # ruff: ignore[too-many-statements-in-try-clause]
            # 1. Create Model and Trainer
            model, trainer = self._create_model_and_trainer(trial, tracker)

            # 2. Setup Training (Schedule, Monitoring, Checkpointing)
            from computronium.execution.training_dynamics import (
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
                    raise TimeoutError(f"Trial exceeded {self.timeout}s limit.")  # ruff: ignore[raise-within-try]

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
                self._events.update_progress(epoch, self.epochs, metrics)

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

        except Exception as exc:  # broad: a failing trial must not stop the loop
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

    def _create_model_and_trainer(self, trial, tracker):  # ruff: ignore[complex-structure, too-many-locals]
        """Instantiate model and trainer based on trial config."""
        config = trial.config
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 4)

        model_cls = self.model_cls
        if model_cls is None:
            raise ValueError(f"No model class provided for {trial.model_name!r}")

        # Single construction layer: routes through construct_model, which
        # handles config-accepting models, the TileAlgorithm .build substrate
        # routing, and reflection-based kwarg filtering uniformly.
        build_config = dict(config)
        build_config.setdefault("hidden_dim", hidden_dim)
        build_config.setdefault("num_layers", num_layers)
        build_config.setdefault("task_type", self.task_name)
        model = construct_model(
            model_cls,
            build_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_name=trial.model_name,
        )
        model = model.to(self.device)

        lr = config.get("lr", 1e-3)
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
        optimizer_name = config.get("optimizer", "adam")
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

        # Attach optimizer to model for models that expect it (e.g. EqProp
        # contrastive_step, energy-based models with custom train_step).
        # The trainer owns the optimizer instance; we mirror it on the model
        # so that `model.optimizer` is available where `train_step` expects it.
        model.optimizer = trainer.optimizer

        # ``model.config`` is a *frozen* ``ModelConfig`` (slots+dataclass), so
        # direct assignment raises ``FrozenInstanceError``. Use ``object.__setattr__``
        # to bypass the frozen guard, and additionally set the live attribute that
        # the training loop actually reads (``model.beta`` / ``model.max_steps``).
        if beta is not None:
            config_obj = getattr(model, "config", None)
            if config_obj is not None and hasattr(config_obj, "beta"):
                try:  # ruff: ignore[suppressible-exception]
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
        param_count = count_parameters(model, trainable_only=False)

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


def run_single_trial_task(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
    task: str,
    model_name: str,
    config: dict[str, object],
    storage_path: str | None = None,
    quick_mode: bool = True,
    verbose: bool = False,
    event_sink: EventSink | None = None,
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

    try:  # ruff: ignore[too-many-statements-in-try-clause]
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
            event_sink=event_sink,
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
            _sink_completed(model_name, task, config, metrics)
            return metrics
        else:
            if verbose:
                logger.warning("Trial %s returned success=False", trial_id)

            # Log logical failure (e.g. NaN, divergence)
            _sink_failure(model_name, task, config, "failed", trial_id=trial_id)
            return None

    except TimeoutError as e:
        logger.exception("Timeout Error")
        _sink_failure(model_name, task, config, "error", error=str(e))
        return None

    except Exception:  # broad: top-level executor safety net
        logger.exception("Execution Error")
        if verbose:
            traceback.print_exc()

        # Log exception failure
        _sink_failure(model_name, task, config, "error", error=traceback.format_exc())
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


def _sink_completed(
    model_name: str,
    task: str,
    config: dict[str, object],
    metrics: dict[str, object],
) -> None:
    """Persist a successful ExecutionEngine trial to the KnowledgeBase (best-effort).

    Separate from the probe-driver path so each success compounds into the
    knowledge layer regardless of which experiment framework produced it.
    """
    if os.environ.get("COMPUTRONIUM_RECORD_RESULTS", "1") == "0":
        return
    try:
        from computronium.experiment.result_sink import record_experiment_result

        record_experiment_result(
            model=model_name,
            task=task,
            config=config,
            metrics=metrics,
            status="completed",
            seed=config.get("seed"),
            epochs=config.get("epochs"),
            device="auto",
            extra={"source": "execution_engine"},
        )
    except Exception:  # pragma: no cover  # best-effort persistence
        logger.exception("result_sink failed for %s/%s", model_name, task)


def _sink_failure(
    model_name: str,
    task: str,
    config: dict[str, object],
    status: str,
    *,
    error: str = "",
    trial_id: object | None = None,
) -> None:
    """Persist a failed ExecutionEngine trial through the single result sink."""
    if os.environ.get("COMPUTRONIUM_RECORD_RESULTS", "1") == "0":
        return
    try:
        from computronium.experiment.result_sink import record_experiment_result

        extra: dict[str, object] = {
            "source": "execution_engine",
            "tier": config.get("tier", "unknown"),
        }
        if error:
            extra["error"] = error
        record_experiment_result(
            model=model_name,
            task=task,
            config=config,
            metrics={},
            status=status,
            seed=config.get("job_id", trial_id),
            epochs=config.get("epochs"),
            device="auto",
            extra=extra,
        )
    except Exception:  # pragma: no cover  # best-effort persistence
        logger.exception("result_sink failed for %s/%s", model_name, task)
