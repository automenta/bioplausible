from enum import Enum
from typing import Generator, Optional, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import dataclasses

from bioplausible.pipeline.config import TrainingConfig
from bioplausible.pipeline.events import Event, ProgressEvent, CompletedEvent, PausedEvent, Event
from bioplausible.hyperopt.tasks import create_task, BaseTask
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec
from bioplausible.training.base import BaseTrainer
from bioplausible.pipeline.results import ResultsManager

class SessionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"

class TrainingSession:
    """Headless training orchestrator (no UI dependencies)."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = SessionState.IDLE
        self.task: Optional[BaseTask] = None
        self.model: Optional[nn.Module] = None
        self.trainer: Optional[BaseTrainer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Unique Run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}_{str(uuid.uuid4())[:8]}"

    def start(self) -> Generator[Event, None, None]:
        """Start training, yield events."""
        self.state = SessionState.RUNNING

        try:
            # 1. Setup Task
            self.task = create_task(self.config.dataset, device=self.device)
            # Some task names might be generic like "vision", usually dataset implies task.
            # config.dataset holds "mnist", "cifar10", "shakespeare", etc.

            self.task.setup()

            # 2. Setup Model
            spec = get_model_spec(self.config.model)

            # Prepare hyperparams
            # We merge config.hyperparams with model spec defaults if needed
            # For now, we assume factory handles spec defaults

            self.model = create_model(
                spec=spec,
                input_dim=self.task.input_dim,
                output_dim=self.task.output_dim,
                device=self.device,
                task_type=self.task.task_type,
                **self.config.hyperparams
            )

            # 3. Create Trainer
            # Different tasks might need different trainer setup
            # BaseTask has create_trainer
            self.trainer = self.task.create_trainer(
                self.model,
                batches_per_epoch=100, # Default, maybe should be in config
                eval_batches=20,
                epochs=self.config.epochs,
                lr=self.config.learning_rate,
                **self.config.hyperparams
            )

            metrics = {}

            # 4. Training Loop
            for epoch in range(self.config.epochs):
                if self.state == SessionState.STOPPED:
                    break

                while self.state == SessionState.PAUSED:
                    yield PausedEvent()
                    import time
                    time.sleep(0.1)

                if self.state == SessionState.STOPPED:
                    break

                metrics = self.trainer.train_epoch()
                yield ProgressEvent(epoch=epoch, metrics=metrics)

            if self.state != SessionState.STOPPED:
                self.state = SessionState.COMPLETED

                # Save results
                self._save_results(metrics)

                yield CompletedEvent(final_metrics=metrics)

        except Exception as e:
            self.state = SessionState.ERROR
            import traceback
            traceback.print_exc()
            raise e

    def _save_results(self, metrics):
        """Save results using ResultsManager."""
        try:
            mgr = ResultsManager()
            # Convert config dataclass to dict
            config_dict = dataclasses.asdict(self.config)
            mgr.save_run(self.run_id, config_dict, metrics)
        except Exception as e:
            print(f"Failed to save results: {e}")

    def pause(self):
        if self.state == SessionState.RUNNING:
            self.state = SessionState.PAUSED

    def resume(self):
        if self.state == SessionState.PAUSED:
            self.state = SessionState.RUNNING

    def stop(self):
        self.state = SessionState.STOPPED
