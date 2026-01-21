from enum import Enum
from typing import Generator, Optional, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass, field

from bioplausible.pipeline.config import TrainingConfig
from bioplausible.pipeline.events import Event, ProgressEvent, CompletedEvent, PausedEvent, Event
from bioplausible.hyperopt.tasks import create_task, BaseTask
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec
from bioplausible.training.base import BaseTrainer

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

            # 4. Training Loop
            for epoch in range(self.config.epochs):
                if self.state == SessionState.STOPPED:
                    break

                while self.state == SessionState.PAUSED:
                    yield PausedEvent()
                    # We need a way to sleep or wait, yielding allows consumer to handle wait
                    # But if we yield, we expect consumer to resume us.
                    # Since this is a generator, we can't easily wait inside without blocking.
                    # Usually generators are pulled.
                    # If paused, we can yield and expect next() to be called later?
                    # Or we check pause flag at start of epoch.
                    # Real pause inside epoch might be tricky with this generator structure.
                    # For now, pause check at epoch start.
                    import time
                    time.sleep(0.1)

                if self.state == SessionState.STOPPED:
                    break

                metrics = self.trainer.train_epoch()
                yield ProgressEvent(epoch=epoch, metrics=metrics)

            if self.state != SessionState.STOPPED:
                self.state = SessionState.COMPLETED
                yield CompletedEvent(final_metrics=metrics)

        except Exception as e:
            self.state = SessionState.ERROR
            import traceback
            traceback.print_exc()
            raise e

    def pause(self):
        if self.state == SessionState.RUNNING:
            self.state = SessionState.PAUSED

    def resume(self):
        if self.state == SessionState.PAUSED:
            self.state = SessionState.RUNNING

    def stop(self):
        self.state = SessionState.STOPPED
