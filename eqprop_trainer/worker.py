"""
Training Worker Thread for EqProp Trainer

Runs training in background thread with rich, real-time progress updates.
"""

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional, Dict, Any
import traceback
import time


class TrainingWorker(QThread):
    """Background worker for model training with real-time updates."""

    # Signals
    progress = pyqtSignal(dict)  # Emit training metrics frequently
    finished = pyqtSignal(dict)  # Emit final results
    error = pyqtSignal(str)      # Emit error message
    generation = pyqtSignal(str) # Emit generated text
    weights_updated = pyqtSignal(dict)  # Emit weight snapshots for visualization

    def __init__(
        self,
        model,
        train_loader,
        epochs: int = 10,
        lr: float = 0.001,
        use_compile: bool = True,
        generate_interval: int = 5,  # Generate text every N epochs
        prompts: list = None,
        hyperparams: dict = None,  # Model-specific hyperparameters
        parent=None,
    ):
        super().__init__(parent)
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr
        self.use_compile = use_compile
        self.generate_interval = generate_interval
        self.prompts = prompts or ["ROMEO:"]
        self.hyperparams = hyperparams or {}

        self._stop_requested = False

    def stop(self):
        """Request training stop."""
        self._stop_requested = True

    def _apply_hyperparams(self, trainer):
        """Apply dynamic hyperparameters to trainer/model."""
        if not self.hyperparams:
            return

        for name, value in self.hyperparams.items():
            # Apply to trainer if applicable (e.g. beta, nudge_steps)
            if hasattr(trainer, name):
                setattr(trainer, name, value)
            # Apply to model config (Research Algorithms)
            elif hasattr(self.model, 'config') and hasattr(self.model.config, name):
                setattr(self.model.config, name, value)
            # Apply to model directly (Legacy/Simple models)
            elif hasattr(self.model, name):
                setattr(self.model, name, value)

    def _convert_input_format(self, x):
        """Convert input tensor to appropriate format based on model type."""
        if x.dim() == 4 and hasattr(self.model, 'input_dim'):
            # Vision: flatten images
            return x.view(x.size(0), -1)
        elif x.dtype == torch.long:
            # LM tokens: convert to float for bioplausible models
            if hasattr(self.model, 'config'):
                vocab_size = getattr(self.model.config, 'input_dim', 256)
                # Flatten sequence tokens and one-hot encode
                x_flat = x.reshape(-1)
                return torch.nn.functional.one_hot(x_flat, num_classes=vocab_size).float()

        return x

    def _compute_loss_and_accuracy(self, output, y):
        """Compute loss and accuracy based on output dimensions."""
        if output.dim() == 3:
            # Language modeling: output is [batch, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = output.shape
            output_flat = output.reshape(batch_size * seq_len, vocab_size)
            y_flat = y.reshape(batch_size * seq_len)
            loss = torch.nn.functional.cross_entropy(output_flat, y_flat)

            pred = output.argmax(dim=-1)
            batch_correct = (pred == y).sum().item()
            batch_total = batch_size * seq_len
        else:
            # Vision: output is [batch, num_classes]
            loss = torch.nn.functional.cross_entropy(output, y)

            pred = output.argmax(dim=1)
            batch_correct = (pred == y).sum().item()
            batch_total = y.size(0)  # Use y.size(0) instead of x.size(0) for consistency

        return loss, batch_correct, batch_total

    def _process_batch(self, x, y, trainer):
        """Process a single batch and return loss and accuracy metrics."""
        x = x.to(trainer.device)
        y = y.to(trainer.device)

        # Handle input conversions
        x = self._convert_input_format(x)

        trainer.optimizer.zero_grad()

        # Forward pass
        output = self.model(x)

        # Compute loss and accuracy
        loss, batch_correct, batch_total = self._compute_loss_and_accuracy(output, y)

        # Backward and optimize
        loss.backward()
        trainer.optimizer.step()

        return loss.item(), batch_correct, batch_total

    def _initialize_trainer(self):
        """Initialize the EqProp trainer with proper error handling."""
        try:
            from eqprop_torch import EqPropTrainer

            trainer = EqPropTrainer(
                self.model,
                lr=self.lr,
                use_compile=self.use_compile,
            )

            # Apply dynamic hyperparameters
            self._apply_hyperparams(trainer)
            return trainer
        except ImportError:
            self.error.emit("EqPropTrainer not available. Please install eqprop_torch.")
            return None
        except Exception as e:
            self.error.emit(f"Failed to initialize trainer: {e}")
            return None

    def _train_epoch(self, epoch, trainer, num_batches, total_start):
        """Train for a single epoch."""
        if self._stop_requested:
            return None

        # Training epoch
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (x, y) in enumerate(self.train_loader):
            if self._stop_requested:
                return None

            batch_start = time.time()

            # Process the batch
            loss_item, batch_correct_batch, batch_total_batch = self._process_batch(x, y, trainer)

            batch_time = time.time() - batch_start

            # Accumulate metrics
            epoch_loss += loss_item * x.size(0)
            epoch_correct += batch_correct_batch
            epoch_total += batch_total_batch

            # Emit batch-level progress every 10 batches or on last batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                self._emit_batch_progress(epoch, batch_idx, num_batches, epoch_loss,
                                        epoch_correct, epoch_total, x, batch_time, total_start)

        # End of epoch: compute Lipschitz
        try:
            lipschitz = trainer.compute_lipschitz() or 0.0
        except:
            lipschitz = 0.0  # Default if computation fails

        # Emit final epoch metrics
        self._emit_epoch_metrics(epoch, num_batches, epoch_loss, epoch_correct,
                               epoch_total, lipschitz)

        return {'loss': epoch_loss / max(epoch_total, 1), 'accuracy': epoch_correct / max(epoch_total, 1)}

    def _emit_batch_progress(self, epoch, batch_idx, num_batches, epoch_loss,
                           epoch_correct, epoch_total, x, batch_time, total_start):
        """Emit progress update for batch processing."""
        current_loss = epoch_loss / max(epoch_total, 1)
        current_acc = epoch_correct / max(epoch_total, 1)

        # Time estimates
        elapsed_total = time.time() - total_start
        batches_done_total = epoch * num_batches + batch_idx + 1
        batches_remaining = self.epochs * num_batches - batches_done_total
        avg_batch_time = elapsed_total / batches_done_total
        eta_seconds = batches_remaining * avg_batch_time

        # Throughput
        samples_per_sec = x.size(0) / max(batch_time, 0.001)

        # Emit rich progress
        metrics = {
            'epoch': epoch + 1,
            'total_epochs': self.epochs,
            'batch': batch_idx + 1,
            'total_batches': num_batches,
            'loss': current_loss,
            'batch_loss': current_loss,  # Use current loss as batch loss
            'accuracy': current_acc,
            'batch_accuracy': batch_correct_batch / max(batch_total_batch, 1),
            'lipschitz': 0.0,  # Placeholder, computed below
            'samples_per_sec': samples_per_sec,
            'eta_seconds': eta_seconds,
            'progress': (batches_done_total / (self.epochs * num_batches)) * 100,
        }
        self.progress.emit(metrics)

        # Emit weight snapshots for visualization
        if (batch_idx + 1) % 10 == 0:
            try:
                from .viz_utils import extract_weights
                weights = extract_weights(self.model)
                if weights:
                    self.weights_updated.emit(weights)
            except Exception:
                pass  # Ignore visualization errors

    def _emit_epoch_metrics(self, epoch, num_batches, epoch_loss, epoch_correct,
                          epoch_total, lipschitz):
        """Emit final metrics for the epoch."""
        final_metrics = {
            'epoch': epoch + 1,
            'total_epochs': self.epochs,
            'batch': num_batches,
            'total_batches': num_batches,
            'loss': epoch_loss / max(epoch_total, 1),
            'accuracy': epoch_correct / max(epoch_total, 1),
            'lipschitz': lipschitz,
            'samples_per_sec': 0.0,
            'eta_seconds': 0.0,
            'progress': ((epoch + 1) / self.epochs) * 100,
        }
        self.progress.emit(final_metrics)

    def _generate_text(self, epoch):
        """Generate text periodically during training."""
        if (epoch + 1) % self.generate_interval == 0 and hasattr(self.model, 'generate'):
            try:
                for prompt in self.prompts:
                    text = self.model.generate(prompt, max_new_tokens=100)
                    self.generation.emit(text)
            except Exception:
                pass  # Ignore generation errors

    def run(self):
        """Run training loop with rich real-time feedback."""
        try:
            trainer = self._initialize_trainer()
            if trainer is None:
                return

            num_batches = len(self.train_loader)
            total_start = time.time()

            for epoch in range(self.epochs):
                if self._stop_requested:
                    break

                epoch_metrics = self._train_epoch(epoch, trainer, num_batches, total_start)

                if epoch_metrics is None:  # Training was stopped
                    break

                # Generate text periodically
                self._generate_text(epoch)

            self.finished.emit({'success': True, 'epochs_completed': epoch + 1})

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class GenerationWorker(QThread):
    """Background worker for text generation."""

    result = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, prompt: str, max_tokens: int = 100, temperature: float = 1.0, parent=None):
        super().__init__(parent)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self):
        try:
            if hasattr(self.model, 'generate'):
                text = self.model.generate(
                    self.prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self.result.emit(text)
            else:
                self.error.emit("Model does not support generation")
        except Exception as e:
            self.error.emit(str(e))