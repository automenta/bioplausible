"""
Training Worker Thread for EqProp Trainer

Runs training in background thread with rich, real-time progress updates.
"""

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional, Dict, Any
import traceback
import time
import numpy as np

# Use new unified training module
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.training.rl import RLTrainer
from bioplausible.hyperopt.tasks import BaseTask

class TrainingWorker(QThread):
    """Background worker for model training with real-time updates."""

    # Signals
    progress = pyqtSignal(dict)  # Emit training metrics frequently
    finished = pyqtSignal(dict)  # Emit final results
    error = pyqtSignal(str)      # Emit error message
    log = pyqtSignal(str)        # Emit log messages
    generation = pyqtSignal(str) # Emit generated text
    weights_updated = pyqtSignal(dict)  # Emit weight snapshots for visualization
    gradients_updated = pyqtSignal(dict)  # Emit gradient snapshots for visualization
    dynamics_update = pyqtSignal(dict)  # Emit dynamics data

    def __init__(
        self,
        model,
        train_loader,
        epochs: int = 10,
        lr: float = 0.001,
        use_compile: bool = True,
        use_kernel: bool = False,
        generate_interval: int = 5,  # Generate text every N epochs
        microscope_interval: int = 0, # Run microscope every N epochs (0=off)
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
        self.use_kernel = use_kernel
        self.generate_interval = generate_interval
        self.microscope_interval = microscope_interval
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

    def _initialize_trainer(self):
        """Initialize the SupervisedTrainer."""
        try:
            # We need a Task object for SupervisedTrainer.
            # Since the UI creates loaders manually, we can create a dummy/wrapper task
            # that helps with metric computation or input handling, OR we pass a lightweight task wrapper.

            # Identify task type from model or context
            # This is a bit hacky, but the worker is generic.
            # Ideally the worker should receive the Task object.
            # For now, we mock it or infer it.

            class WorkerTask(BaseTask):
                def __init__(self, model):
                    super().__init__("ui_worker_task")
                    self.model = model
                    # Guess task type
                    if getattr(model, 'has_embed', False):
                        self._task_type = "lm"
                    else:
                        self._task_type = "vision"

                @property
                def task_type(self):
                    return self._task_type

                def setup(self): pass
                def get_batch(self, split="train"): pass
                def create_trainer(self, model, **kwargs): pass

                def compute_metrics(self, logits, y, loss):
                    # Logic duplicated from tasks.py temporarily or reuse
                    if logits.dim() == 3 and self.task_type == "lm":
                        logits = logits[:, -1, :]

                    acc = (logits.argmax(1) == y).float().mean().item()
                    ppl = np.exp(min(loss, 10)) if self.task_type == "lm" else 0.0
                    return {"loss": loss, "accuracy": acc, "perplexity": ppl}

            task = WorkerTask(self.model)

            trainer = SupervisedTrainer(
                self.model,
                task=task,
                lr=self.lr,
                device="cuda" if torch.cuda.is_available() else "cpu"
                # use_compile and use_kernel are not standard args for SupervisedTrainer yet?
                # We should add them to SupervisedTrainer or handle them here.
                # For now, let's assume they are handled via **kwargs or ignored if not supported by standard trainer.
            )

            # Apply dynamic hyperparameters
            self._apply_hyperparams(trainer)
            return trainer

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

            x, y = x.to(trainer.device), y.to(trainer.device)

            batch_start = time.time()

            # Process the batch using SupervisedTrainer's train_batch
            # returns dict with loss, accuracy
            metrics = trainer.train_batch(x, y)

            loss_val = metrics['loss']
            acc_val = metrics.get('accuracy', 0.0)

            batch_time = time.time() - batch_start

            # Calculate counts from average accuracy for reporting
            # x.size(0) might vary on last batch
            batch_size = x.size(0)
            batch_correct_count = int(acc_val * batch_size)
            batch_total_count = batch_size

            # Accumulate metrics
            epoch_loss += loss_val * batch_size
            epoch_correct += batch_correct_count
            epoch_total += batch_total_count

            # Emit batch-level progress every 10 batches or on last batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                self._emit_batch_progress(epoch, batch_idx, num_batches, epoch_loss,
                                        epoch_correct, epoch_total, x, batch_time, total_start,
                                        batch_correct_count, batch_total_count)

        # End of epoch: compute Lipschitz
        # Standard models have compute_lipschitz
        try:
            lipschitz = trainer.model.compute_lipschitz() if hasattr(trainer.model, 'compute_lipschitz') else 0.0
        except:
            lipschitz = 0.0

        # Microscope analysis
        if self.microscope_interval > 0 and (epoch + 1) % self.microscope_interval == 0:
            self._run_microscope(trainer)

        # Emit final epoch metrics
        self._emit_epoch_metrics(epoch, num_batches, epoch_loss, epoch_correct,
                               epoch_total, lipschitz)

        return {'loss': epoch_loss / max(epoch_total, 1), 'accuracy': epoch_correct / max(epoch_total, 1)}

    def _run_microscope(self, trainer):
        """Run a single forward pass with dynamics tracking."""
        try:
            # Get a single batch
            x, _ = next(iter(self.train_loader))

            # Use trainer's prep logic if possible, or manual
            # SupervisedTrainer._prepare_input is internal.
            # But the model expects prepared input usually.

            x = x.to(trainer.device)
            # We mimic trainer prep:
            if getattr(trainer, 'has_embed', False):
                 # LM
                 pass # Embedding layer is inside model or attached?
                 # In SupervisedTrainer: self.embed(x).mean(dim=1)
                 # Here we might need to do that manually if the model expects vector input.
                 # Wait, trainer.train_batch calls _prepare_input.
                 # But for microscope we call model directly.
                 # If model.has_embed is True, model itself doesn't handle embedding usually (it's attached).
                 # So we need to embed.
                 if hasattr(trainer, 'embed') and trainer.embed:
                     x = trainer.embed(x).mean(dim=1)
            else:
                 # Vision
                 if x.dim() > 2 and "Conv" not in trainer.model.__class__.__name__:
                      x = x.view(x.size(0), -1)

            # Determine arguments for dynamics
            kwargs = {}
            import inspect
            sig = inspect.signature(self.model.forward)
            if 'return_dynamics' in sig.parameters:
                kwargs['return_dynamics'] = True

            # Run forward pass (no grad needed for visualization)
            with torch.no_grad():
                self.model.eval()
                out = self.model(x, **kwargs)
                self.model.train() # Switch back to train mode

                dynamics = {}
                if isinstance(out, tuple) and len(out) > 1:
                    dynamics = out[1]
                elif hasattr(self.model, 'dynamics'):
                    dynamics = self.model.dynamics

                if dynamics:
                    self.dynamics_update.emit(dynamics)
                    self.log.emit(f"Microscope: Captured dynamics for epoch.")

        except Exception as e:
            self.log.emit(f"Microscope failed: {e}")
            traceback.print_exc()

    def _emit_batch_progress(self, epoch, batch_idx, num_batches, epoch_loss,
                           epoch_correct, epoch_total, x, batch_time, total_start,
                           batch_correct_last, batch_total_last):
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
            'batch_accuracy': batch_correct_last / max(batch_total_last, 1),
            'lipschitz': 0.0,  # Placeholder, computed below
            'samples_per_sec': samples_per_sec,
            'eta_seconds': eta_seconds,
            'progress': (batches_done_total / (self.epochs * num_batches)) * 100,
        }
        self.progress.emit(metrics)

        # Emit weight/gradient snapshots for visualization
        if (batch_idx + 1) % 10 == 0:
            try:
                from .viz_utils import extract_weights, extract_gradients

                # Weights
                weights = extract_weights(self.model)
                if weights:
                    self.weights_updated.emit(weights)

                # Gradients (Synaptic Flow)
                grads = extract_gradients(self.model)
                if grads:
                    self.gradients_updated.emit(grads)

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


class BenchmarkWorker(QThread):
    """Worker for running validation tracks (benchmarks)."""

    progress = pyqtSignal(str) # Log message
    finished = pyqtSignal(dict) # Final results dict
    error = pyqtSignal(str)

    def __init__(self, track_ids, quick_mode=True, parent=None):
        super().__init__(parent)
        self.track_ids = track_ids
        self.quick_mode = quick_mode
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            from bioplausible.verify import Verifier
            import io
            from contextlib import redirect_stdout

            # Custom Verifier that respects stop signal and emits progress
            class SignalVerifier(Verifier):
                def __init__(self, worker, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.worker = worker

                def run_tracks(self, track_ids):
                    results = {}
                    for i, track_id in enumerate(track_ids):
                        if self.worker._stop_requested:
                            break

                        self.worker.progress.emit(f"Running Track {track_id}...")

                        try:
                            name, method = self.tracks[track_id]
                            result = method(self)
                            results[track_id] = result

                            status_icon = "✅" if result.status == "pass" else "❌"
                            self.worker.progress.emit(f"{status_icon} Track {track_id}: {result.status.upper()} ({result.score}/100)")

                        except Exception as e:
                            self.worker.progress.emit(f"❌ Track {track_id} Failed: {e}")

                    return results

            verifier = SignalVerifier(
                self,
                quick_mode=self.quick_mode,
                seed=42
            )

            self.progress.emit(f"Starting Benchmark Suite (Quick={self.quick_mode})...")
            results = verifier.run_tracks(self.track_ids)

            # Convert results to dict for signal
            final_results = {}
            for tid, res in results.items():
                final_results[tid] = {
                    'status': res.status,
                    'score': res.score,
                    'metrics': res.metrics
                }

            self.finished.emit(final_results)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()


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


class RLWorker(QThread):
    """Background worker for RL training."""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model, env_name, episodes=500, lr=1e-3, gamma=0.99, device='cpu', parent=None):
        super().__init__(parent)
        self.model = model
        self.env_name = env_name
        self.episodes = episodes
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self._stop_requested = False

    def stop(self):
        """Request training stop."""
        self._stop_requested = True

    def run(self):
        try:
            trainer = RLTrainer(self.model, self.env_name, device=self.device, lr=self.lr, gamma=self.gamma)

            start_time = time.time()

            for ep in range(self.episodes):
                if self._stop_requested:
                    break

                metrics = trainer.train_episode()

                # Calculate rolling average
                avg_reward = np.mean(trainer.reward_history[-50:]) if trainer.reward_history else 0.0

                self.progress.emit({
                    'episode': ep + 1,
                    'total_episodes': self.episodes,
                    'reward': metrics['reward'],
                    'loss': metrics['loss'],
                    'avg_reward': avg_reward,
                    'time': time.time() - start_time
                })

            self.finished.emit({'success': True})

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
