"""
Experiment Runner

Executes hyperparameter optimization trials and collects metrics.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional
import sys

from bioplausible.config import GLOBAL_CONFIG
from bioplausible.models.registry import get_model_spec, ModelSpec
from bioplausible.models.base import ModelConfig
from bioplausible.models.factory import create_model
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.hyperopt.tasks import create_task, BaseTask

class ExperimentAlgorithm:
    """
    Wrapper for models to unify interface for experiment runner.
    """

    def __init__(
        self,
        spec: ModelSpec,
        output_dim: int,
        input_dim: int = None,
        hidden_dim: int = 128,
        num_layers: int = 4,
        device: str = "cpu",
        task_type: str = "lm",  # "lm", "vision", "rl"
    ):
        self.spec = spec
        self.name = spec.name
        self.device = device
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task_type = task_type

        # Hyperparameters from spec
        self.lr = spec.default_lr
        self.beta = spec.default_beta
        self.steps = spec.default_steps

        self.has_embed = False  # Default

        # Create model
        self.model = self._create_model()

        # Optimizer (only for supervised models that don't have internal optimizers)
        if not hasattr(self.model, 'optimizer'):
            params = list(self.model.parameters())
            if self.has_embed:
                params.extend(list(self.embed.parameters()))
            self.opt = torch.optim.Adam(params, lr=self.lr)
        else:
            self.opt = None

        self.criterion = nn.CrossEntropyLoss()

        # Calculate param count
        self.param_count = sum(p.numel() for p in self.model.parameters())

    def _create_model(self):
        """Factory method for model creation using bioplausible models."""
        model = create_model(
            spec=self.spec,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            device=self.device,
            task_type=self.task_type
        )

        # Attach embedding if created by factory
        if getattr(model, 'has_embed', False):
            self.has_embed = True
            self.embed = model.embed

        return model

    def update_hyperparams(
        self, lr: float = None, beta: float = None, steps: int = None, **kwargs
    ):
        if lr is not None:
            self.lr = lr
            if self.opt:
                for g in self.opt.param_groups:
                    g["lr"] = lr
        if beta is not None:
            self.beta = beta
        if steps is not None:
            self.steps = steps

        # Handle Hebbian-specific updates
        if isinstance(self.model, DeepHebbianChain):
            if 'hebbian_lr' in kwargs:
                self.model.hebbian_lr = kwargs['hebbian_lr']
            if 'use_oja' in kwargs:
                self.model.use_oja = kwargs['use_oja']
                for layer in self.model.chain:
                    if hasattr(layer, 'original_layer'): # If spectral normed
                        layer.original_layer.use_oja = kwargs['use_oja']
                        layer.original_layer.learning_rate = self.model.hebbian_lr
                    else:
                        layer.use_oja = kwargs['use_oja']
                        layer.learning_rate = self.model.hebbian_lr

    def train_step(self, x, y, step_num) -> Any:
        """Single training iteration (Supervised)."""
        t0 = time.time()

        self.model.train()
        if self.opt:
            self.opt.zero_grad()

        try:
            # Handle Embedding if separate
            if self.has_embed:
                # Average pooling over sequence for MLP-like models
                h = self.embed(x).mean(dim=1)
            else:
                # Vision or direct input
                if x.dim() > 2 and self.task_type in ["vision", "rl"]:
                    if self.spec.model_type == "modern_conv_eqprop":
                         # Do not flatten
                         h = x
                    else:
                         h = x.view(x.size(0), -1)
                else:
                     h = x

            # Check for custom train_step (BioModel)
            if hasattr(self.model, "train_step"):
                metrics = self.model.train_step(h, y)
                loss = metrics.get("loss", 0.0)
                acc = metrics.get("accuracy", 0.0)
            else:
                # Standard forward/backward
                if hasattr(self.model, "eq_steps"):
                    logits = self.model(h, steps=self.steps)
                else:
                    logits = self.model(h)

                if logits.dim() == 3 and self.task_type == "lm":
                    # logits: [B, T, V] -> [B, V] (last token)
                    logits = logits[:, -1, :]

                loss = self.criterion(logits, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.opt:
                    self.opt.step()

                if self.task_type == "lm" or self.task_type == "vision":
                     acc = (logits.argmax(1) == y).float().mean().item()
                else:
                     acc = 0.0

                loss = loss.item()

            # VRAM estimate
            if torch.cuda.is_available() and "cuda" in self.device:
                vram = torch.cuda.memory_allocated() / 1e9
            else:
                vram = (self.param_count * 4) / 1e9

            # Helper class for result
            class TrainingState:
                def __init__(self, loss, accuracy, perplexity, iter_time, vram_gb, step):
                    self.loss = loss
                    self.accuracy = accuracy
                    self.perplexity = perplexity
                    self.iter_time = iter_time
                    self.vram_gb = vram_gb
                    self.step = step

            iter_time = time.time() - t0
            return TrainingState(
                loss=loss,
                accuracy=acc,
                perplexity=np.exp(min(loss, 10)) if self.task_type == "lm" else 0.0,
                iter_time=iter_time,
                vram_gb=vram,
                step=step_num,
            )

        except Exception as e:
            print(f"Error in {self.name} train_step: {e}")
            import traceback
            traceback.print_exc()

            class TrainingState:
                def __init__(self, loss, accuracy, perplexity, iter_time, vram_gb, step):
                    self.loss = loss
                    self.accuracy = accuracy
                    self.perplexity = perplexity
                    self.iter_time = iter_time
                    self.vram_gb = vram_gb
                    self.step = step

            return TrainingState(
                loss=10.0,
                accuracy=0.0,
                perplexity=100.0,
                iter_time=0.01,
                vram_gb=0.0,
                step=step_num,
            )


class TrialRunner:
    """Runs individual hyperparameter optimization trials."""

    def __init__(
        self,
        storage: HyperoptStorage = None,
        device: str = "auto",
        task: str = "shakespeare",
        quick_mode: bool = True,
    ):
        self.storage = storage or HyperoptStorage()
        self.device = (
            "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        )
        self.task_name = task
        self.quick_mode = quick_mode
        self.epochs = GLOBAL_CONFIG.epochs

        if GLOBAL_CONFIG.quick_mode:
            self.batches_per_epoch = 100
            self.eval_batches = 20
        else:
            self.batches_per_epoch = 200
            self.eval_batches = 50

        # Initialize Task abstraction
        self.task_obj = create_task(task, self.device, quick_mode)
        self.task_obj.setup()

        self.input_dim = self.task_obj.input_dim
        self.output_dim = self.task_obj.output_dim

    def run_trial(self, trial_id: int, pruning_callback=None) -> bool:
        """Run a single trial and record results."""
        # Get trial
        trial = self.storage.get_trial(trial_id)
        if not trial:
            print(f"Trial {trial_id} not found")
            return False

        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: {trial.model_name}")
        print(f"Config: {trial.config}")
        print(f"{'='*60}\n")

        self.storage.update_trial(trial_id, status="running")

        try:
            # Create model using wrapper
            spec = get_model_spec(trial.model_name)
            config = trial.config
            hidden_dim = config.get("hidden_dim", 128)
            num_layers = config.get("num_layers", 4)

            algo = ExperimentAlgorithm(
                spec,
                output_dim=self.output_dim,
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=self.device,
                task_type=self.task_obj.task_type
            )

            # Apply hyperparameters
            lr = config.get("lr", spec.default_lr)
            beta = config.get("beta", spec.default_beta) if spec.has_beta else None
            steps = config.get("steps", spec.default_steps) if spec.has_steps else None

            # Additional params
            extra_params = {}
            if "hebbian_lr" in config:
                extra_params["hebbian_lr"] = config["hebbian_lr"]
            if "use_oja" in config:
                extra_params["use_oja"] = config["use_oja"]

            algo.update_hyperparams(lr=lr, beta=beta, steps=steps, **extra_params)

            # --- RL Branch ---
            if self.task_obj.task_type == "rl":
                print("Running RL Training Loop...")
                gamma = config.get("gamma", 0.99)
                rl_trainer = self.task_obj.create_trainer(algo.model, lr=lr, gamma=gamma)

                episodes_per_epoch = 10
                total_epochs = self.epochs
                epoch_times = []

                for epoch in range(total_epochs):
                    epoch_start = time.time()
                    epoch_reward_sum = 0
                    epoch_loss_sum = 0

                    for _ in range(episodes_per_epoch):
                        metrics = rl_trainer.train_episode()
                        epoch_reward_sum += metrics['reward']
                        epoch_loss_sum += metrics['loss']

                    epoch_time = time.time() - epoch_start
                    epoch_times.append(epoch_time)

                    avg_reward = epoch_reward_sum / episodes_per_epoch
                    avg_loss = epoch_loss_sum / episodes_per_epoch

                    self.storage.log_epoch(
                        trial_id, epoch, avg_loss, avg_reward, 0.0, epoch_time
                    )

                    print(
                        f"Epoch {epoch+1}/{total_epochs}: "
                        f"loss={avg_loss:.4f}, avg_reward={avg_reward:.2f}, "
                        f"time={epoch_time:.1f}s"
                    )

                    if pruning_callback:
                         metrics = {
                            "loss": avg_loss,
                            "accuracy": avg_reward,
                            "perplexity": 0.0,
                            "time": epoch_time,
                            "iteration_time": epoch_time / episodes_per_epoch,
                        }
                         if pruning_callback(trial_id, epoch + 1, metrics):
                            print(f"✂️ Trial {trial_id} PRUNED")
                            self.storage.update_trial(trial_id, status="pruned")
                            return False

                final_loss = avg_loss
                final_reward = avg_reward
                avg_iter_time = np.mean(epoch_times) / episodes_per_epoch
                param_count_millions = algo.param_count / 1e6

                self.storage.update_trial(
                    trial_id,
                    status="completed",
                    epochs_completed=total_epochs,
                    final_loss=final_loss,
                    accuracy=final_reward,
                    perplexity=0.0,
                    iteration_time=avg_iter_time,
                    param_count=param_count_millions,
                )
                print(f"\n✅ Trial {trial_id} completed!")
                return True

            # --- Supervised Branch (LM/Vision) ---
            else:
                epoch_times = []
                n_epochs = self.epochs

                for epoch in range(n_epochs):
                    epoch_start = time.time()

                    # Training
                    algo.model.train()
                    train_losses = []

                    for _ in range(self.batches_per_epoch):
                        x, y = self.task_obj.get_batch("train")
                        state = algo.train_step(x, y, epoch * self.batches_per_epoch + _)
                        train_losses.append(state.loss)

                    # Validation
                    algo.model.eval()
                    val_losses = []
                    val_accs = []

                    with torch.no_grad():
                        for _ in range(self.eval_batches):
                            x, y = self.task_obj.get_batch("val") # Task should handle "val"

                            if algo.has_embed:
                                h = algo.embed(x).mean(dim=1)
                                logits = algo.model(h)
                            else:
                                if x.dim() > 2 and algo.spec.model_type == "modern_conv_eqprop":
                                    x_in = x
                                elif x.dim() > 2:
                                    x_in = x.view(x.size(0), -1)
                                else:
                                    x_in = x

                                logits = (
                                    algo.model(x_in, steps=algo.steps)
                                    if hasattr(algo.model, "eq_steps")
                                    else algo.model(x_in)
                                )
                                if logits.dim() == 3 and self.task_obj.task_type == "lm":
                                    logits = logits[:, -1, :]

                            # Calculate metrics using task (simplified here for validation loop)
                            loss = algo.criterion(logits, y)
                            metrics = self.task_obj.compute_metrics(logits, y, loss.item())

                            val_losses.append(metrics["loss"])
                            val_accs.append(metrics.get("accuracy", 0.0))

                    epoch_time = time.time() - epoch_start
                    epoch_times.append(epoch_time)

                    avg_val_loss = np.mean(val_losses)
                    avg_val_acc = np.mean(val_accs)
                    avg_val_ppl = np.exp(min(avg_val_loss, 10)) if self.task_obj.task_type == "lm" else 0.0

                    self.storage.log_epoch(
                        trial_id, epoch, avg_val_loss, avg_val_acc, avg_val_ppl, epoch_time
                    )

                    print(
                        f"Epoch {epoch+1}/{n_epochs}: "
                        f"loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}, "
                        f"ppl={avg_val_ppl:.2f}, time={epoch_time:.1f}s"
                    )

                    if pruning_callback:
                        metrics = {
                            "loss": avg_val_loss,
                            "accuracy": avg_val_acc,
                            "perplexity": avg_val_ppl,
                            "time": epoch_time,
                            "iteration_time": epoch_time / self.batches_per_epoch,
                        }
                        if pruning_callback(trial_id, epoch + 1, metrics):
                            print(f"✂️ Trial {trial_id} PRUNED at epoch {epoch+1}")
                            self.storage.update_trial(trial_id, status="pruned")
                            return False

                final_loss = np.mean(val_losses)
                final_acc = np.mean(val_accs)
                final_ppl = np.exp(min(final_loss, 10)) if self.task_obj.task_type == "lm" else 0.0
                avg_epoch_time = np.mean(epoch_times)
                avg_iter_time = avg_epoch_time / self.batches_per_epoch
                param_count_millions = algo.param_count / 1e6

                self.storage.update_trial(
                    trial_id,
                    status="completed",
                    epochs_completed=n_epochs,
                    final_loss=final_loss,
                    accuracy=final_acc,
                    perplexity=final_ppl,
                    iteration_time=avg_iter_time,
                    param_count=param_count_millions,
                )

                print(f"\n✅ Trial {trial_id} completed successfully!")
                return True

        except Exception as e:
            print(f"\n❌ Trial {trial_id} failed: {e}")
            import traceback
            traceback.print_exc()
            self.storage.update_trial(trial_id, status="failed")
            return False
