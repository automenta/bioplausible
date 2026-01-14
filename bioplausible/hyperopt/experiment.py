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
from bioplausible.datasets import get_lm_dataset
from bioplausible.models.base import ModelConfig
from bioplausible.models.factory import create_model
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.rl.trainer import RLTrainer


from .storage import HyperoptStorage
from .metrics import TrialMetrics


class ExperimentAlgorithm:
    """
    Wrapper for models to unify interface for experiment runner.
    Replaces legacy AlgorithmWrapper.
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
        # Note: Some bio-plausible models (StandardFA) manage their own optimizers.
        # But we create a default one for standard models.
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
                # Input x is [batch, seq_len]
                h = self.embed(x).mean(dim=1)
            else:
                # Vision or direct input: flatten if needed for MLP-like models
                # But keep 2D/3D for Conv models if we support them later
                # For now, our models (LoopedMLP, FA, CHL, Hebbian) expect vector input (batch, dim)
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
                # Models like StandardFA/CHL have their own update rules
                metrics = self.model.train_step(h, y)
                loss = metrics.get("loss", 0.0)
                acc = metrics.get("accuracy", 0.0)
                # They manage their own optimization step
            else:
                # Standard forward/backward (Backprop, LoopedMLP via autograd/EqProp)
                # Note: LoopedMLP via ExperimentAlgorithm uses standard backprop
                # unless we wrap it in EqPropTrainer or use its specialized methods.
                # However, LoopedMLP has 'gradient_method' which defaults to 'equilibrium' or 'bptt'.
                # ExperimentAlgorithm here seems to assume a simple forward/backward interface.
                # For LoopedMLP, backward() triggers the hook if gradient_method='equilibrium'.

                # Handling specialized arguments (steps)
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
                     acc = 0.0 # Placeholder for RL/Regression

                loss = loss.item()

            # VRAM estimate
            if torch.cuda.is_available() and "cuda" in self.device:
                vram = torch.cuda.memory_allocated() / 1e9
            else:
                vram = (self.param_count * 4) / 1e9

            # Helper class for result
            class TrainingState:
                def __init__(
                    self, loss, accuracy, perplexity, iter_time, vram_gb, step
                ):
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
        self.task = task
        self.quick_mode = quick_mode

        # Training config
        self.epochs = GLOBAL_CONFIG.epochs

        if GLOBAL_CONFIG.quick_mode:
            self.batches_per_epoch = 100
            self.eval_batches = 20
        else:
            self.batches_per_epoch = 200
            self.eval_batches = 50

        # RL Config
        self.rl_episodes = 20 if quick_mode else 100

        self.epochs = GLOBAL_CONFIG.epochs
        self.batch_size = 32
        self.seq_len = 64

        # Dimensions
        self.input_dim = None # For Vision/RL
        self.output_dim = None # Previously vocab_size

        # Load data using new dataset utils
        print(f"Loading {task} dataset...")

        if task == "tiny_shakespeare" or task == "shakespeare":
            try:
                self.dataset = get_lm_dataset("tiny_shakespeare", seq_len=self.seq_len)
                self.data = self.dataset.data
                self.output_dim = self.dataset.vocab_size
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                raise e

            # Split train/val
            n = int(0.9 * len(self.data))
            self.data_train = self.data[:n]
            self.data_val = self.data[n:]
            print(f"Dataset ready: {len(self.data_train)} train, {len(self.data_val)} val tokens")

        elif task in ["mnist", "cifar10"]:
            from bioplausible.datasets import get_vision_dataset
            # Vision tasks use flattened inputs for MLP or 2D for Conv

            # Load vision data
            self.train_dataset = get_vision_dataset(task, train=True, flatten=False)
            self.test_dataset = get_vision_dataset(task, train=False, flatten=False)

            # Simple in-memory approach for now
            self.train_x = torch.stack([t[0] for t in self.train_dataset]).to(self.device)
            self.train_y = torch.tensor([t[1] for t in self.train_dataset]).to(self.device)

            # Subsample val for speed
            val_size = 1000
            self.val_x = torch.stack([self.test_dataset[i][0] for i in range(val_size)]).to(self.device)
            self.val_y = torch.tensor([self.test_dataset[i][1] for i in range(val_size)]).to(self.device)

            if task == "mnist":
                self.output_dim = 10
                self.input_dim = 784 # Flattened
            else:
                self.output_dim = 10
                self.input_dim = 3072 # Flattened

        elif task == "cartpole":
            # RL Environment
            import gymnasium as gym
            self.env = gym.make("CartPole-v1")
            self.output_dim = self.env.action_space.n
            self.input_dim = self.env.observation_space.shape[0]

        else:
            raise ValueError(f"Unknown task: {task}")

    def get_batch(self, data, device):
        """Get a random batch."""
        if self.task in ["shakespeare", "tiny_shakespeare"]:
            idx = torch.randint(0, len(data) - self.seq_len - 1, (self.batch_size,))
            x = torch.stack([data[i : i + self.seq_len] for i in idx]).to(device)
            y = torch.stack([data[i + self.seq_len] for i in idx]).to(device)
            return x, y

        elif self.task in ["mnist", "cifar10"]:
            # Data is (x, y) tensors
            if data == "train":
                dataset_x, dataset_y = self.train_x, self.train_y
            else:
                dataset_x, dataset_y = self.val_x, self.val_y

            idx = torch.randint(0, len(dataset_x), (self.batch_size,))
            x = dataset_x[idx]
            y = dataset_y[idx]

            return x, y

        return None, None

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

        # Update status
        self.storage.update_trial(trial_id, status="running")

        try:
            # Create model using wrapper
            spec = get_model_spec(trial.model_name)

            config = trial.config
            hidden_dim = config.get("hidden_dim", 128)
            num_layers = config.get("num_layers", 4)

            # Determine task type
            if self.task in ["shakespeare", "tiny_shakespeare"]:
                task_type = "lm"
            elif self.task in ["mnist", "cifar10"]:
                task_type = "vision"
            elif self.task in ["cartpole"]:
                task_type = "rl"
            else:
                task_type = "lm"

            algo = ExperimentAlgorithm(
                spec,
                output_dim=self.output_dim,
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=self.device,
                task_type=task_type
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

            # --- Branch: Reinforcement Learning ---
            if task_type == "rl":
                print("Running RL Training Loop...")
                rl_trainer = RLTrainer(algo.model, "CartPole-v1", device=self.device, lr=lr)

                # Apply extra params to RL trainer's model if needed
                # (algo.model is reference passed to RLTrainer)

                episode_rewards = []
                epoch_times = []

                # Use "epochs" as batches of episodes or just episodes?
                # Let's map epochs to X episodes.
                episodes_per_epoch = 10
                total_epochs = self.epochs

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

                    # Store reward in "accuracy" field for visualization compatibility
                    # Store -reward in "perplexity" ? Or just 0.

                    self.storage.log_epoch(
                        trial_id, epoch, avg_loss, avg_reward, 0.0, epoch_time
                    )

                    print(
                        f"Epoch {epoch+1}/{total_epochs}: "
                        f"loss={avg_loss:.4f}, avg_reward={avg_reward:.2f}, "
                        f"time={epoch_time:.1f}s"
                    )

                    # Pruning (if reward is too low after some time)
                    if pruning_callback:
                         metrics = {
                            "loss": avg_loss,
                            "accuracy": avg_reward, # Mapping reward to accuracy for metric genericism
                            "perplexity": 0.0,
                            "time": epoch_time,
                            "iteration_time": epoch_time / episodes_per_epoch,
                        }
                         if pruning_callback(trial_id, epoch + 1, metrics):
                            print(f"✂️ Trial {trial_id} PRUNED")
                            self.storage.update_trial(trial_id, status="pruned")
                            return False

                # Final stats
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

            # --- Branch: Supervised Learning (LM / Vision) ---
            else:
                epoch_times = []
                n_epochs = self.epochs

                for epoch in range(n_epochs):
                    epoch_start = time.time()

                    # Training
                    algo.model.train()
                    train_losses = []

                    for _ in range(self.batches_per_epoch):
                        x, y = self.get_batch(self.data_train, self.device)
                        state = algo.train_step(x, y, epoch * self.batches_per_epoch + _)
                        train_losses.append(state.loss)

                    # Validation
                    algo.model.eval()
                    val_losses = []
                    val_accs = []

                    with torch.no_grad():
                        for _ in range(self.eval_batches):
                            x, y = self.get_batch(self.data_val, self.device)

                            if algo.has_embed:
                                h = algo.embed(x).mean(dim=1)
                                # Simple forward
                                logits = algo.model(h)
                            else:
                                # Vision or direct input
                                # For Vision MLPs, we need to flatten if shape is [B, C, H, W]
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
                                if logits.dim() == 3:
                                    logits = logits[:, -1, :]

                            loss = algo.criterion(logits, y)
                            acc = (logits.argmax(1) == y).float().mean()

                            val_losses.append(loss.item())
                            val_accs.append(acc.item())

                    epoch_time = time.time() - epoch_start
                    epoch_times.append(epoch_time)

                    avg_val_loss = np.mean(val_losses)
                    avg_val_acc = np.mean(val_accs)
                    avg_val_ppl = np.exp(min(avg_val_loss, 10)) if task_type == "lm" else 0.0

                    # Log epoch
                    self.storage.log_epoch(
                        trial_id, epoch, avg_val_loss, avg_val_acc, avg_val_ppl, epoch_time
                    )

                    print(
                        f"Epoch {epoch+1}/{n_epochs}: "
                        f"loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}, "
                        f"ppl={avg_val_ppl:.2f}, time={epoch_time:.1f}s"
                    )

                    # Check for pruning
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

                # Final metrics
                final_loss = np.mean(val_losses)
                final_acc = np.mean(val_accs)
                final_ppl = np.exp(min(final_loss, 10)) if task_type == "lm" else 0.0
                avg_epoch_time = np.mean(epoch_times)
                avg_iter_time = avg_epoch_time / self.batches_per_epoch
                param_count_millions = algo.param_count / 1e6

                # Update trial
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
                print(f"   Final Accuracy: {final_acc:.4f}")
                print(f"   Final Perplexity: {final_ppl:.2f}")
                print(f"   Avg Iter Time: {avg_iter_time*1000:.1f}ms")
                print(f"   Param Count: {param_count_millions:.2f}M\n")

                return True

        except Exception as e:
            print(f"\n❌ Trial {trial_id} failed: {e}")
            import traceback

            traceback.print_exc()

            self.storage.update_trial(trial_id, status="failed")
            return False
