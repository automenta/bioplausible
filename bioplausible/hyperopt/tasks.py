"""
Task Abstraction for Hyperopt and Experiments

Encapsulates data loading, batch generation, and evaluation logic for different tasks.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional

from bioplausible.datasets import get_lm_dataset, get_vision_dataset
from bioplausible.training.base import BaseTrainer

class BaseTask(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, name: str, device: str = "cpu", quick_mode: bool = False):
        self.name = name
        self.device = device
        self.quick_mode = quick_mode
        self._input_dim = None
        self._output_dim = None

    @abstractmethod
    def setup(self):
        """Load datasets and prepare for training."""
        pass

    @abstractmethod
    def get_batch(self, split: str = "train", batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        pass

    @abstractmethod
    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        """Create a trainer specific to this task."""
        pass

    @property
    def input_dim(self) -> Optional[int]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return 'lm', 'vision', or 'rl'."""
        pass

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor, loss: float) -> Dict[str, float]:
        """Compute task-specific metrics."""
        return {"loss": loss}


class LMTask(BaseTask):
    """Language Modeling Task (Character level)."""

    def __init__(self, name: str = "tiny_shakespeare", device: str = "cpu", quick_mode: bool = False, seq_len: int = 64):
        super().__init__(name, device, quick_mode)
        self.seq_len = seq_len
        self.data_train = None
        self.data_val = None

    @property
    def task_type(self) -> str:
        return "lm"

    def setup(self):
        print(f"Loading LM dataset: {self.name}...")
        try:
            dataset = get_lm_dataset(self.name, seq_len=self.seq_len)
            data = dataset.data
            self._output_dim = dataset.vocab_size
            self._input_dim = None # Uses embeddings

            # Split train/val
            n = int(0.9 * len(data))
            self.data_train = data[:n]
            self.data_val = data[n:]
            print(f"Dataset ready: {len(self.data_train)} train, {len(self.data_val)} val tokens")
        except Exception as e:
            print(f"Failed to load dataset {self.name}: {e}")
            raise

    def get_batch(self, split: str = "train", batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_train is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        data = self.data_train if split == "train" else self.data_val
        idx = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + self.seq_len] for i in idx]).to(self.device)
        y = torch.stack([data[i + self.seq_len] for i in idx]).to(self.device)
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer
        return SupervisedTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor, loss: float) -> Dict[str, float]:
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        perplexity = np.exp(min(loss, 10))
        return {"loss": loss, "accuracy": acc, "perplexity": perplexity}


class VisionTask(BaseTask):
    """Vision Task (MNIST, CIFAR-10)."""

    def __init__(self, name: str = "mnist", device: str = "cpu", quick_mode: bool = False):
        super().__init__(name, device, quick_mode)
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None

    @property
    def task_type(self) -> str:
        return "vision"

    def setup(self):
        print(f"Loading Vision dataset: {self.name}...")
        try:
            dataset = get_vision_dataset(self.name, train=True, flatten=False)
            test_dataset = get_vision_dataset(self.name, train=False, flatten=False)

            # In-memory loading (replicating TrialRunner logic for speed)
            self.train_x = torch.stack([t[0] for t in dataset]).to(self.device)
            self.train_y = torch.tensor([t[1] for t in dataset]).to(self.device)

            val_size = 1000 if self.quick_mode else 5000
            self.val_x = torch.stack([test_dataset[i][0] for i in range(min(len(test_dataset), val_size))]).to(self.device)
            self.val_y = torch.tensor([test_dataset[i][1] for i in range(min(len(test_dataset), val_size))]).to(self.device)

            if self.name == "mnist":
                self._output_dim = 10
                self._input_dim = 784
            else:
                self._output_dim = 10
                self._input_dim = 3072
        except Exception as e:
            print(f"Failed to load dataset {self.name}: {e}")
            raise

    def get_batch(self, split: str = "train", batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_x is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        if split == "train":
            dataset_x, dataset_y = self.train_x, self.train_y
        else:
            dataset_x, dataset_y = self.val_x, self.val_y

        idx = torch.randint(0, len(dataset_x), (batch_size,))
        x = dataset_x[idx]
        y = dataset_y[idx]
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer
        return SupervisedTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor, loss: float) -> Dict[str, float]:
        if logits.dim() == 3:
             logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc, "perplexity": 0.0}


class RLTask(BaseTask):
    """Reinforcement Learning Task (CartPole)."""

    def __init__(self, name: str = "cartpole", device: str = "cpu", quick_mode: bool = False):
        super().__init__(name, device, quick_mode)
        self.env_name = "CartPole-v1" if name == "cartpole" else name
        self.env = None

    @property
    def task_type(self) -> str:
        return "rl"

    def setup(self):
        import gymnasium as gym
        try:
            self.env = gym.make(self.env_name)
            self._output_dim = self.env.action_space.n
            self._input_dim = self.env.observation_space.shape[0]
        except Exception as e:
            print(f"Failed to load env {self.env_name}: {e}")
            raise

    def get_batch(self, split: str = "train", batch_size: int = 32):
        raise NotImplementedError("RL Task does not support get_batch directly, use RLTrainer")

    def create_trainer(self, model, **kwargs):
        from bioplausible.training.rl import RLTrainer
        # Filter relevant args for RLTrainer
        rl_args = {}
        valid_keys = ['episodes', 'lr', 'gamma', 'max_steps']
        for k in valid_keys:
            if k in kwargs:
                rl_args[k] = kwargs[k]

        return RLTrainer(model, self.env_name, device=self.device, **rl_args)


def create_task(task_name: str, device: str = "cpu", quick_mode: bool = False) -> BaseTask:
    """Factory function for tasks."""
    if task_name in ["shakespeare", "tiny_shakespeare"]:
        return LMTask(task_name, device, quick_mode)
    elif task_name in ["mnist", "cifar10", "cifar-10"]:
        # Normalize name
        name = "cifar10" if "cifar" in task_name else "mnist"
        return VisionTask(name, device, quick_mode)
    elif task_name in ["cartpole", "rl"]:
        return RLTask("cartpole", device, quick_mode)
    else:
        # Default to LM
        print(f"Warning: Unknown task '{task_name}', defaulting to tiny_shakespeare LM")
        return LMTask("tiny_shakespeare", device, quick_mode)
