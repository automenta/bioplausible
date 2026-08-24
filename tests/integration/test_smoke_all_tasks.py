import os
import pathlib
import unittest

import pytest
import torch
from torch import nn

from computronium.domains import create_task

_SKIP_ON_DOWNLOAD_KEYWORDS = ("download", "socket", "timeout", "url", "ssl")
_OFFLINE_VISION_DATASETS = {
    "cifar10",
    "cifar100",
    "svhn",
    "kmnist",
    "usps",
    "fashion_mnist",
}


def _dataset_available(task_name: str) -> bool:
    """Check whether the dataset's local files already exist."""
    data_dir = os.path.join(os.getcwd(), "data")
    if task_name == "cifar10":
        return pathlib.Path(os.path.join(data_dir, "cifar-10-batches-py")).is_dir()
    if task_name == "cifar100":
        return pathlib.Path(os.path.join(data_dir, "cifar-100-python")).is_dir()
    return True


class TestSmokeAllTasks(unittest.TestCase):
    def _test_task(self, task_name, task_type):
        print(f"\n>>> Smoke Testing Task: {task_name} ({task_type})")
        if (
            task_type == "vision"
            and task_name in _OFFLINE_VISION_DATASETS
            and not _dataset_available(task_name)
        ):
            pytest.skip(f"Skipping {task_name}: dataset not present locally")
        try:
            task = create_task(task_name, device="cpu", quick_mode=True)
            task.setup()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in _SKIP_ON_DOWNLOAD_KEYWORDS):
                pytest.skip(f"Skipping {task_name}: dataset unavailable")
            self.fail(f"{task_name} setup failed: {e}")

        # Basic Check
        if task_type == "vision" or task_type == "lm":
            x, y = task.get_batch(split="train", batch_size=2)
            self.assertEqual(x.shape[0], 2)

        # Create minimal model
        input_dim = task.input_dim
        output_dim = task.output_dim

        # Simple MLP
        if task_type == "vision":
            # Flatten input if needed
            input_flat = input_dim
            if isinstance(input_flat, tuple):
                import math

                input_flat = math.prod(input_flat)
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_flat, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
            )
        elif task_type == "rl":
            # RL expects (B, Obs) -> (B, Actions) logits
            model = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, output_dim)
            )
        else:  # LM
            # LMTask input_dim is None (uses embeddings).
            # output_dim is vocab size.
            # Model expects (B, T).
            class SimpleLM(nn.Module):
                def __init__(self, vocab_size):
                    super().__init__()
                    self.emb = nn.Embedding(vocab_size, 16)
                    self.head = nn.Linear(16, vocab_size)

                def forward(self, x):
                    if x.dtype in [
                        torch.float32,
                        torch.float64,
                        torch.float16,
                        torch.bfloat16,
                    ]:
                        x = x.long()
                    return self.head(self.emb(x))  # (B, T, V)

            model = SimpleLM(output_dim)

        # Trainer
        try:
            trainer = task.create_trainer(model)
            # Run one epoch (or episode)
            # For RL, episodes_per_epoch=1 to be fast
            if task_type == "rl":
                trainer.episodes_per_epoch = 1

            # Monkey patch episodes if needed
            metrics = trainer.train_epoch()
            self.assertIn("loss", metrics)
            print(f"    ✓ {task_name} passed with loss {metrics['loss']:.4f}")
        except Exception as e:
            self.fail(f"{task_name} training failed: {e}")

    # Vision
    def test_vision_digits(self):
        self._test_task("digits", "vision")

    def test_vision_usps(self):
        self._test_task("usps", "vision")

    def test_vision_kmnist(self):
        try:
            self._test_task("kmnist", "vision")
        except Exception as e:
            if "kmnist" in str(e).lower() and "download" in str(e).lower():
                self.skipTest("Skipping KMNIST due to download timeout")
            else:
                raise

    def test_vision_mnist(self):
        self._test_task("mnist", "vision")

    def test_vision_fashion(self):
        self._test_task("fashion_mnist", "vision")

    # def test_vision_svhn(self): self._test_task("svhn", "vision")
    def test_vision_cifar10(self):
        self._test_task("cifar10", "vision")

    # def test_vision_cifar100(self): self._test_task("cifar100", "vision")

    # LM
    def test_lm_tiny_shakespeare(self):
        self._test_task("tiny_shakespeare", "lm")

    def test_lm_char_ngram(self):
        self._test_task("char_ngram", "lm")

    # RL
    def test_rl_cartpole(self):
        self._test_task("cartpole", "rl")

    def test_rl_acrobot(self):
        self._test_task("acrobot", "rl")

    # Pendulum is now fixed (Continuous RLTrainer)
    def test_rl_pendulum(self):
        self._test_task("pendulum", "rl")


if __name__ == "__main__":
    unittest.main()
