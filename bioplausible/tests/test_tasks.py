import os
import pathlib
import unittest

import pytest
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.hyperopt.tasks import VisionTask, create_task

_SKIP_ON_DOWNLOAD_KEYWORDS = ("download", "socket", "timeout", "url", "ssl")


def _dataset_available(task_name: str) -> bool:
    data_dir = os.path.join(os.getcwd(), "data")
    if task_name == "cifar10":
        return pathlib.Path(os.path.join(data_dir, "cifar-10-batches-py")).is_dir()
    if task_name == "cifar100":
        return pathlib.Path(os.path.join(data_dir, "cifar-100-python")).is_dir()
    return True


def _instantiate_backprop_mlp(
    input_dim: tuple[int, ...] | int,
    output_dim: int,
    hidden_dim: int = 32,
    num_layers: int = 1,
    **_: object,
) -> torch.nn.Module:
    cls = Registry.get(ComponentCategory.MODEL, "backprop_mlp")
    if isinstance(input_dim, tuple):
        input_dim = int(torch.prod(torch.tensor(input_dim)).item())
    return cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )


class TestVisionTask(unittest.TestCase):
    def _test_task(self, task_name, expected_channels, expected_h, expected_w):
        print(f"Testing task: {task_name}")
        if not _dataset_available(task_name):
            pytest.skip(f"Skipping {task_name}: dataset not present locally")
        try:
            task = create_task(task_name, device="cpu", quick_mode=True)
            task.setup()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in _SKIP_ON_DOWNLOAD_KEYWORDS):
                pytest.skip(f"Skipping {task_name}: dataset unavailable")
            raise

        self.assertIsInstance(task, VisionTask, f"{task_name} should be VisionTask")

        x, y = task.get_batch(split="train", batch_size=4)
        self.assertEqual(x.dim(), 4, f"{task_name}: Expected 4D input (NCHW)")
        self.assertEqual(
            x.shape[1],
            expected_channels,
            f"{task_name}: Expected {expected_channels} channel(s)",
        )
        self.assertEqual(
            x.shape[2], expected_h, f"{task_name}: Expected height {expected_h}"
        )
        self.assertEqual(
            x.shape[3], expected_w, f"{task_name}: Expected width {expected_w}"
        )

        model = _instantiate_backprop_mlp(
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            hidden_dim=32,
            num_layers=1,
            task_type="vision",
        )

        trainer = task.create_trainer(model)

        metrics = trainer.train_epoch()
        self.assertIn("loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_digits(self):
        self._test_task("digits", 1, 8, 8)

    def test_usps(self):
        self._test_task("usps", 1, 16, 16)

    def test_fashion_mnist(self):
        self._test_task("fashion_mnist", 1, 28, 28)

    def test_cifar10(self):
        self._test_task("cifar10", 3, 32, 32)

    def test_cifar100(self):
        self._test_task("cifar100", 3, 32, 32)


if __name__ == "__main__":
    unittest.main()
