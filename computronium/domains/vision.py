"""
Vision Domain Tasks

Standard vision datasets (MNIST, CIFAR, ImageNet, etc.)
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from computronium.data.transforms import build_transform, normalization
from computronium.data.vision import get_vision_dataset
from computronium.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)

__all__ = [
    "SplitMNIST",
    "VisionTask",
]


class VisionTask(DomainTask):
    """Vision domain tasks (MNIST, CIFAR, ImageNet, etc.)."""

    def __init__(
        self,
        name: str = "mnist",
        dataset_name: str = "mnist",
        data_dir: str = "./data",
        train_transform=None,
        val_transform=None,
        download: bool = True,
        num_workers: int = 2,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.download = download
        self.num_workers = num_workers

    @property
    def domain_type(self) -> DomainType:
        return DomainType.VISION

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.VISION,
            description=f"Vision task: {self.dataset_name}",
            default_metrics=["accuracy", "loss"],
            supported_tasks=["classification", "detection", "segmentation"],
            default_batch_size=64,
            default_lr=1e-3,
            requires_spatial=True,
            tags=["vision", "image"],
        )

    def setup(self) -> None:
        """Load vision datasets."""
        key = self.dataset_name.lower()
        default_transform = build_transform(key) if key in normalization else None
        if self.train_transform is None:
            self.train_transform = default_transform
        if self.val_transform is None:
            self.val_transform = default_transform

        # Load dataset based on name
        if self.dataset_name.lower() == "mnist":
            train_ds = datasets.MNIST(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.MNIST(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (1, 28, 28)
            self._output_dim = 10
        elif self.dataset_name.lower() == "cifar10":
            train_ds = datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (3, 32, 32)
            self._output_dim = 10
        elif self.dataset_name.lower() == "cifar100":
            train_ds = datasets.CIFAR100(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.CIFAR100(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (3, 32, 32)
            self._output_dim = 100
        elif self.dataset_name.lower() == "fashion_mnist":
            train_ds = datasets.FashionMNIST(
                self.data_dir,
                train=True,
                download=self.download,
                transform=self.train_transform,
            )
            val_ds = datasets.FashionMNIST(
                self.data_dir,
                train=False,
                download=self.download,
                transform=self.val_transform,
            )
            self._input_dim = (1, 28, 28)
            self._output_dim = 10
        else:
            # Fallback: use get_vision_dataset for datasets not in torchvision
            # (e.g. "digits" from scikit-learn)
            from torch.utils.data import TensorDataset

            train_ds = get_vision_dataset(self.dataset_name, train=True)
            val_ds = get_vision_dataset(self.dataset_name, train=False)

            if isinstance(train_ds, TensorDataset):
                train_x, train_y = train_ds.tensors
                val_x, val_y = val_ds.tensors
            else:
                train_x, train_y = train_ds.data, train_ds.targets
                val_x, val_y = val_ds.data, val_ds.targets

            if isinstance(train_y, (list, np.ndarray)):
                train_y = torch.tensor(train_y)
                val_y = torch.tensor(val_y)

            if not isinstance(train_x, torch.Tensor):
                train_x = torch.from_numpy(train_x)
                val_x = torch.from_numpy(val_x)

            # Normalize uint8 images to [0, 1]
            if train_x.dtype == torch.uint8:
                train_x = train_x.float() / 255.0
                val_x = val_x.float() / 255.0
            elif train_x.dtype != torch.float32:
                train_x = train_x.float()
                val_x = val_x.float()

            if train_x.dim() == 3:
                train_x = train_x.unsqueeze(1)
                val_x = val_x.unsqueeze(1)

            self._input_dim = tuple(train_x.shape[1:])
            self._output_dim = int(train_y.max().item() + 1)

            train_ds = TensorDataset(train_x, train_y)
            val_ds = TensorDataset(val_x, val_y)

        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self._test_loader = self._val_loader
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> DataLoader:
        if not self._setup_done:
            self.setup()

        if split == TaskSplit.TRAIN:
            return self._train_loader
        elif split in (TaskSplit.VAL, TaskSplit.TEST):
            return self._val_loader
        else:
            return self._train_loader

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: int | None = None,
    ) -> Metrics:
        was_training = model.training
        model.eval()
        try:
            loader = self.get_dataloader(split)

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for i, (inputs, targets) in enumerate(loader):
                    if max_batches and i >= max_batches:
                        break

                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = self.compute_loss(outputs, targets)

                    total_loss += loss.item() * inputs.size(0)
                    total_correct += (outputs.argmax(1) == targets).sum().item()
                    total_samples += inputs.size(0)

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        finally:
            if was_training:
                model.train()

        return Metrics(loss=avg_loss, accuracy=accuracy)


class SplitMNIST(DomainTask):
    """Split-MNIST continual learning benchmark.

    Splits MNIST 10-class classification into 5 binary tasks:
    - Task 0: 0 vs 1
    - Task 1: 2 vs 3
    - Task 2: 4 vs 5
    - Task 3: 6 vs 7
    - Task 4: 8 vs 9

    Each task is a binary classification problem. The continual learning
    challenge is to learn all 5 tasks sequentially without catastrophic forgetting.
    """

    def __init__(
        self,
        name: str = "split_mnist",
        data_dir: str = "./data",
        download: bool = True,
        num_workers: int = 2,
        task_id: int | None = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.data_dir = data_dir
        self.download = download
        self.num_workers = num_workers
        self.task_id = task_id
        self._full_train_ds = None
        self._full_test_ds = None
        self._class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    @property
    def domain_type(self) -> DomainType:
        return DomainType.VISION

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.VISION,
            description="Split-MNIST continual learning (5 binary tasks)",
            default_metrics=["accuracy", "loss", "backward_transfer", "forgetting"],
            supported_tasks=["continual_learning", "classification"],
            default_batch_size=64,
            default_lr=1e-3,
            requires_spatial=True,
            tags=["vision", "continual", "mnist"],
        )

    def _get_task_classes(self, task_id: int) -> tuple[int, int]:
        """Get the two classes for a given task ID."""
        if task_id < 0 or task_id >= len(self._class_pairs):
            raise ValueError(f"task_id must be in [0, 4], got {task_id}")
        return self._class_pairs[task_id]

    def _filter_dataset(self, dataset, class_a: int, class_b: int):
        """Filter dataset to only include the two target classes."""
        targets = dataset.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        mask = (targets == class_a) | (targets == class_b)
        indices = torch.where(mask)[0]
        filtered_data = dataset.data[indices]
        filtered_targets = targets[indices]
        # Remap targets to 0 and 1
        filtered_targets = (filtered_targets == class_b).long()
        return torch.utils.data.TensorDataset(
            filtered_data.unsqueeze(1).float() / 255.0, filtered_targets
        )

    def setup(self) -> None:
        """Load full MNIST and create task-specific splits."""
        from torchvision import datasets

        # Load full MNIST datasets
        self._full_train_ds = datasets.MNIST(
            self.data_dir,
            train=True,
            download=self.download,
            transform=None,
        )
        self._full_test_ds = datasets.MNIST(
            self.data_dir,
            train=False,
            download=self.download,
            transform=None,
        )

        # If task_id is specified, set up only that task
        # Otherwise, we'll set up on demand in get_dataloader
        self._input_dim = (1, 28, 28)
        self._output_dim = 2  # Binary classification
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> DataLoader:
        if not self._setup_done:
            self.setup()

        if self.task_id is None:
            raise ValueError(
                "SplitMNIST requires a task_id to be set. "
                "Use set_task(task_id) or pass task_id at construction."
            )

        class_a, class_b = self._get_task_classes(self.task_id)

        if split == TaskSplit.TRAIN:
            ds = self._filter_dataset(self._full_train_ds, class_a, class_b)
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        else:
            ds = self._filter_dataset(self._full_test_ds, class_a, class_b)
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def set_task(self, task_id: int) -> None:
        """Set the active task for this SplitMNIST instance."""
        if task_id < 0 or task_id >= len(self._class_pairs):
            raise ValueError(f"task_id must be in [0, 4], got {task_id}")
        self.task_id = task_id
        # Reset loaders to force recreation with new task
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def get_all_task_loaders(
        self, split: TaskSplit = TaskSplit.TEST
    ) -> list[DataLoader]:
        """Get dataloaders for all 5 tasks (useful for evaluation)."""
        loaders = []
        for task_id in range(5):
            self.set_task(task_id)
            loaders.append(self.get_dataloader(split))
        return loaders

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: int | None = None,
    ) -> Metrics:
        """Evaluate on the current task."""
        if self.task_id is None:
            raise ValueError("task_id must be set before evaluation")
        return super().evaluate(model, split, max_batches)

    def evaluate_all_tasks(
        self, model: nn.Module, split: TaskSplit = TaskSplit.TEST
    ) -> list[Metrics]:
        """Evaluate model on all 5 tasks."""
        results = []
        for task_id in range(5):
            self.set_task(task_id)
            results.append(self.evaluate(model, split))
        return results
