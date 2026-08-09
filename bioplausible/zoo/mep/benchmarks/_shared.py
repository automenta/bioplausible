"""Shared dataclasses & helpers for MEP benchmarks.

The following were previously duplicated between ``compare.py`` and
``tuned_compare.py``: ``BenchmarkConfig``, ``get_dataloaders``,
``get_input_dim``, ``get_num_classes`` and the ``cnn`` branch of
``get_model``. Each pair of duplicates was byte-for-byte identical apart
from whitespace/comments, so one canonical copy now lives here.
"""

from dataclasses import dataclass
from typing import Protocol

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = [
    "BenchmarkConfig",
    "BuildModelFn",
    "EpochMetrics",
    "OptimizerResult",
    "cnn_classifier",
    "get_dataloaders",
    "get_input_dim",
    "get_num_classes",
]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time: float


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Results for a single optimizer."""

    name: str
    metrics: list[EpochMetrics]
    total_time: float
    best_val_acc: float
    final_train_acc: float


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    dataset: str = "mnist"
    model: str = "mlp"
    epochs: int = 10
    batch_size: int = 128
    lr: float = 0.01
    weight_decay: float = 0.0005
    subset_train: int = 5000
    subset_test: int = 1000
    device: str = "cuda"


def get_dataloaders(config: BenchmarkConfig) -> tuple[DataLoader, DataLoader]:
    """Get data loaders for the specified dataset.

    Builds MNIST / Fashion-MNIST / CIFAR-10 with their canonical
    pre-trained-normalisation transforms, then wraps them in ``Subset``
    so callers can cheaply cap iteration cost via
    ``config.subset_train``/``subset_test``.
    """
    if config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    elif config.dataset == "fashion":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_dataset = datasets.FashionMNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST("./data", train=False, transform=transform)

    elif config.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_dataset = datasets.CIFAR10(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_indices = list(range(min(config.subset_train, len(train_dataset))))
    test_indices = list(range(min(config.subset_test, len(test_dataset))))

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_subset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def get_input_dim(config: BenchmarkConfig) -> int:
    """Get input dimension for the dataset (flattened)."""
    if config.dataset == "cifar10":
        return 3072
    return 784


def get_num_classes(config: BenchmarkConfig) -> int:
    """Get number of classes for the dataset.

    Every dataset currently supported (MNIST, Fashion-MNIST, CIFAR-10) is a
    10-class classification task, so the lookup collapses to ``10``.
    """
    return 10


def cnn_classifier(input_channels: int, side: int, num_classes: int) -> nn.Module:
    """Two-conv-block CNN classifier used by all ``cnn`` benchmark variants.

    Args:
        input_channels: Image channel count (1 for MNIST/Fashion, 3 for CIFAR).
        side: Spatial size of the feature map entering the first ``Linear``
            (e.g. 8 for CIFAR after two ``MaxPool2d(2)`` from 32×32, 7 for
            MNIST after two pools from 28×28).
        num_classes: Number of output logits.
    """
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * side * side, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


class BuildModelFn(Protocol):
    """Protocol for a ``get_model(config, input_dim, num_classes) -> nn.Module`` factory."""

    def __call__(
        self, config: BenchmarkConfig, input_dim: int, num_classes: int
    ) -> nn.Module: ...
