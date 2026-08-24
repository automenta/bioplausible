"""Canonical torchvision transform pipelines (REFACTOR.md §1).

Single source of truth for image preprocessing. Every ``Compose`` previously
inlined at benchmark/track/dataset sites now references these constants, so
per-dataset statistics (mean/std) and augmentation arguments live in exactly
one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CIFAR10_TRANSFORM",
    "CIFAR100_TRANSFORM",
    "FASHION_MNIST_TRANSFORM",
    "KMNIST_TRANSFORM",
    "MNIST_TRANSFORM",
    "SVHN_TRANSFORM",
    "build_transform",
    "create_dataloader",
    "normalization",
]


#: Canonical per-channel dataset statistics (mean, std) in [0, 1] pixel space.
normalization: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "kmnist": ((0.1904,), (0.3476,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "usps": ((0.5,), (0.5,)),
}

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalization["mnist"]),
])
FASHION_MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalization["fashion_mnist"]),
])
KMNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalization["kmnist"]),
])
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalization["cifar10"]),
])
CIFAR100_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalization["cifar100"]),
])
SVHN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalization["svhn"]),
])


def build_transform(
    name: str,
    *,
    flatten: bool = False,
    augment: bool = False,
) -> transforms.Compose:
    """Build the canonical transform pipeline for a dataset.

    Args:
        name: Canonical dataset key (see :data:`normalization`).
        flatten: Append a ``Layout``-free reshape to ``(-1,)`` after
            normalisation (consumers expect 1-D feature vectors).
        augment: Prepend the canonical training augmentation for the dataset
            (random-crop + flip for CIFAR/SVHN; random-affine for MNIST-family).

    Returns:
        A ``transforms.Compose`` of ``ToTensor`` + ``Normalize`` (+ optional
        augmentation / flatten).

    Raises:
        ValueError: If *name* is not in :data:`normalization`.
    """
    params = normalization.get(name)
    if params is None:
        raise ValueError(
            f"Unknown dataset {name!r} for canonical transforms: "
            f"{sorted(normalization)}"
        )
    steps: list[object] = []
    if augment:
        if name in {"cifar10", "cifar100", "svhn"}:
            steps.append(transforms.RandomCrop(32, padding=4))
            steps.append(transforms.RandomHorizontalFlip())
        elif name in {"mnist", "fashion_mnist", "kmnist"}:
            steps.append(transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(*params))
    if flatten:
        steps.append(transforms.Lambda(lambda x: x.view(-1)))
    return transforms.Compose(steps)  # type: ignore[arg-type]


def create_dataloader(  # ruff: ignore[too-many-arguments] - canonical DataLoader factory
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """Canonical ``DataLoader`` construction.

    ``persistent_workers`` is silently dropped when ``num_workers == 0`` (the
    default) because torch rejects a persistent worker pool with no workers.
    """
    workers = num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and workers > 0,
    )
