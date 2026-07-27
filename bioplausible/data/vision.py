"""
Vision Dataset Utilities

Functions for loading and creating DataLoaders for standard vision datasets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def get_vision_dataset(
    name: str = "mnist",
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    flatten: bool = False,
    included_classes: Optional[list] = None,
    augment: bool = False,
) -> Dataset:
    """
    Load a vision dataset with standard transforms.

    Args:
        name: Dataset name ('mnist', 'fashion_mnist', 'cifar10', 'cifar100',
              'kmnist', 'svhn', 'digits')
        root: Data directory
        train: If True, load training set
        download: If True, download if not present
        flatten: If True, flatten images to 1D
        included_classes: List of class indices to include (optional)
        augment: If True, apply data augmentation for training.

    Returns:
        PyTorch Dataset
    """
    if name == "digits":
        return _load_sklearn_digits(train, flatten)

    transform = _build_transforms(name, flatten, augment=augment and train)
    dataset_class = _get_dataset_class(name)

    if name == "svhn":
        split = "train" if train else "test"
        dataset = dataset_class(
            root, split=split, download=download, transform=transform
        )
    else:
        dataset = dataset_class(
            root, train=train, download=download, transform=transform
        )

    if included_classes is not None:
        targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        indices = [i for i, t in enumerate(targets) if t in included_classes]
        from torch.utils.data import Subset

        return Subset(dataset, indices)

    return dataset


def _load_sklearn_digits(
    train: bool,
    flatten: bool,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dataset:
    """Load sklearn 8x8 digits dataset."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    X /= 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_data = X_train if train else X_test
    y_data = y_train if train else y_test

    if not flatten:
        X_data = X_data.reshape(-1, 1, 8, 8)

    return TensorDataset(torch.from_numpy(X_data), torch.from_numpy(y_data))


def _build_transforms(name: str, flatten: bool, augment: bool = False):
    """Build the appropriate transforms for the given dataset."""
    from torchvision import transforms

    transform_list = []
    if augment:
        if name in ["cifar10", "cifar100", "svhn"]:
            transform_list.append(transforms.RandomCrop(32, padding=4))
            transform_list.append(transforms.RandomHorizontalFlip())
        elif name in ["mnist", "fashion_mnist", "kmnist"]:
            transform_list.append(
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1))
            )

    transform_list.append(transforms.ToTensor())
    if name in ["mnist", "fashion_mnist", "kmnist", "usps"]:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    elif name in ["cifar10", "cifar100", "svhn"]:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    return transforms.Compose(transform_list)


def _get_dataset_class(name: str) -> type:
    """Get the appropriate dataset class for the given name."""
    from torchvision import datasets

    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "kmnist": datasets.KMNIST,
        "svhn": datasets.SVHN,
        "usps": datasets.USPS,
    }
    if name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(dataset_map.keys())} + ['digits']"
        )
    return dataset_map[name]


class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, seq_len: int = 128) -> None:
        self.seq_len = seq_len
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

    def decode(self, indices: torch.Tensor) -> str:
        return "".join(self.idx_to_char[i.item()] for i in indices)


def create_data_loaders(
    dataset_name: str = "mnist",
    batch_size: int = 64,
    num_workers: int = 0,
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders for a vision dataset."""
    train_data = get_vision_dataset(dataset_name, train=True, flatten=flatten)
    test_data = get_vision_dataset(dataset_name, train=False, flatten=flatten)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
