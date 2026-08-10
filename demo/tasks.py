"""Task selector backends (Sprint 3.3).

Each task exposes a tiny supervised interface: ``input_dim``, ``output_dim``,
``train_batch()``, ``eval_batch()``. Toy tasks are generated on the fly (no
downloads); real tasks (MNIST) lazily load from torchvision's cache via the
project's domain layer. All loaders are GPU-transfer-ready through the project
``device`` fixture convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bioplausible.data.vision import generate_toy_points

_MNIST_CACHE: tuple[torch.Tensor, torch.Tensor] | None = None
_CIFAR_CACHE: tuple[torch.Tensor, torch.Tensor] | None = None
_LM_CACHE: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class TaskSpec:
    """Descriptor + sampler for one demo task."""

    name: str
    input_dim: int
    output_dim: int
    kind: str  # "toy" | "digits" | "mnist" | "cifar10" | "lm"
    _gen: object | None = None
    downloads: bool = False  # True if sample() may fetch data on first call

    def sample(
        self, batch: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) batch. ``_gen`` is a callable(batch, device) -> pair."""
        if self._gen is None:
            raise NotImplementedError(f"task {self.name!r} has no sampler")
        return self._gen(batch, device)


def _xor(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = generate_toy_points("xor", batch, device=device)
    return x, y


def _spiral(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = generate_toy_points("spiral", batch, device=device)
    return x, y


def _circles(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = generate_toy_points("circles", batch, device=device)
    return x, y


def _digits(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from sklearn import datasets as sk

    x, y = sk.load_digits(return_X_y=True)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    idx = torch.randint(0, len(xt), (batch,))
    return xt[idx].to(device), yt[idx].to(device)


def _mnist(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    global _MNIST_CACHE
    if _MNIST_CACHE is None:
        from torchvision import datasets, transforms

        ds = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        data = torch.cat([x.view(1, -1) for x, _ in ds], dim=0)
        labels = torch.tensor([y for _, y in ds], dtype=torch.long)
        _MNIST_CACHE = (data, labels)
    x, y = _MNIST_CACHE
    idx = torch.randint(0, len(x), (batch,))
    return x[idx].to(device), y[idx].to(device)


def _cifar(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    global _CIFAR_CACHE
    if _CIFAR_CACHE is None:
        from torchvision import datasets, transforms

        ds = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        data = torch.cat([x.reshape(1, -1).to(torch.float32) for x, _ in ds], dim=0)
        labels = torch.tensor([y for _, y in ds], dtype=torch.long)
        _CIFAR_CACHE = (data, labels)
    x, y = _CIFAR_CACHE
    idx = torch.randint(0, len(x), (batch,))
    return x[idx].to(device), y[idx].to(device)


def _tiny_shakespeare(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Character-level LM: (seq context, next-char). 2x flattened in/out.

    ``input_dim == output_dim == vocab_size`` so a plain MLP can imitate a
    bigram/n-gram model. The project's CharDataset yields (x[t:t+L], x[t+1:..])
    pairs, so we build a tiny in-memory cache of (ctx, next-char) samples.
    """
    global _LM_CACHE
    if _LM_CACHE is None:
        from bioplausible.data.lm import get_lm_dataset

        ds = get_lm_dataset("tiny_shakespeare", seq_len=16)
        x = ds.data[: ds.seq_len * 2000].view(-1, ds.seq_len)
        y = ds.data[: ds.seq_len * 2000 + ds.seq_len - 1][ds.seq_len - 1 :].view(
            -1, ds.seq_len
        )
        ctx = x[:-1]
        nxt = y[:-1][:, 0]
        _LM_CACHE = (ctx, nxt)
    ctx, nxt = _LM_CACHE
    idx = torch.randint(0, len(ctx), (batch,))
    return ctx[idx].to(device), nxt[idx].to(device)


def build_tasks() -> list[TaskSpec]:
    """Construct the demo task card (toy + digits + MNIST + CIFAR + LM)."""
    return [
        TaskSpec("xor", 2, 2, "toy", _xor),
        TaskSpec("spiral", 2, 2, "toy", _spiral),
        TaskSpec("circles", 2, 2, "toy", _circles),
        TaskSpec("digits", 64, 10, "digits", _digits),
        TaskSpec("mnist", 784, 10, "mnist", _mnist, downloads=True),
        TaskSpec("cifar10", 3072, 10, "cifar10", _cifar, downloads=True),
        TaskSpec("tiny_shakespeare", 16, 16, "lm", _tiny_shakespeare, downloads=True),
    ]
