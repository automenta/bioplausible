"""Task selector backends (Sprint 3.3).

Each task exposes a tiny supervised interface: ``input_dim``, ``output_dim``,
``train_batch()``, ``eval_batch()``. Toy tasks are generated on the fly (no
downloads); real tasks (MNIST) lazily load from torchvision's cache via the
project's domain layer. All loaders are GPU-transfer-ready through the project
``device`` fixture convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_MNIST_CACHE: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class TaskSpec:
    """Descriptor + sampler for one demo task."""

    name: str
    input_dim: int
    output_dim: int
    kind: str  # "toy" | "digits" | "mnist" | "lm"
    _gen: object | None = None

    def sample(
        self, batch: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) batch. ``_gen`` is a callable(batch, device) -> pair."""
        if self._gen is None:
            raise NotImplementedError(f"task {self.name!r} has no sampler")
        return self._gen(batch, device)


def _xor(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, 2, (batch, 2), device=device).float()
    y = (x[:, 0] != x[:, 1]).long()
    return x, y


def _spiral(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = batch
    theta = torch.linspace(0, 4 * math.pi, n, device=device)
    r = torch.linspace(0.1, 1.0, n, device=device)
    x0 = (r * torch.cos(theta) + torch.randn(n, device=device) * 0.05)
    x1 = (r * torch.sin(theta) + torch.randn(n, device=device) * 0.05)
    x = torch.stack([x0, x1], dim=1)
    y = (theta > 2 * math.pi).long()
    return x, y


def _circles(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.rand(batch, device=device)
    b = torch.rand(batch, device=device)
    r = torch.where(a < 0.5, 0.2, 0.8) + torch.randn(batch, device=device) * 0.03
    th = 2 * math.pi * b
    x = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1)
    y = (r > 0.5).long()
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


def build_tasks() -> list[TaskSpec]:
    """Construct the demo task card (toy + digits + MNIST)."""
    return [
        TaskSpec("xor", 2, 2, "toy", _xor),
        TaskSpec("spiral", 2, 2, "toy", _spiral),
        TaskSpec("circles", 2, 2, "toy", _circles),
        TaskSpec("digits", 64, 10, "digits", _digits),
        TaskSpec("mnist", 784, 10, "mnist", _mnist),
    ]
