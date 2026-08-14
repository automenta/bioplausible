#!/usr/bin/env python3
"""Manual REFACTOR5 EQPROP parity benchmark: kernel backend vs PyTorch on MNIST.

The kernel backend (``EqPropKernel``, NumPy/CuPy on GPU) is the O(1)-memory
contrastive-Hebbian path; the PyTorch engine (``LoopedMLP``) is the canonical
layered engine. GATE-0 already proves *gradient/equilibrium* parity; this
benchmark checks the *learning* parity the plan requires: final MNIST accuracy
within 1 percentage point of the PyTorch path, plus the GPU time/memory story.

This is a manual benchmark (not CI-gated): it downloads MNIST, trains both
paths on a fixed subset, and reports the accuracy delta + timing.

Usage:
    uv run python tools/benchmark_kernel_parity.py [--train-samples N] [--epochs K] [--gpu]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from bioplausible.acceleration.kernels import EqPropKernel
from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

ACCURACY_BUDGET = 0.01  # 1 percentage point


def _load_mnist(n_train: int, n_test: int = 1000):
    """Return (X_train, y_train, X_test, y_test) flattened float32 arrays."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(
        root="/tmp/mnist_bench", train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root="/tmp/mnist_bench", train=False, download=True, transform=transform
    )

    def _tensor(ds, n):
        xs, ys = [], []
        for i in range(min(n, len(ds))):
            x, y = ds[i]
            xs.append(x.flatten().numpy())
            ys.append(y)
        return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)

    return _tensor(train, n_train) + _tensor(test, n_test)


def _train_kernel(X, y, X_test, y_test, epochs, batch_size, use_gpu):
    kernel = EqPropKernel(
        input_dim=X.shape[1],
        hidden_dim=128,
        output_dim=10,
        max_steps=15,
        lr=0.02,  # tuned: default 0.001 under-trains the kernel path
        use_gpu=use_gpu,
        use_spectral_norm=True,
    )
    n = X.shape[0]
    for _ in range(epochs):
        for i in range(0, n, batch_size):
            kernel.train_step(X[i : i + batch_size], y[i : i + batch_size])
    acc = kernel.evaluate(X_test, y_test)["accuracy"]
    return acc


def _train_pytorch(X, y, X_test, y_test, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoopedMLP(
        input_dim=X.shape[1],
        hidden_dim=128,
        output_dim=10,
        max_steps=15,
        gradient_method="contrastive",
        use_spectral_norm=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    x_te = torch.from_numpy(X_test).to(device)
    y_te = torch.from_numpy(y_test).to(device)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            model.train()
            opt.zero_grad()
            logits = model(x_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(x_te).argmax(dim=1)
    return float((preds == y_te).float().mean().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    X, y, X_test, y_test = _load_mnist(args.train_samples)

    t0 = time.time()
    kernel_acc = _train_kernel(
        X, y, X_test, y_test, args.epochs, args.batch_size, args.gpu
    )
    t_kernel = time.time() - t0

    t0 = time.time()
    pt_acc = _train_pytorch(X, y, X_test, y_test, args.epochs, args.batch_size)
    t_pt = time.time() - t0

    delta = abs(kernel_acc - pt_acc)
    print(
        f"kernel[{args.gpu and 'gpu' or 'cpu'}] acc: {kernel_acc:.4f} | pytorch {pt_acc:.4f} | delta {delta:.4f}"
    )
    print(f"kernel time {t_kernel:.1f}s | pytorch time {t_pt:.1f}s")
    ok = delta <= ACCURACY_BUDGET
    print(f"PARITY {'PASS' if ok else 'FAIL'} (budget {ACCURACY_BUDGET:.2f})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
