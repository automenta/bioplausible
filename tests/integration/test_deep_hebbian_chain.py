"""R11.3.14 lock: deep local Hebbian chain carries signal at any depth.

Claims (measured regime in `computronium/models/native/deep_hebbian_native.py`):

1. Per-layer signal norms stay O(1) at depth 10/50/100 — the
   runaway-gain/NaN pathology of unnormalized Hebbian chains is fixed
   structurally (spectral renorm + tanh + Oja decay + activity renorm).
2. Hebbian features + linear readout far exceed chance at every depth
   when class identity lives in the chain's dominant direction.
3. Unnormalized control: signal decays to noise (the refuted baseline).
4. Honest boundary (R11.5.5 refutation slot): with 10 direction-coded
   classes the readout decays with depth but stays above chance —
   activity-subspace collapse under compounding renorm+Oja sharpening.
"""

from __future__ import annotations

import math

import pytest
import torch

from computronium.models.native import DeepHebbianChain

_INPUT_DIM = 32
_HIDDEN_DIM = 32
_BATCH = 2048
_EVAL = 512
_TOL = 0.05


def _class_directions(generator: torch.Generator, num_classes: int) -> torch.Tensor:
    if num_classes == 2:
        v = torch.randn(_INPUT_DIM, generator=generator)
        v /= v.norm()
        return torch.stack([v, -v]) * 3.0
    basis = torch.linalg.qr(torch.randn(_INPUT_DIM, num_classes, generator=generator))[
        0
    ]
    return basis.T * 3.0


def _sample(
    means: torch.Tensor, n: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.randint(0, means.shape[0], (n,), generator=generator)
    features = means[targets] + torch.randn(n, _INPUT_DIM, generator=generator) * 0.5
    return features, targets


def _centroid_accuracy(
    features: torch.Tensor,
    targets: torch.Tensor,
    eval_features: torch.Tensor,
    eval_targets: torch.Tensor,
    num_classes: int,
) -> float:
    centroids = torch.stack([
        features[targets == k].mean(0) for k in range(num_classes)
    ])
    return (
        (torch.cdist(eval_features, centroids).argmin(dim=1) == eval_targets)
        .float()
        .mean()
        .item()
    )


def _train(
    depth: int,
    *,
    normalize: bool = True,
    num_classes: int = 2,
    seed: int = 1,
) -> tuple[DeepHebbianChain, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = DeepHebbianChain(
        _INPUT_DIM,
        _HIDDEN_DIM,
        depth,
        learning_rate=1e-3,
        normalize=normalize,
    )
    generator = torch.Generator().manual_seed(seed)
    means = _class_directions(generator, num_classes)
    x_train, y_train = _sample(means, _BATCH, generator)
    x_eval, y_eval = _sample(means, _EVAL, generator)
    order = torch.randperm(_BATCH, generator=generator)
    for i in range(0, _BATCH, 64):
        model.local_update(x_train[order[i : i + 64]])
    return model, x_train, y_train, x_eval, y_eval


@pytest.mark.parametrize("depth", [10, 50, 100])
def test_signal_norm_o1_at_depth(depth: int) -> None:
    model, x_train, _, _, _ = _train(depth)
    norms = model.layer_norms(x_train[:128])
    assert all(not math.isnan(n) for n in norms), "NaN in per-layer norms"
    reference = norms[0]
    assert all(0.25 * reference < n < 4.0 * reference for n in norms), norms


@pytest.mark.parametrize("depth", [10, 50, 100])
def test_dominant_direction_readout_survives_depth(depth: int) -> None:
    model, x_train, y_train, x_eval, y_eval = _train(depth)
    acc = _centroid_accuracy(model(x_train)[-1], y_train, model(x_eval)[-1], y_eval, 2)
    assert acc > 0.9, f"depth {depth}: dominant-direction readout {acc:.3f}"


def test_unnormalized_control_signal_dies() -> None:
    model, x_train, *_ = _train(50, normalize=False)
    norms = model.layer_norms(x_train[:64])
    assert norms[-1] < 1e-6 * norms[0], norms


def test_subspace_collapse_boundary() -> None:
    """10 direction-coded classes: depth-1 readout near-perfect, deep
    readout above chance but degraded — the honest local-Hebbian depth
    boundary (module docstring, measured regime)."""
    model, x_train, y_train, x_eval, y_eval = _train(100, num_classes=10)
    acts_train = model(x_train)
    acts_eval = model(x_eval)
    acc_first = _centroid_accuracy(acts_train[1], y_train, acts_eval[1], y_eval, 10)
    acc_last = _centroid_accuracy(acts_train[-1], y_train, acts_eval[-1], y_eval, 10)
    assert acc_first > 0.9, acc_first
    assert acc_last > 0.15, acc_last  # > chance (0.1) with margin
    assert acc_last < acc_first - _TOL  # the collapse boundary is real
