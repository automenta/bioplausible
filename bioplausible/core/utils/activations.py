"""Canonical activation and spectral-norm utilities.

Single source of truth for activation lookup, approximate spectral norm
(power iteration), and the NumPy/CuPy array-library helpers that several
``acceleration`` modules previously duplicated.

Centralising these utilities removes ad-hoc ``match`` blocks and inline
power-iteration code that had drifted out of sync between
``core/model.py``, ``equitile/core/model.py`` and ``acceleration/_array_ops.py``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "ActivationName",
    "approx_spectral_norm",
    "cross_entropy",
    "get_activation",
    "get_backend",
    "softmax",
    "spectral_normalize",
    "to_numpy",
]


type ActivationName = Literal["silu", "relu", "tanh", "gelu", "mish"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "mish": nn.Mish,
}


def get_activation(
    name: str,
    default: str | type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Return an instantiated activation module by name.

    Lookup is case-insensitive. Unknown names fall back to ``default``:
    either an activation name string (e.g. ``"relu"``, ``"gelu"``) or an
    ``nn.Module`` class. This lets call-sites preserve their original
    default-activation policy (``core`` historically defaulted to ``ReLU``,
    ``equitile`` to ``GELU``) while sharing one implementation.

    Args:
        name: Activation name (e.g. ``"silu"``, ``"gelu"``).
        default: Fallback activation name or ``nn.Module`` class.

    Returns:
        A new instance of the requested activation module.
    """
    cls = _ACTIVATIONS.get(name.lower())
    if cls is None:
        if isinstance(default, str):
            cls = _ACTIVATIONS.get(default.lower(), nn.ReLU)
        else:
            cls = default
    return cls()


def approx_spectral_norm(weight: torch.Tensor, n_iter: int = 10) -> float:
    """Approximate the largest singular value of ``weight`` via power iteration.

    Faster than a full SVD and sufficiently accurate for Lipschitz audits.
    Returns ``0.0`` for scalar/0-D tensors (which have no spectral norm).

    Args:
        weight: Weight tensor; the matrix is formed by flattening all but
            the leading dimension.
        n_iter: Number of power-iteration steps.

    Returns:
        Estimated spectral norm (largest singular value) as a Python float.
    """
    if weight.dim() < 2:  # ruff: ignore[magic-value-comparison] - dim is structurally 0/1 here
        return 0.0

    w_mat = weight.view(weight.size(0), -1)
    out_dim, _in_dim = w_mat.shape

    u = torch.randn(out_dim, device=weight.device, dtype=weight.dtype)

    for _ in range(n_iter):
        v = F.normalize(torch.mv(w_mat.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.mv(w_mat, v), dim=0, eps=1e-12)

    return torch.dot(u, torch.mv(w_mat, v)).item()


# ─── NumPy/CuPy array-library helpers ────────────────────────────────────
#
# These are the canonical implementations previously duplicated in
# ``acceleration/_array_ops.py`` and ``acceleration/kernels.py``. Importing
# ``backends`` lazily-locally would create a cycle so the cupy availability
# flag is read inside each function.


def _has_cupy() -> bool:
    from bioplausible.acceleration.backends import HAS_CUPY

    return HAS_CUPY


def get_backend(use_gpu: bool) -> object:
    """Return the appropriate array library (CuPy when GPU+CUDA, else NumPy)."""
    if use_gpu and _has_cupy():
        import cupy as cp

        return cp
    return np


def to_numpy(arr: object) -> np.ndarray:
    """Convert a NumPy or CuPy array to ``np.ndarray`` (no-op for NumPy)."""
    if _has_cupy():
        try:
            import cupy as cp

            if hasattr(arr, "__class__") and arr.__class__.__module__.startswith(
                "cupy"
            ):
                return cp.asnumpy(arr)
        except ImportError:
            pass
    return arr  # type: ignore[return-value]


def softmax(x: np.ndarray, xp: object = None) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    if xp is None:
        xp = np
    x_max = xp.max(x, axis=-1, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray, xp: object = None) -> float:
    """Mean cross-entropy loss from logits and integer targets."""
    if xp is None:
        xp = np
    batch_size = logits.shape[0]
    probs = softmax(logits, xp)
    probs = xp.clip(probs, 1e-10, 1.0)
    log_probs = xp.log(probs)
    loss = -xp.sum(log_probs[xp.arange(batch_size), targets]) / batch_size
    return float(loss)


def spectral_normalize(
    w_matrix: np.ndarray,
    num_iters: int = 5,
    u: np.ndarray | None = None,
    xp: object = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Power-iteration spectral normalization of weight matrix ``W``.

    Normalizes ``W`` by its largest singular value (≈1 operator norm), which
    keeps the Lipschitz constant of the layer below 1.

    Args:
        w_matrix: Weight matrix ``[out_dim, in_dim]``.
        num_iters: Power-iteration steps (one is typically enough).
        u: Optional warm-start left singular vector.
        xp: Array module (``np`` or ``cp``); defaults to NumPy.

    Returns:
        ``(W_normalized, u_new, sigma)`` where ``sigma`` is the estimated
        spectral norm.
    """
    if xp is None:
        xp = np
    out_dim, _in_dim = w_matrix.shape

    if u is None:
        u = xp.random.randn(out_dim).astype(w_matrix.dtype)
    u = u / xp.linalg.norm(u)

    for _ in range(num_iters):
        v = w_matrix.T @ u
        v = v / (xp.linalg.norm(v) + 1e-12)
        u = w_matrix @ v
        u = u / (xp.linalg.norm(u) + 1e-12)

    sigma = float(u @ w_matrix @ v)
    w_normalized = w_matrix / (sigma + 1e-12)

    return w_normalized, u, sigma
