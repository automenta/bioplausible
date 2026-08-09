"""Device resolution helper — single source of truth for ``torch.device`` selection.

Replaces the 30+ inline ``device = "cuda" if torch.cuda.is_available() else "cpu"``
patterns sprinkled across the codebase. The default ``"auto"`` prefers CUDA,
then Apple MPS, then CPU.
"""

from __future__ import annotations

import torch

__all__ = ["get_device", "get_optimal_backend"]


def get_optimal_backend() -> str:
    """Return the best available backend name (``"cuda"`` / ``"mps"`` / ``"cpu"``)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device(
    device: str | torch.device | None = "auto",
    *,
    fallback: str = "cpu",
) -> torch.device:
    """Resolve a device hint to a ``torch.device``.

    Args:
        device: A device hint — ``"auto"`` (or ``None``) selects the best
            available backend; an explicit string like ``"cuda"`` or
            ``"cuda:0"`` is honoured as-is; an existing ``torch.device`` is
            returned untouched.
        fallback: Backend used when ``device == "auto"`` and the preferred
            backend is unavailable (e.g. CUDA missing). Defaults to ``"cpu"``.

    Returns:
        The resolved ``torch.device``.

    Note:
        When CUDA is requested via a ``"cuda"``/``"cuda:N"`` string but is
        not available, the call falls back to ``torch.device(fallback)``
        rather than raising — callers such as benchmarking harnesses that
        gate on availability rely on this graceful degradation.
    """
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device(get_optimal_backend())
    if (
        isinstance(device, str)
        and device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        return torch.device(fallback)
    return torch.device(device)
