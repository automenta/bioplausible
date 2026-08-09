"""Array utilities for acceleration backends (CuPy/NumPy interop).

Thin re-export layer over :mod:`bioplausible.core.utils.activations`, which
now holds the canonical implementations of the NumPy/CuPy array helpers
(``get_backend``, ``to_numpy``, ``softmax``, ``cross_entropy`` and
``spectral_normalize``). Importing from ``core.utils`` keeps a single source
of truth while preserving this module's public surface.
"""

from bioplausible.core.utils.activations import (
    cross_entropy,
    get_backend,
    softmax,
    spectral_normalize,
    to_numpy,
)

__all__ = [
    "cross_entropy",
    "get_backend",
    "get_kernel_classes",
    "get_triton_ops",
    "softmax",
    "spectral_normalize",
    "to_numpy",
]


def get_kernel_classes() -> tuple[type[object], type[object]]:
    """Lazily import kernel classes to avoid circular imports."""
    from bioplausible.acceleration.kernels import EqPropKernel as _EqPropKernel
    from bioplausible.acceleration.kernels import (
        EqPropKernelBPTT as _EqPropKernelBPTT,
    )

    return _EqPropKernel, _EqPropKernelBPTT


def get_triton_ops() -> type[object] | None:
    """Lazily import Triton ops, returning None if unavailable."""
    try:
        from bioplausible.acceleration.triton_kernels import (
            TritonEqPropOps as _TritonEqPropOps,
        )
    except ImportError:
        return None
    return _TritonEqPropOps
