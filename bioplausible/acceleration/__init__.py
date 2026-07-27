"""
Acceleration Module for Bioplausible

Provides multiple acceleration backends for Equilibrium Propagation:

Backends (in order of priority for speed):
    1. Triton Kernels: Custom GPU kernels for fused operations (fastest)
    2. CuPy: NumPy-compatible GPU arrays via CUDA
    3. torch.compile: PyTorch 2.0+ JIT compilation
    4. Pure PyTorch: Standard autograd (fallback)
    5. Pure NumPy: CPU-only kernel (portability)

Usage:
    from bioplausible.acceleration import (
        get_optimal_backend,
        compile_model,
        HAS_CUPY,
        HAS_TRITON,
    )

    # Check available backends
    print(f"CuPy: {HAS_CUPY}, Triton: {HAS_TRITON}")

    # Use optimal backend
    device = get_optimal_backend()

    # Compile model for speed
    model = compile_model(model, mode='reduce-overhead')
"""

from bioplausible.acceleration._array_ops import (
    cross_entropy,
    get_backend,
    get_kernel_classes,
    get_triton_ops,
    softmax,
    spectral_normalize,
    to_numpy,
)
from bioplausible.acceleration.backends import (
    HAS_CUPY,
    HAS_TRITON,
    BackendDetector,
    CupyChecker,
    TritonChecker,
    check_cupy_available,
    check_triton_available,
    enable_tf32,
    get_optimal_backend,
)
from bioplausible.acceleration.compile import compile_model, compile_settling_loop

__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "BackendDetector",
    "CupyChecker",
    "TritonChecker",
    "check_cupy_available",
    "check_triton_available",
    "compile_model",
    "compile_settling_loop",
    "cross_entropy",
    "enable_tf32",
    "get_backend",
    "get_kernel_classes",
    "get_optimal_backend",
    "get_triton_ops",
    "softmax",
    "spectral_normalize",
    "to_numpy",
]
