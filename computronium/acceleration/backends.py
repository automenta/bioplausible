"""
Backend Detection and Configuration

Detects and configures the optimal compute backend for acceleration.
Provides auto-dispatch with profiling, benchmarking, and fallback chain.
"""

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch

from computronium.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelRegistry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "AutoDispatcher",
    "BackendBenchmark",
    "BackendDetector",
    "BackendType",
    "KernelProfiler",
    "check_cupy_available",
    "check_triton_available",
    "dispatch_kernel",
    "enable_tf32",
    "get_optimal_backend",
    "profile_kernel",
]


class BackendType(str, Enum):  # ruff: ignore[replace-str-enum]
    """Compute backend types in priority order."""

    TRITON = "triton"
    CUDA = "cuda"
    CUPY = "cupy"
    CPU = "cpu"
    NUMPY = "numpy"


@dataclass(frozen=True, slots=True)
class BackendBenchmark:
    """Benchmark results for a backend operation."""

    backend: BackendType
    operation: str
    shape: tuple[int, ...]
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_gflops: float | None = None
    memory_mb: float | None = None
    success: bool = True
    error: str | None = None


@dataclass
class KernelProfiler:
    """Profiles kernel performance across backends."""

    warmup_runs: int = 5
    benchmark_runs: int = 20
    _cache: dict[str, list[BackendBenchmark]] = field(default_factory=dict)

    def benchmark_operation(
        self,
        operation: Callable,
        shapes: list[tuple[int, ...]],
        backends: list[BackendType] | None = None,
    ) -> dict[str, list[BackendBenchmark]]:
        """Benchmark an operation across backends and shapes."""
        if backends is None:
            backends = self._get_available_backends()

        results = {}
        for shape in shapes:
            shape_key = f"{operation.__name__}_{shape}"
            results[shape_key] = []

            for backend in backends:
                bench = self._benchmark_single(operation, shape, backend)
                results[shape_key].append(bench)

        return results

    def _get_available_backends(self) -> list[BackendType]:
        """Get list of available backends in priority order."""
        backends = []

        if HAS_TRITON:
            backends.append(BackendType.TRITON)
        if torch.cuda.is_available():
            backends.append(BackendType.CUDA)
        if HAS_CUPY:
            backends.append(BackendType.CUPY)
        backends.append(BackendType.CPU)
        backends.append(BackendType.NUMPY)

        return backends

    def _benchmark_single(
        self,
        operation: Callable,
        shape: tuple[int, ...],
        backend: BackendType,
    ) -> BackendBenchmark:
        """Benchmark a single operation on a specific backend."""
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            # Prepare inputs
            inputs = self._prepare_inputs(shape, backend)

            # Warmup
            for _ in range(self.warmup_runs):
                _ = operation(*inputs)

            # Synchronize for accurate timing
            if backend in (BackendType.TRITON, BackendType.CUDA, BackendType.CUPY):  # ruff: ignore[literal-membership]
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(self.benchmark_runs):
                if backend in (BackendType.TRITON, BackendType.CUDA, BackendType.CUPY):  # ruff: ignore[literal-membership]
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = operation(*inputs)
                if backend in (BackendType.TRITON, BackendType.CUDA, BackendType.CUPY):  # ruff: ignore[literal-membership]
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # ms

            times = np.array(times)
            mean_time = float(np.mean(times))
            std_time = float(np.std(times))

            return BackendBenchmark(
                backend=backend,
                operation=operation.__name__,
                shape=shape,
                mean_time_ms=mean_time,
                std_time_ms=std_time,
                min_time_ms=float(np.min(times)),
                max_time_ms=float(np.max(times)),
            )
        except Exception as e:
            return BackendBenchmark(
                backend=backend,
                operation=operation.__name__,
                shape=shape,
                mean_time_ms=float("inf"),
                std_time_ms=0.0,
                min_time_ms=float("inf"),
                max_time_ms=float("inf"),
                success=False,
                error=str(e),
            )

    def _prepare_inputs(self, shape: tuple[int, ...], backend: BackendType):
        """Prepare inputs for benchmarking based on backend."""
        if backend in (BackendType.CUPY,):  # ruff: ignore[literal-membership]
            import cupy as cp

            return [cp.random.randn(*shape).astype(cp.float32) for _ in range(2)]
        else:
            return [
                torch.randn(
                    *shape, device="cuda" if backend != BackendType.CPU else "cpu"
                )
                for _ in range(2)
            ]

    def select_best_backend(
        self,
        operation: Callable,
        shape: tuple[int, ...],
        metric: str = "mean_time_ms",
    ) -> BackendType:
        """Select the fastest backend for an operation/shape."""
        shape_key = f"{operation.__name__}_{shape}"

        if shape_key in self._cache:
            benchmarks = self._cache[shape_key]
        else:
            benchmarks = self._benchmark_single(
                operation, shape, self._get_available_backends()[0]
            )
            benchmarks = [benchmarks]  # Simplified

        # Filter successful
        successful = [b for b in benchmarks if b.success]
        if not successful:
            return BackendType.CPU

        # Select best by metric
        best = min(successful, key=lambda b: getattr(b, metric))
        return best.backend


class AutoDispatcher:
    """Automatic backend dispatcher with fallback chain."""

    def __init__(self, profile_mode: bool = False):
        self.profile_mode = profile_mode
        self._profiler = KernelProfiler()
        self._dispatch_cache: dict[str, BackendType] = {}
        self._fallback_order = [
            BackendType.TRITON,
            BackendType.CUDA,
            BackendType.CUPY,
            BackendType.CPU,
            BackendType.NUMPY,
        ]

    def dispatch(
        self,
        algorithm: AlgorithmFamily,
        operation: str,
        *args,
        **kwargs,
    ):
        """Dispatch operation to best available backend."""
        # Get kernel backend from registry
        hardware = self._select_hardware()
        backend = KernelRegistry.get_best(algorithm, hardware)

        if backend is None:
            # Fallback to Python implementation
            return self._fallback_dispatch(algorithm, operation, *args, **kwargs)

        # Call the appropriate method
        method = getattr(backend, operation, None)
        if method is None:
            return self._fallback_dispatch(algorithm, operation, *args, **kwargs)

        try:
            return method(*args, **kwargs)
        except Exception as e:
            if self.profile_mode:
                warnings.warn(f"Backend {hardware} failed: {e}, falling back")
            return self._fallback_dispatch(algorithm, operation, *args, **kwargs)

    def _select_hardware(self) -> HardwareTarget:
        """Select best hardware target."""
        if HAS_TRITON and torch.cuda.is_available():
            return HardwareTarget.TRITON
        elif torch.cuda.is_available():
            return HardwareTarget.CUDA
        else:
            return HardwareTarget.CPU

    def _fallback_dispatch(
        self, algorithm: AlgorithmFamily, operation: str, *args, **kwargs
    ):
        """Fallback to next available backend."""
        for hw in [HardwareTarget.CUDA, HardwareTarget.CPU]:
            backend = KernelRegistry.get(algorithm, hw)
            if backend is not None:
                method = getattr(backend, operation, None)
                if method is not None:
                    try:
                        return method(*args, **kwargs)
                    except Exception:  # ruff: ignore[try-except-continue]
                        continue
        raise RuntimeError(f"No available backend for {algorithm}.{operation}")

    def get_backend_info(self, algorithm: AlgorithmFamily) -> dict:
        """Get info about available backends for an algorithm."""
        info = {
            "algorithm": algorithm.value,
            "available_backends": {},
            "recommended": None,
        }

        for hw in [HardwareTarget.TRITON, HardwareTarget.CUDA, HardwareTarget.CPU]:
            backend = KernelRegistry.get(algorithm, hw)
            if backend is not None:
                info["available_backends"][hw.value] = {
                    "class": backend.__class__.__name__,
                    "memory_complexity": getattr(
                        backend, "memory_complexity", "unknown"
                    ),
                    "locality_level": getattr(backend, "locality_level", "unknown"),
                    "supports_autograd": getattr(backend, "supports_autograd", False),
                    "requires_settle": getattr(backend, "requires_settle", False),
                }

        # Recommend best
        best = KernelRegistry.get_best(algorithm)
        if best is not None:
            info["recommended"] = best.__class__.__name__

        return info


class BackendDetector:
    """Helper class to detect the optimal compute backend."""

    @staticmethod
    def detect_best_backend() -> str:
        """Detect the best available compute backend."""
        if HAS_TRITON and torch.cuda.is_available():
            return BackendType.TRITON.value
        elif torch.cuda.is_available():
            return BackendType.CUDA.value
        elif HAS_CUPY:
            return BackendType.CUPY.value
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return BackendType.CPU.value

    @staticmethod
    def get_fallback_chain() -> list[BackendType]:
        """Get the fallback chain for the current system."""
        chain = []
        if HAS_TRITON and torch.cuda.is_available():
            chain.append(BackendType.TRITON)
        if torch.cuda.is_available():
            chain.append(BackendType.CUDA)
        if HAS_CUPY:
            chain.append(BackendType.CUPY)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chain.append(BackendType.CUDA)  # MPS uses CUDA-like path
        chain.append(BackendType.CPU)
        chain.append(BackendType.NUMPY)
        return chain


def get_optimal_backend() -> str:
    """Detect best available compute backend."""
    return BackendDetector.detect_best_backend()


def enable_tf32(enable: bool = True) -> None:
    """Enable TensorFloat-32 (TF32) for significant speedup on Ampere+ GPUs."""
    if torch.cuda.is_available():
        precision = "high" if enable else "highest"
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        torch.set_float32_matmul_precision(precision)


def check_cupy_available() -> tuple[bool, str]:
    """Check if CuPy is available with proper CUDA configuration."""
    try:
        import cupy as cp

        _ = cp.zeros(10)
        return True, "CuPy available with CUDA"  # ruff: ignore[try-consider-else]
    except ImportError:
        return False, "CuPy not installed. Install with: pip install cupy-cuda12x"
    except Exception as e:
        return False, f"CuPy installed but CUDA failed: {e}"


def check_triton_available() -> tuple[bool, str]:
    """Check if Triton is available for custom kernels."""
    if HAS_TRITON:
        return True, "Triton available"
    return False, "Triton not installed. Install with: pip install triton"


# Triton/CuPy availability
HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    if hasattr(libdevice, "tanh"):
        HAS_TRITON = True
except ImportError:
    triton = None
    tl = None

HAS_CUPY = False
try:  # ruff: ignore[too-many-statements-in-try-clause]
    import cupy as cp

    if hasattr(cp, "cuda") and cp.cuda.is_available():
        with cp.cuda.Device(0):
            _ = cp.array([1.0])
            _ = cp.random.rand(1)
        HAS_CUPY = True
    else:
        cp = None
except ImportError, Exception:
    cp = None
    HAS_CUPY = False


# High-level dispatch functions
def dispatch_kernel(
    algorithm: AlgorithmFamily,
    operation: str,
    *args,
    **kwargs,
):
    """Dispatch a kernel operation to the best available backend."""
    dispatcher = AutoDispatcher()
    return dispatcher.dispatch(algorithm, operation, *args, **kwargs)


def profile_kernel(
    algorithm: AlgorithmFamily,
    operation: str,
    shapes: list[tuple[int, ...]],
) -> dict[str, list[BackendBenchmark]]:
    """Profile a kernel operation across backends and shapes."""
    profiler = KernelProfiler()  # ruff: ignore[unused-variable]
    # This is a placeholder - actual implementation would need
    # to extract the operation from the registered backend
    return {}


# Default auto-dispatcher instance
_DEFAULT_DISPATCHER = AutoDispatcher()


def get_dispatcher() -> AutoDispatcher:
    """Get the default auto-dispatcher."""
    return _DEFAULT_DISPATCHER


# Backwards compatibility
class CupyChecker:
    """Helper class to check CuPy availability."""

    @staticmethod
    def check_availability() -> tuple[bool, str]:
        """Check if CuPy is available with proper CUDA configuration."""
        try:
            import cupy as cp

            _ = cp.zeros(10)
        except ImportError:
            return False, "CuPy not installed. Install with: pip install cupy-cuda12x"
        except Exception as e:  # broad: optional-backend availability probe
            return False, f"CuPy installed but CUDA failed: {e}"
        else:
            return True, "CuPy available with CUDA"


class TritonChecker:
    """Helper class to check Triton availability."""

    @staticmethod
    def check_availability() -> tuple[bool, str]:
        """Check if Triton is available for custom kernels."""
        if HAS_TRITON:
            return True, "Triton available"
        return False, "Triton not installed. Install with: pip install triton"
