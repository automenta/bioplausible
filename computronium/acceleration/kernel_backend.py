"""Unified Kernel Backend Infrastructure for Bio-Plausible Algorithms.

Provides the KernelBackend protocol, KernelRegistry for auto-selection,
and configuration dataclasses for hardware-agnostic kernel acceleration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import torch


class AlgorithmFamily(StrEnum):
    """Supported bio-plausible algorithm families."""

    EQPROP = "eqprop"
    FA = "fa"
    HEBBIAN = "hebbian"
    FF = "ff"
    PEPITA = "pepita"
    TP = "tp"
    PC = "pc"
    SNN = "snn"
    TILE = "tile"
    MEP = "mep"
    O1MEMORY = "o1memory"
    BACKPROP = "backprop"


class HardwareTarget(StrEnum):
    """Supported hardware targets."""

    CPU = "cpu"
    CUDA = "cuda"
    TRITON = "triton"
    FPGA = "fpga"
    NEUROMORPHIC = "neuromorphic"
    OPTICAL = "optical"
    CROSSBAR = "crossbar"
    QUANTUM = "quantum"


class LocalityLevel(StrEnum):
    """Credit assignment locality level."""

    GLOBAL = "global"
    LAYERWISE = "layerwise"
    LOCAL = "local"
    EQUILIBRIUM = "equilibrium"
    FORWARD_ONLY = "forward_only"


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Configuration for a kernel backend.

    Args:
        algorithm: Algorithm family this kernel implements.
        hardware: Target hardware backend.
        dtype: Computation dtype (default float32).
        use_autograd: Whether to use autograd (False = O(1) contrastive path).
        settle_steps: Number of settling steps for algorithms with settling dynamics.
        beta: Nudge strength (EqProp, MEP, Contrastive).
        gamma: Decay/leak factor.
        spectral_norm: Whether to apply spectral normalization.
        **kwargs: Algorithm-specific extra parameters.
    """

    algorithm: AlgorithmFamily
    hardware: HardwareTarget
    dtype: torch.dtype = torch.float32
    use_autograd: bool = False
    settle_steps: int = 0
    beta: float = 0.0
    gamma: float = 1.0
    spectral_norm: bool = False
    # Algorithm-specific extras (validated at backend construction)
    # FA: dropout_prob, feedback_mode
    # Hebbian: use_oja, learning_rate
    # FF: threshold, num_layers
    # PEPITA: feedback_matrix_scale
    # TP: target_lr, inverse_net_lr
    # PC: infer_steps, eta_infer
    # SNN: num_steps, spike_grad, tau_mem, tau_syn
    # Tile: neurons_per_tile, tiles_per_layer, num_hidden_layers
    # MEP: ns_steps, rank_frac, fisher_damping, loss_type
    # O1Memory: loss_type, softmax_temperature
    # Backprop: grad_clip, accumulation_steps
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.algorithm == AlgorithmFamily.EQPROP and self.settle_steps == 0:
            object.__setattr__(self, "settle_steps", 30)
        if (
            self.algorithm in (AlgorithmFamily.MEP, AlgorithmFamily.O1MEMORY)
            and self.settle_steps == 0
        ):
            object.__setattr__(self, "settle_steps", 30)


@runtime_checkable
class KernelBackend(Protocol):
    """Hardware-agnostic kernel backend for a bio-plausible algorithm family.

    Implementations must be stateless or reset between training steps.
    All tensor operations should respect the configured dtype and device.
    """

    name: AlgorithmFamily
    supported_dtypes: tuple[torch.dtype, ...]
    supports_autograd: bool
    requires_settle: bool
    memory_complexity: str  # "O(1)", "O(L)", "O(L*H)"
    locality_level: LocalityLevel

    def initialize(self, config: KernelConfig) -> None: ...
    def forward(self, *args: object, **kwargs: object) -> object: ...
    def backward(self, *args: object, **kwargs: object) -> object: ...
    def update_weights(self, *args: object, **kwargs: object) -> None: ...
    def get_memory_stats(self) -> dict[str, float]: ...
    def get_settle_telemetry(self) -> dict[str, object] | None: ...


# Registry
class KernelRegistry:
    """Global registry for kernel backends with auto-selection and auto-tuning logic."""

    _backends: dict[AlgorithmFamily, dict[HardwareTarget, type]] = {}
    _instances: dict[tuple[AlgorithmFamily, HardwareTarget], object] = {}
    # Auto-tuning cache: (algorithm, hardware, op_name, shape) -> best_hardware
    _autotune_cache: dict[
        tuple[AlgorithmFamily, HardwareTarget, str, tuple[int, ...]], HardwareTarget
    ] = {}
    # Benchmark results: (algorithm, hardware, op_name, shape) -> list of (hardware, time_ms)
    _benchmark_cache: dict[
        tuple[AlgorithmFamily, HardwareTarget, str, tuple[int, ...]],
        list[tuple[HardwareTarget, float]],
    ] = {}

    @classmethod
    def register(
        cls,
        algorithm: AlgorithmFamily,
        hardware: HardwareTarget,
        backend_cls: type,
    ) -> None:
        """Register a kernel backend class."""
        if algorithm not in cls._backends:
            cls._backends[algorithm] = {}
        cls._backends[algorithm][hardware] = backend_cls

    @classmethod
    def get(cls, algorithm: AlgorithmFamily, hardware: HardwareTarget) -> object | None:
        """Get or create a backend instance."""
        key = (algorithm, hardware)
        if key in cls._instances:
            return cls._instances[key]

        if algorithm not in cls._backends or hardware not in cls._backends[algorithm]:
            return None

        backend_cls = cls._backends[algorithm][hardware]
        instance = backend_cls()
        cls._instances[key] = instance
        return instance

    @classmethod
    def get_best(
        cls, algorithm: AlgorithmFamily, preferred: HardwareTarget | str = "cuda"
    ) -> object | None:
        """Get best available backend for algorithm, falling back through priority.

        Priority: TRITON > CUDA > CPU
        """
        preferred_hw = (
            HardwareTarget(preferred) if isinstance(preferred, str) else preferred
        )

        # Try preferred first
        backend = cls.get(algorithm, preferred_hw)
        if backend is not None:
            return backend

        # Fallback priority
        for hw in (HardwareTarget.TRITON, HardwareTarget.CUDA, HardwareTarget.CPU):
            backend = cls.get(algorithm, hw)
            if backend is not None:
                return backend

        return None

    @classmethod
    def get_best_for_shape(
        cls,
        algorithm: AlgorithmFamily,
        op_name: str,
        shape: tuple[int, ...],
        benchmark_fn: Callable[[object, tuple[int, ...]], float] | None = None,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> object | None:
        """Get best backend for a specific operation shape using auto-tuning.

        Args:
            algorithm: Algorithm family
            op_name: Operation name (e.g., "forward", "backward", "update_weights")
            shape: Input tensor shape
            benchmark_fn: Optional custom benchmark function. If None, uses default timing.
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations

        Returns:
            Best backend instance for this operation/shape combination
        """
        cache_key = (algorithm, op_name, shape)

        # Check auto-tune cache first
        if cache_key in cls._autotune_cache:
            best_hw = cls._autotune_cache[cache_key]
            return cls.get(algorithm, best_hw)

        # Get available hardware targets for this algorithm
        available_hw = cls.list_for(algorithm)
        if not available_hw:
            return None

        # Priority order for fallback
        priority_order = [
            HardwareTarget.TRITON,
            HardwareTarget.CUDA,
            HardwareTarget.CPU,
        ]
        candidate_hw = [hw for hw in priority_order if hw in available_hw]

        if len(candidate_hw) == 1:
            # Only one option, no need to benchmark
            cls._autotune_cache[cache_key] = candidate_hw[0]
            return cls.get(algorithm, candidate_hw[0])

        # Benchmark each backend for this shape
        best_hw = cls._benchmark_backends(
            algorithm,
            op_name,
            shape,
            candidate_hw,
            benchmark_fn,
            warmup_runs,
            benchmark_runs,
        )

        cls._autotune_cache[cache_key] = best_hw
        return cls.get(algorithm, best_hw)

    @classmethod
    def _benchmark_backends(
        cls,
        algorithm: AlgorithmFamily,
        op_name: str,
        shape: tuple[int, ...],
        candidate_hw: list[HardwareTarget],
        benchmark_fn: Callable[[object, tuple[int, ...]], float] | None,
        warmup_runs: int,
        benchmark_runs: int,
    ) -> HardwareTarget:
        """Benchmark multiple backends and return the fastest."""
        bench_key = (algorithm, op_name, shape)
        results = []

        for hw in candidate_hw:
            backend = cls.get(algorithm, hw)
            if backend is None:
                continue

            try:
                if benchmark_fn is not None:
                    # Use custom benchmark function
                    time_ms = benchmark_fn(backend, shape)
                else:
                    # Default: time the forward pass with dummy inputs
                    time_ms = cls._default_benchmark(
                        backend, op_name, shape, warmup_runs, benchmark_runs
                    )

                if time_ms > 0 and not np.isinf(time_ms):
                    results.append((hw, time_ms))
            except Exception:
                # Backend failed, skip
                continue

        # Cache benchmark results
        if results:
            cls._benchmark_cache[bench_key] = results

        if not results:
            # All failed, return first candidate as fallback
            return candidate_hw[0]

        # Return fastest
        return min(results, key=lambda x: x[1])[0]

    @classmethod
    def _default_benchmark(
        cls,
        backend: object,
        op_name: str,
        shape: tuple[int, ...],
        warmup_runs: int,
        benchmark_runs: int,
    ) -> float:
        """Default benchmark using forward pass with random inputs."""
        import time

        # Create dummy inputs based on shape
        if hasattr(backend, "initialize"):
            # Try to initialize with minimal config
            try:
                from computronium.acceleration.kernel_backend import (
                    HardwareTarget,
                    KernelConfig,
                )

                config = KernelConfig(
                    algorithm=backend.name
                    if hasattr(backend, "name")
                    else AlgorithmFamily.BACKPROP,
                    hardware=HardwareTarget.CPU,
                    extra={"num_layers": 2, "hidden_dim": shape[-1] if shape else 256},
                )
                backend.initialize(config)
            except Exception:
                pass

        # Get the operation method
        method = getattr(backend, op_name, None)
        if method is None:
            return float("inf")

        # Warmup
        try:
            for _ in range(warmup_runs):
                if op_name == "forward":
                    x = torch.randn(*shape, device="cpu")
                    _ = method(x)
                else:
                    # For other ops, try with minimal args
                    _ = method()
        except Exception:
            return float("inf")

        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            try:
                if op_name == "forward":
                    x = torch.randn(*shape, device="cpu")
                    _ = method(x)
                else:
                    _ = method()
            except Exception:
                return float("inf")
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        return float(np.mean(times))

    @classmethod
    def clear_autotune_cache(cls) -> None:
        """Clear the auto-tuning cache (e.g., when hardware changes)."""
        cls._autotune_cache.clear()
        cls._benchmark_cache.clear()

    @classmethod
    def get_benchmark_results(
        cls, algorithm: AlgorithmFamily, op_name: str, shape: tuple[int, ...]
    ) -> list[tuple[HardwareTarget, float]] | None:
        """Get cached benchmark results for an operation/shape."""
        return cls._benchmark_cache.get((algorithm, op_name, shape))

    @classmethod
    def has(cls, algorithm: AlgorithmFamily, hardware: HardwareTarget) -> bool:
        """Check if a backend is registered."""
        return algorithm in cls._backends and hardware in cls._backends[algorithm]

    @classmethod
    def list_for(cls, algorithm: AlgorithmFamily) -> list[HardwareTarget]:
        """List registered hardware targets for an algorithm."""
        return list(cls._backends.get(algorithm, {}).keys())

    @classmethod
    def list_all(cls) -> dict[AlgorithmFamily, list[HardwareTarget]]:
        """List all registered backends."""
        return {alg: list(hw.keys()) for alg, hw in cls._backends.items()}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear instantiated backends (for testing/reload)."""
        cls._instances.clear()


def infer_algorithm_family(model_name: str) -> AlgorithmFamily | None:
    """Infer algorithm family from model registry name."""
    name = model_name.lower()
    if "eqprop" in name or "looped" in name:
        return AlgorithmFamily.EQPROP
    # ``tile_pc``/``tile_snn``/``tile_target_prop`` must resolve to TILE — the
    # tile marker is the most specific prefix, so check it before the generic
    # family substrings those names also carry.
    if "tile" in name or "equitile" in name:
        return AlgorithmFamily.TILE
    # ``fabricpc_graph_pcn`` carries a "pc" substring and should be PC, while
    # ``predictive_coding_hybrid`` carries neither "pc" nor "fa" — check both
    # before the generic FA substring (which the "fabric" prefix would match).
    if "predictive" in name or "pc" in name:
        return AlgorithmFamily.PC
    if "spiking" in name or "snn" in name or "stdp" in name:
        return AlgorithmFamily.SNN
    # Hebbian before FA: ``three_factor_hebbian`` carries "fa" (in "factor")
    # and must not be mis-resolved to the FA family.
    if "hebbian" in name:
        return AlgorithmFamily.HEBBIAN
    if "fa" in name or "feedback" in name or "dfa" in name:
        return AlgorithmFamily.FA
    if "forward_only" in name or "forward_forward" in name or name == "ff":
        return AlgorithmFamily.FF
    if "pepita" in name:
        return AlgorithmFamily.PEPITA
    if "target" in name or "tp" in name:
        return AlgorithmFamily.TP
    if (
        "mep" in name
        or "o1memory" in name
        or "muon" in name
        or "dion" in name
        or "fisher" in name
    ):
        return AlgorithmFamily.MEP
    if "backprop" in name:
        return AlgorithmFamily.BACKPROP
    return None


__all__ = [
    "AlgorithmFamily",
    "HardwareTarget",
    "KernelBackend",
    "KernelConfig",
    "KernelRegistry",
    "LocalityLevel",
    "infer_algorithm_family",
]
