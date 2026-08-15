"""Unified Kernel Backend Infrastructure for Bio-Plausible Algorithms.

Provides the KernelBackend protocol, KernelRegistry for auto-selection,
and configuration dataclasses for hardware-agnostic kernel acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

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
    """Global registry for kernel backends with auto-selection logic."""

    _backends: dict[AlgorithmFamily, dict[HardwareTarget, type]] = {}
    _instances: dict[tuple[AlgorithmFamily, HardwareTarget], object] = {}

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
    def get(
        cls, algorithm: AlgorithmFamily, hardware: HardwareTarget
    ) -> object | None:
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
    if "eqprop" in name or "looped" in name or "eqprop" in name:
        return AlgorithmFamily.EQPROP
    if "fa" in name or "feedback" in name or "dfa" in name:
        return AlgorithmFamily.FA
    if "hebbian" in name:
        return AlgorithmFamily.HEBBIAN
    if "forward" in name and "forward_only" in name:
        return AlgorithmFamily.FF
    if "pepita" in name:
        return AlgorithmFamily.PEPITA
    if "target" in name or "tp" in name:
        return AlgorithmFamily.TP
    if "predictive" in name or "pc" in name:
        return AlgorithmFamily.PC
    if "spiking" in name or "snn" in name or "stdp" in name:
        return AlgorithmFamily.SNN
    if "tile" in name or "equitile" in name:
        return AlgorithmFamily.TILE
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
    "LocalityLevel",
    "KernelConfig",
    "KernelBackend",
    "KernelRegistry",
    "infer_algorithm_family",
]
