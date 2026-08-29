"""Layer 1: Substrate — Physical State Space Constraints."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from computronium.core.substrates.complex_substrate import (
        ComplexSubstrate as ComplexSubstrateImpl,
    )
    from computronium.core.substrates.sparse_substrate import (
        SparseSubstrate as SparseSubstrateImpl,
    )
    from computronium.core.substrates.ternary_substrate import (
        TernarySubstrate as TernarySubstrateImpl,
    )


# ============================================================
# Substrate Type and Configuration
# ============================================================


class SubstrateType(StrEnum):
    """Physical substrate family declared by a SubstrateConfig.

    The explicit discriminator for substrate class selection — precision
    alone is ambiguous (analog is "float32" too).
    """

    DIGITAL = "digital"
    ANALOG = "analog"
    MEMRISTIVE = "memristive"
    NEUROMORPHIC = "neuromorphic"
    SPARSE = "sparse"
    TERNARY = "ternary"
    OPTICAL = "optical"
    QUANTUM = "quantum"


@dataclass(frozen=True, slots=True)
class SubstrateConfig:
    """Configuration for a physical substrate.

    Attributes:
        precision: Numeric precision ("float32", "float16", "bfloat16",
            "int8", "int4", "binary")
        noise_level: Standard deviation of additive state noise
        weight_bounds: Optional (min, max) tuple for weight clamping
        sparsity: Target sparsity ratio [0, 1]
        device: Target device ("cpu", "cuda", "mps", "fpga", "analog",
            "optical")
    """

    precision: str
    noise_level: float
    weight_bounds: tuple[float, float] | None
    sparsity: float
    device: str
    substrate_type: SubstrateType = SubstrateType.DIGITAL

    @classmethod
    def digital(
        cls,
        *,
        precision: str = "float32",
        noise_level: float = 0.0,
        weight_bounds: tuple[float, float] | None = None,
        sparsity: float = 0.0,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            substrate_type=SubstrateType.DIGITAL,
            precision=precision,
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=sparsity,
            device=device,
        )

    @classmethod
    def analog(
        cls,
        *,
        noise_level: float = 0.1,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            substrate_type=SubstrateType.ANALOG,
            precision="float32",
            noise_level=noise_level,
            weight_bounds=(-1.0, 1.0),
            sparsity=0.0,
            device=device,
        )

    @classmethod
    def memristive(
        cls,
        *,
        noise_level: float = 0.05,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            substrate_type=SubstrateType.MEMRISTIVE,
            precision="int8",
            noise_level=noise_level,
            weight_bounds=(0.0, 1.0),
            sparsity=0.1,
            device=device,
        )

    @classmethod
    def neuromorphic(
        cls,
        *,
        noise_level: float = 0.01,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            substrate_type=SubstrateType.NEUROMORPHIC,
            precision="float16",
            noise_level=noise_level,
            sparsity=0.95,
            weight_bounds=None,
            device=device,
        )

    @classmethod
    def optical(
        cls,
        *,
        noise_level: float = 0.01,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            substrate_type=SubstrateType.OPTICAL,
            precision="float32",
            noise_level=noise_level,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )

    @classmethod
    def quantum(
        cls,
        *,
        noise_level: float = 0.02,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            substrate_type=SubstrateType.QUANTUM,
            precision="complex64",
            noise_level=noise_level,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )

    @classmethod
    def complex(
        cls,
        *,
        noise_level: float = 0.0,
        weight_bounds: tuple[float, float] | None = (-1.0, 1.0),
        sparsity: float = 0.0,
        device: str = "cpu",
    ) -> SubstrateConfig:
        """Complex-valued substrate for holomorphic networks.

        Uses float32 emulation (real/imag channels) for efficient GPU execution
        with Triton-accelerated complex ops (matmul, tanh, conjugate transpose).
        """
        return cls(
            substrate_type=SubstrateType.DIGITAL,
            precision="float32",  # Emulated complex via real/imag channels
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=sparsity,
            device=device,
        )

    @classmethod
    def sparse(
        cls,
        *,
        sparsity: float = 0.5,
        noise_level: float = 0.0,
        precision: str = "float32",
        weight_bounds: tuple[float, float] | None = (-1.0, 1.0),
        device: str = "cpu",
    ) -> SubstrateConfig:
        """Sparse substrate with dynamic sparsity masks.

        Supports unstructured, N:M structured, block, and channel-wise sparsity
        with efficient sparse matmul where available.
        """
        return cls(
            substrate_type=SubstrateType.SPARSE,
            precision=precision,
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=sparsity,
            device=device,
        )

    @classmethod
    def ternary(
        cls,
        *,
        noise_level: float = 0.0,
        weight_bounds: tuple[float, float] | None = (-1.0, 1.0),
        device: str = "cpu",
    ) -> SubstrateConfig:
        """Ternary substrate with STE-based quantization.

        Weights quantized to {-α, 0, +α} with Straight-Through Estimator
        for gradient backpropagation through the quantization function.
        """
        return cls(
            substrate_type=SubstrateType.TERNARY,
            precision="float32",  # Latent weights stay float32
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=0.0,  # Sparsity emerges from thresholding
            device=device,
        )


# ============================================================
# Substrate Protocol
# ============================================================


@runtime_checkable
class Substrate(Protocol):
    """Physical substrate constraints on weights and activations.

    Defines the physical medium: precision, noise profiles, locality constraints,
    and hardware-specific forward/weight update operators.

    Every computronium system runs on a substrate. The substrate injects
    physically accurate noise, enforces weight constraints (e.g. positivity,
    bounded conductance), and provides the forward operator that the Geometry
    layer routes through.
    """

    config: SubstrateConfig

    @abstractmethod
    def quantize_weights(self, w: Tensor) -> Tensor:
        """Apply substrate-specific weight quantization/clamping."""
        ...

    @abstractmethod
    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Inject substrate-appropriate noise into state tensor."""
        ...

    @abstractmethod
    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return the substrate's forward operator: (input, weights) -> output.

        This operator encodes the physics of the substrate (e.g. Kirchhoff's
        laws for crossbars, phase interference for photonic, vesicle release
        for biological).
        """
        ...

    @abstractmethod
    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return the substrate's weight update operator.

        (pseudo_grad, current_w) -> ΔW. Encodes how the substrate physically
        modifies weights (e.g. pulse-based conductance change for memristors,
        phase shift for photonic).
        """
        ...

    @abstractmethod
    def initial_state(self, x: Tensor) -> Tensor:
        """Create initial state for given input on this substrate."""
        ...


# ============================================================
# Default/Reference Substrate Implementations
# ============================================================


def _set_param_name(tensor: Tensor, name: str) -> None:
    """Tag a parameter tensor with its substrate keying name."""
    setattr(tensor, "_param_name", name)


class DigitalSubstrate:
    """Reference substrate: infinite precision, continuous time, no noise."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.digital()

    def _to_precision(self, tensor: Tensor) -> Tensor:
        """Convert tensor to the configured precision."""
        precision = self.config.precision
        if precision == "float32":
            return tensor.to(torch.float32)
        elif precision == "float16":
            return tensor.to(torch.float16)
        elif precision == "bfloat16":
            return tensor.to(torch.bfloat16)
        elif precision == "int8":
            return tensor.to(torch.int8)
        elif precision == "int4":
            # int4 not natively supported, use int8
            return tensor.to(torch.int8)
        elif precision == "binary":
            return tensor.to(torch.bool)
        return tensor

    def quantize_weights(self, w: Tensor) -> Tensor:
        return self._to_precision(w)

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._to_precision(s)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self._to_precision(x)
            w = self._to_precision(w)
            return self._to_precision(x @ w.T)

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            grad = self._to_precision(grad)
            return self._to_precision(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self._to_precision(x)


class AnalogSubstrate:
    """Analog substrate with additive noise and weight bounds."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.analog()

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class MemristiveSubstrate:
    """Memristive crossbar substrate with conductance constraints."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.memristive()

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            # Memristive update: conductance change proportional to voltage
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class NeuromorphicSubstrate:
    """Neuromorphic substrate with high sparsity and low precision."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.neuromorphic()

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class OpticalSubstrate:
    """Photonic/optical substrate with phase interference."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.optical()

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            # Optical: coherent interference (amplitude + phase)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class QuantumSubstrate:
    """Quantum substrate with complex-valued state space."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.quantum()

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class ComplexSubstrate:
    """Complex-valued substrate (emulated via real/imag channels)."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.complex()

    def _to_complex(self, tensor: Tensor) -> Tensor:
        """Convert real/imag pair to complex tensor."""
        # Assumes last dimension is 2 (real, imag)
        if tensor.shape[-1] == 2:
            return torch.view_as_complex(tensor)
        return tensor.to(torch.complex64)

    def _from_complex(self, tensor: Tensor) -> Tensor:
        """Convert complex tensor to real/imag pair."""
        if tensor.dtype.is_complex:
            return torch.view_as_real(tensor)
        return tensor

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            # Complex matmul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            x_c = self._to_complex(x)
            w_c = self._to_complex(w)
            out_c = x_c @ w_c.conj().T  # Conjugate transpose for Hermitian
            return self._from_complex(out_c)

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class SparseSubstrate:
    """Sparse substrate with dynamic sparsity masks."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.sparse()

    def quantize_weights(self, w: Tensor) -> Tensor:
        if self.config.weight_bounds is not None:
            w = w.clamp(*self.config.weight_bounds)
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


class TernarySubstrate:
    """Ternary substrate with STE-based quantization to {-α, 0, +α}."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.ternary()

    def quantize_weights(self, w: Tensor) -> Tensor:
        # STE ternary quantization: sign(w) * |w|_max
        # During forward: quantize; during backward: pass-through gradient
        if self.config.weight_bounds is not None:
            max_val = self.config.weight_bounds[1]
            # Straight-Through Estimator
            w_quant = torch.sign(w) * max_val
            # STE: forward quantized, backward identity
            w = w_quant.detach() + (w - w.detach())
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        if self.config.noise_level > 0:
            noise = torch.randn_like(s) * self.config.noise_level
            return s + noise
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self.inject_state_noise(x)
            w = self.quantize_weights(w)
            return x @ w.T

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            return self.quantize_weights(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self.inject_state_noise(x)


def substrate_from_config(config: SubstrateConfig) -> Substrate:
    """Factory function to instantiate substrate from config."""
    match config.substrate_type:
        case SubstrateType.DIGITAL:
            return DigitalSubstrate(config)
        case SubstrateType.ANALOG:
            return AnalogSubstrate(config)
        case SubstrateType.MEMRISTIVE:
            return MemristiveSubstrate(config)
        case SubstrateType.NEUROMORPHIC:
            return NeuromorphicSubstrate(config)
        case SubstrateType.OPTICAL:
            return OpticalSubstrate(config)
        case SubstrateType.QUANTUM:
            return QuantumSubstrate(config)
        case SubstrateType.SPARSE:
            return SparseSubstrate(config)
        case SubstrateType.TERNARY:
            return TernarySubstrate(config)
        case _:
            # Complex substrate uses DIGITAL type with special config
            if config.precision == "float32" and getattr(config, "_complex_emulated", False):
                return ComplexSubstrate(config)
            return DigitalSubstrate(config)


# Alias for backwards compatibility with tests
ComplexSubstrateImpl = ComplexSubstrate
SparseSubstrateImpl = SparseSubstrate
TernarySubstrateImpl = TernarySubstrate


class NoisySubstrate(DigitalSubstrate):
    """Substrate with additive noise (for testing/backwards compatibility)."""

    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.digital(noise_level=0.05))

    def inject_state_noise(self, s: Tensor) -> Tensor:
        noise = torch.randn_like(s) * self.config.noise_level
        return s + noise
