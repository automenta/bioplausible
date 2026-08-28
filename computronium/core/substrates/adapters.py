"""Cross-Substrate Emulation Adapters.

Enables efficient cross-ontology compositions where native substrate support
is unavailable on target hardware. Each adapter wraps a source substrate
and emulates the target substrate's behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.substrates.complex_substrate import ComplexSubstrate
from computronium.core.substrates.sparse_substrate import SparseSubstrate
from computronium.core.substrates.ternary_substrate import TernarySubstrate
from computronium.ontology import (
    AnalogSubstrate,
    DigitalSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    OpticalSubstrate,
    QuantumSubstrate,
    Substrate,
    SubstrateConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for magic values
_MIN_WEIGHT_NDIM = 2
_MATRIX_NDIM = 2
_QUANTUM_PHASE_LEVELS = 255


# ============================================================
# Adapter Configuration Dataclasses
# ============================================================


@dataclass(frozen=True, slots=True)
class TernaryAdapterConfig:
    """Configuration for DigitalToTernaryAdapter."""

    threshold: float = 0.05
    alpha_init: float = 1.0
    ternary_type: str = "standard"
    learn_threshold: bool = False


@dataclass(frozen=True, slots=True)
class SparseAdapterConfig:
    """Configuration for DigitalToSparseAdapter."""

    sparsity_type: str = "unstructured"
    n_m_ratio: tuple[int, int] = (2, 4)
    block_size: tuple[int, int] = (8, 8)
    update_frequency: int = 100
    prune_criterion: str = "magnitude"
    regrow_criterion: str = "gradient"


@dataclass(frozen=True, slots=True)
class NeuromorphicAdapterConfig:
    """Configuration for DigitalToNeuromorphicAdapter."""

    time_steps: int = 100
    firing_rate_scale: float = 100.0
    surrogate_type: str = "fast_sigmoid"
    beta: float = 10.0


@dataclass(frozen=True, slots=True)
class QuantumAdapterConfig:
    """Configuration for DigitalToQuantumAdapter."""

    n_qubits: int = 8
    n_layers: int = 3
    encoding: str = "amplitude"
    noise_model: str = "depolarizing"


# ============================================================
# Base Adapter Class
# ============================================================


class SubstrateAdapter(DigitalSubstrate):
    """Base class for cross-substrate emulation adapters.

    Wraps a source substrate and emulates a target substrate's behavior
    by translating operations through the source substrate's operators.
    """

    def __init__(
        self,
        source_substrate: Substrate,
        target_config: SubstrateConfig | None = None,
    ):
        # Use target config for interface, but delegate to source
        super().__init__(target_config or source_substrate.config)
        self._source = source_substrate
        self._target_config = target_config or source_substrate.config

    @property
    def source_substrate(self) -> Substrate:
        return self._source

    @property
    def target_config(self) -> SubstrateConfig:
        return self._target_config

    # Delegate to source for operations not overridden
    def quantize_weights(self, w: Tensor) -> Tensor:
        return self._source.quantize_weights(w)

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._source.inject_state_noise(s)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._source.get_forward_operator()

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._source.get_weight_update_operator()

    def initial_state(self, x: Tensor) -> Tensor:
        return self._source.initial_state(x)


# ============================================================
# Digital -> Complex Adapter
# ============================================================


class DigitalToComplexAdapter(SubstrateAdapter):
    """Emulate complex arithmetic on digital hardware.

    Maps float32 real-valued weights to complex64 via real/imag channel
    emulation. Enables holomorphic networks on standard GPUs.

    Source: DigitalSubstrate (float32)
    Target: ComplexSubstrate (complex64 emulated as float32 x 2)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.complex()
        super().__init__(source, target_config)
        self._complex_substrate = ComplexSubstrate(target_config)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return complex forward operator using source's real matmul."""

        def complex_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: [..., in_features] complex (emulated as [..., 2*in_features])
            # w: [out_features, in_features] complex (emulated [out, 2*in])
            return self._complex_substrate.complex_linear(x, w)

        return complex_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Complex weight update using gradient descent on real/imag parts."""

        def complex_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            step_size = getattr(self._target_config, "step_size", 0.01)
            # SGD on emulated real/imag channels
            return current_w - step_size * pseudo_grad

        return complex_update

    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Add complex Gaussian noise to emulated state."""
        return self._complex_substrate.inject_state_noise(s)

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Apply complex weight bounds if configured."""
        if self._target_config.weight_bounds is not None:
            min_val, max_val = self._target_config.weight_bounds
            # Apply bounds to both real and imag parts
            w_real = w[..., ::2].clamp(min_val, max_val)
            w_imag = w[..., 1::2].clamp(min_val, max_val)
            result = torch.empty_like(w)
            result[..., ::2] = w_real
            result[..., 1::2] = w_imag
            return result
        return w


# ============================================================
# Complex -> Optical Adapter (MZI Phase Mapping)
# ============================================================


class ComplexToOpticalAdapter(SubstrateAdapter):
    """Map complex weights to MZI mesh phases for photonic execution.

    Translates complex-valued weights (real/imag) to phase shifts
    in a Mach-Zehnder Interferometer (MZI) mesh.

    Source: ComplexSubstrate (complex64 emulated)
    Target: OpticalSubstrate (phase/amplitude encoding)
    """

    def __init__(
        self,
        source_substrate: ComplexSubstrate | None = None,
        config: SubstrateConfig | None = None,
    ):
        source = source_substrate or ComplexSubstrate()
        target_config = config or SubstrateConfig.optical()
        super().__init__(source, target_config)
        self._optical_substrate = OpticalSubstrate(target_config)
        self._wavelength = 1550e-9
        self._phase_shifter_range = 2 * math.pi

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Photonic forward: complex weights -> MZI phases -> interference."""

        def optical_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: complex input (emulated [..., 2*N])
            # w: complex weights (emulated [out, 2*in])
            # Convert complex weights to MZI phases
            w_complex = self._source.to_complex(w)
            # Phase = arg(w), Amplitude = |w|
            phases = torch.angle(w_complex)  # [-pi, pi]
            amplitudes = torch.abs(w_complex)

            # MZI mesh transfer: cos(phi/2) for bar, i*sin(phi/2) for cross
            # For real-valued I/O, we use the real part of coherent sum
            cos_half = torch.cos(phases / 2)

            # Effective real weight matrix from MZI mesh
            # Simplified: each MZI contributes cos(phi) * amplitude
            effective_w = amplitudes * cos_half

            # Apply to input (take real part of complex input)
            x_real = x[..., ::2] if x.shape[-1] % 2 == 0 else x
            return self._source.get_forward_operator()(x_real, effective_w)

        return optical_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Phase shifter update: map gradient to phase shifts."""

        def optical_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # Current weights are complex (emulated)
            w_complex = self._source.to_complex(current_w)
            current_phases = torch.angle(w_complex)

            # Gradient is also complex (emulated)
            grad_complex = self._source.to_complex(pseudo_grad)
            # Phase gradient: dL/dphi = Re(dL/dw * conj(w)) / |w|
            phase_grad = (grad_complex * w_complex.conj()).real / (
                w_complex.abs() + 1e-8
            )

            # Update phases with step size
            step_size = getattr(self._target_config, "step_size", 0.01)
            new_phases = (current_phases - step_size * phase_grad) % (
                2 * math.pi
            ) - math.pi

            # Reconstruct complex weights from phases + amplitudes
            amplitudes = w_complex.abs()
            new_w_complex = amplitudes * torch.exp(1j * new_phases)
            return self._source.to_real(new_w_complex)

        return optical_update

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Quantize phases to MZI phase shifter resolution (e.g., 8-bit)."""
        w_complex = self._source.to_complex(w)
        phases = torch.angle(w_complex)
        # Quantize to 256 levels (8-bit phase shifters)
        phase_quantized = (
            torch.round(phases / (2 * math.pi) * _QUANTUM_PHASE_LEVELS)
            / _QUANTUM_PHASE_LEVELS
            * (2 * math.pi)
        )
        w_quantized = w_complex.abs() * torch.exp(1j * phase_quantized)
        return self._source.to_real(w_quantized)


# ============================================================
# Digital -> Ternary Adapter (Post-Training Quantization)
# ============================================================


class DigitalToTernaryAdapter(SubstrateAdapter):
    """Post-training ternary quantization with STE for gradient estimation.

    Takes a trained digital (float32) model and quantizes weights to
    ternary {-a, 0, +a} with Straight-Through Estimator for fine-tuning.

    Source: DigitalSubstrate (float32)
    Target: TernarySubstrate ({-1, 0, +1} with STE)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
        adapter_config: TernaryAdapterConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.ternary()
        super().__init__(source, target_config)
        adapter_config = adapter_config or TernaryAdapterConfig()
        self._ternary_substrate = TernarySubstrate(
            target_config,
            ternary_type=adapter_config.ternary_type,
            threshold_init=adapter_config.threshold,
            learn_threshold=adapter_config.learn_threshold,
            alpha_init=adapter_config.alpha_init,
        )
        self._threshold = adapter_config.threshold
        self._calibrated = False

    def calibrate(self, model_params: dict[str, Tensor]) -> None:
        """Calibrate ternary quantization parameters from a trained model.

        Computes optimal a scales and thresholds from weight statistics.
        """
        for name, param in model_params.items():
            if param.ndim >= _MIN_WEIGHT_NDIM and "weight" in name:
                w = param.detach()
                # Compute scale as mean absolute weight
                alpha = w.abs().mean()
                # Set threshold as fraction of max weight
                threshold = w.abs().max() * 0.05

                # Pre-set quantization parameters
                if name not in self._ternary_substrate._alpha:
                    self._ternary_substrate._alpha[name] = torch.nn.Parameter(
                        alpha.to(param.device)
                    )
                else:
                    self._ternary_substrate._alpha[name].data.copy_(alpha)

                if (
                    self._ternary_substrate.ternary_type == "delta"
                    and name not in self._ternary_substrate._alpha_neg
                ):
                    self._ternary_substrate._alpha_neg[name] = torch.nn.Parameter(
                        alpha.to(param.device)
                    )

                if (
                    self._ternary_substrate.learn_threshold
                    and name not in self._ternary_substrate._threshold
                ):
                    self._ternary_substrate._threshold[name] = torch.nn.Parameter(
                        torch.tensor(threshold, device=param.device)
                    )

                self._ternary_substrate._latent_weights[name] = w.clone()

        self._calibrated = True

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Quantize to ternary using calibrated parameters."""
        return self._ternary_substrate.quantize_weights(w)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._ternary_substrate.get_forward_operator()

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._ternary_substrate.get_weight_update_operator()

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._ternary_substrate.inject_state_noise(s)


# ============================================================
# Digital -> Sparse Adapter (Dynamic Sparsity)
# ============================================================


class DigitalToSparseAdapter(SubstrateAdapter):
    """Dynamic sparsity emulation on dense digital hardware.

    Implements dynamic sparse training (rigorous lottery ticket, SNIP, GraSP)
    with efficient sparse matmul where available.

    Source: DigitalSubstrate (dense float32)
    Target: SparseSubstrate (CSR/COO with dynamic masks)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
        adapter_config: SparseAdapterConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.sparse()
        super().__init__(source, target_config)
        adapter_config = adapter_config or SparseAdapterConfig()
        self._sparse_substrate = SparseSubstrate(
            target_config,
            sparsity_type=adapter_config.sparsity_type,
            n_m_ratio=adapter_config.n_m_ratio,
            block_size=adapter_config.block_size,
            update_mask_frequency=adapter_config.update_frequency,
            prune_criterion=adapter_config.prune_criterion,
            regrow_criterion=adapter_config.regrow_criterion,
        )

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Apply sparsity mask (quantization in sparse context)."""
        return self._sparse_substrate.quantize_weights(w)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._sparse_substrate.get_forward_operator()

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._sparse_substrate.get_weight_update_operator()

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._sparse_substrate.inject_state_noise(s)

    def get_mask(self, name: str) -> Tensor | None:
        """Get current sparsity mask for inspection."""
        return self._sparse_substrate.get_mask(name)

    def sparsity_stats(self) -> dict[str, float]:
        """Return sparsity statistics."""
        return self._sparse_substrate.sparsity_stats()


# ============================================================
# Digital -> Neuromorphic Adapter (Rate-to-Spike Encoding)
# ============================================================


class DigitalToNeuromorphicAdapter(SubstrateAdapter):
    """Rate-to-spike encoding with surrogate gradients for neuromorphic emulation.

    Converts rate-coded activations to spike trains using Poisson encoding
    and provides surrogate gradients for backpropagation through spikes.

    Source: DigitalSubstrate (float32 rates)
    Target: NeuromorphicSubstrate (spike trains, AER)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
        adapter_config: NeuromorphicAdapterConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.neuromorphic()
        super().__init__(source, target_config)
        adapter_config = adapter_config or NeuromorphicAdapterConfig()
        self._neuromorphic_substrate = NeuromorphicSubstrate(target_config)
        self._time_steps = adapter_config.time_steps
        self._firing_rate_scale = adapter_config.firing_rate_scale
        self._surrogate_type = adapter_config.surrogate_type
        self._beta = adapter_config.beta

    def _rate_to_spikes(self, rates: Tensor) -> Tensor:
        """Convert firing rates to Poisson spike trains.

        Args:
            rates: [..., n_neurons] firing rates in [0, 1] or [0, max_rate]

        Returns:
            [..., time_steps, n_neurons] binary spike trains
        """
        # Normalize rates to [0, 1]
        rates_norm = torch.clamp(rates / self._firing_rate_scale, 0, 1)
        # Poisson spike generation
        spike_probs = rates_norm.unsqueeze(-2)  # [..., 1, n_neurons]
        spikes = torch.bernoulli(spike_probs.expand(-1, self._time_steps, -1))
        return spikes

    def _spikes_to_rates(self, spikes: Tensor) -> Tensor:
        """Convert spike trains back to firing rates."""
        return spikes.float().mean(dim=-2) * self._firing_rate_scale

    def _surrogate_gradient(self, v: Tensor) -> Tensor:
        """Surrogate gradient for spiking non-linearity."""
        if self._surrogate_type == "fast_sigmoid":
            return self._beta / (1 + self._beta * v.abs()) ** 2
        if self._surrogate_type == "piecewise":
            return (v.abs() < 1.0 / self._beta).float() * self._beta
        if self._surrogate_type == "gaussian":
            return torch.exp(-0.5 * (self._beta * v) ** 2) * self._beta
        return torch.ones_like(v)  # Straight-through

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Neuromorphic forward: rate-coded input -> spikes -> synaptic current."""
        source_op = self._source.get_forward_operator()

        def neuromorphic_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: rate-coded input [..., n_inputs]
            # w: synaptic weights [n_outputs, n_inputs]

            # Encode input rates as spike trains
            x_spikes = self._rate_to_spikes(x)  # [..., T, n_inputs]

            # Synaptic current: convolution of spikes with weights
            # Simplified: average rate over time window
            x_rates = x_spikes.mean(dim=-2)  # [..., n_inputs]

            # Standard matmul for synaptic integration
            out_rates = source_op(x_rates, w)  # [..., n_outputs]

            # Convert output rates to spikes
            out_spikes = self._rate_to_spikes(out_rates)

            return out_spikes

        return neuromorphic_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """STDP-based update with surrogate gradients."""

        def neuromorphic_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # pseudo_grad is in spike domain (surrogate gradient)
            # Apply surrogate gradient to get weight update
            surrogate = self._surrogate_gradient(current_w)
            step_size = getattr(self._target_config, "step_size", 0.01)
            return current_w - step_size * pseudo_grad * surrogate

        return neuromorphic_update

    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Add spike dropout and thermal noise."""
        return self._neuromorphic_substrate.inject_state_noise(s)


# ============================================================
# Digital -> Quantum Adapter (Variational Circuit Emulation)
# ============================================================


class DigitalToQuantumAdapter(SubstrateAdapter):
    """Classical emulation of variational quantum circuits.

    Maps digital weights to parameterized quantum circuit angles
    and simulates the circuit classically for GPU execution.

    Source: DigitalSubstrate (float32)
    Target: QuantumSubstrate (amplitude encoding, parameterized unitaries)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
        adapter_config: QuantumAdapterConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.quantum()
        super().__init__(source, target_config)
        adapter_config = adapter_config or QuantumAdapterConfig()
        self._quantum_substrate = QuantumSubstrate(target_config)
        self._n_qubits = adapter_config.n_qubits
        self._n_layers = adapter_config.n_layers
        self._encoding = adapter_config.encoding
        self._noise_model = adapter_config.noise_model

    def _amplitude_encode(self, x: Tensor) -> Tensor:
        """Amplitude encoding: normalize and embed in quantum state."""
        # x: [..., n_features] -> normalize
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        # Pad/truncate to 2^n_qubits
        target_dim = 2**self._n_qubits
        if x_norm.shape[-1] < target_dim:
            pad = target_dim - x_norm.shape[-1]
            x_norm = torch.nn.functional.pad(x_norm, (0, pad))
        elif x_norm.shape[-1] > target_dim:
            x_norm = x_norm[..., :target_dim]
        return x_norm

    def _variational_circuit(self, state: Tensor, params: Tensor) -> Tensor:
        """Classical simulation of parameterized quantum circuit.

        Args:
            state: [..., 2^n_qubits] quantum state amplitudes
            params: [n_layers * n_qubits * 3] circuit parameters (RY, RZ, CNOT)

        Returns:
            [..., 2^n_qubits] output state
        """
        # Simplified classical simulation
        # For efficiency, we use a differentiable approximation
        # Real implementation would use state vector simulation

        batch_shape = state.shape[:-1]
        state_flat = state.view(-1, 2**self._n_qubits)

        # Apply variational layers
        param_idx = 0
        for _layer in range(self._n_layers):
            # Single-qubit rotations (RY)
            for _q in range(self._n_qubits):
                if param_idx < params.shape[-1]:
                    theta = params[..., param_idx]
                    param_idx += 1
                    # RY(theta) = cos(theta/2) I - i sin(theta/2) Y
                    # Simplified: rotate amplitudes
                    state_flat = self._apply_ry(state_flat, theta)

            # Single-qubit rotations (RZ)
            for _q in range(self._n_qubits):
                if param_idx < params.shape[-1]:
                    theta = params[..., param_idx]
                    param_idx += 1
                    state_flat = self._apply_rz(state_flat, theta)

            # Entanglement layer (CNOT chain)
            for q in range(self._n_qubits - 1):
                state_flat = self._apply_cnot(state_flat, q, q + 1)

        # Measurement: Z expectation values (probability of |1>)
        probs = state_flat.abs().pow(2)
        return probs.view(*batch_shape, -1)

    @staticmethod
    def _apply_ry(state: Tensor, theta: Tensor) -> Tensor:
        """Apply RY rotation to state vector (simplified)."""
        # This is a placeholder - real implementation would manipulate
        # the state vector properly. For now, use differentiable approximation.
        n = state.shape[-1]
        cos_half = torch.cos(theta / 2).unsqueeze(-1)
        sin_half = torch.sin(theta / 2).unsqueeze(-1)
        # Simplified: just scale by cos/sin factors
        return state * cos_half + state.roll(shifts=n // 2, dims=-1) * sin_half

    @staticmethod
    def _apply_rz(state: Tensor, theta: Tensor) -> Tensor:
        """Apply RZ rotation (phase shift)."""
        return state * torch.exp(1j * theta.unsqueeze(-1) / 2)

    @staticmethod
    def _apply_cnot(state: Tensor, control: int, target: int) -> Tensor:
        """Apply CNOT gate (simplified classical simulation)."""
        # Placeholder - real implementation would swap amplitudes
        _ = control, target
        return state

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Quantum circuit forward: encode -> variational circuit -> measure."""
        source_op = self._source.get_forward_operator()

        def quantum_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: classical input [..., n_features]
            # w: circuit parameters [n_params]

            # Amplitude encode input
            state = self._amplitude_encode(x)

            # If w is 2D, treat as weight matrix; if 1D, treat as circuit params
            if w.ndim == _MATRIX_NDIM:
                # Classical fallback: x @ w.T
                return source_op(x, w)
            # Quantum circuit
            return self._variational_circuit(state, w)

        return quantum_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Parameter-shift rule for quantum gradients."""

        def quantum_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            step_size = getattr(self._target_config, "step_size", 0.01)
            if current_w.ndim == _MATRIX_NDIM:
                # Classical weights
                return current_w - step_size * pseudo_grad

            # Parameter-shift: grad f(theta) = [f(theta+pi/2) - f(theta-pi/2)] / 2
            # Simplified: use pseudo_grad as gradient estimate
            return current_w - step_size * pseudo_grad

        return quantum_update


# ============================================================
# Digital -> Memristive Adapter (Conductance Quantization)
# ============================================================


class DigitalToMemristiveAdapter(SubstrateAdapter):
    """Memristive crossbar emulation: conductance quantization + IR-drop model.

    Maps digital weights to memristor conductance values with physical
    constraints (bounded, positive conductance, IR-drop).

    Source: DigitalSubstrate (float32)
    Target: MemristiveSubstrate (int8 conductance, IR-drop)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.memristive()
        super().__init__(source, target_config)
        self._memristive_substrate = MemristiveSubstrate(target_config)

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Quantize to memristor conductance levels."""
        return self._memristive_substrate.quantize_weights(w)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._memristive_substrate.get_forward_operator()

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._memristive_substrate.get_weight_update_operator()

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._memristive_substrate.inject_state_noise(s)


# ============================================================
# Digital -> Analog Adapter (Noise Injection)
# ============================================================


class DigitalToAnalogAdapter(SubstrateAdapter):
    """Analog compute emulation: continuous values with noise injection.

    Source: DigitalSubstrate (float32)
    Target: AnalogSubstrate (continuous, noisy)
    """

    def __init__(
        self,
        source_substrate: DigitalSubstrate | None = None,
        config: SubstrateConfig | None = None,
    ):
        source = source_substrate or DigitalSubstrate()
        target_config = config or SubstrateConfig.analog()
        super().__init__(source, target_config)
        self._analog_substrate = AnalogSubstrate(target_config)

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._analog_substrate.inject_state_noise(s)

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Apply analog weight bounds."""
        if self._target_config.weight_bounds is not None:
            min_val, max_val = self._target_config.weight_bounds
            return w.clamp(min_val, max_val)
        return w

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._analog_substrate.get_forward_operator()

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._analog_substrate.get_weight_update_operator()


# ============================================================
# Factory Functions
# ============================================================


class AdapterNotFoundError(ValueError):
    """Raised when no adapter is found for a given source/target pair."""

    def __init__(
        self, source_type: str, target_type: str, available: list[tuple[str, str]]
    ) -> None:
        super().__init__(
            f"No adapter for {source_type} -> {target_type}. Available: {available}"
        )
        self.source_type = source_type
        self.target_type = target_type


def create_substrate_adapter(
    source_type: str,
    target_type: str,
    source_substrate: Substrate | None = None,
    config: SubstrateConfig | None = None,
    **kwargs,
) -> SubstrateAdapter:
    """Factory function for cross-substrate adapters.

    Args:
        source_type: Source substrate type ("digital", "complex", etc.)
        target_type: Target substrate type ("complex", "optical", "ternary",
                     "sparse", "neuromorphic", "quantum", "memristive", "analog")
        source_substrate: Optional source substrate instance
        config: Optional target substrate config
        **kwargs: Additional adapter-specific parameters

    Returns:
        Configured SubstrateAdapter instance
    """
    adapter_map = {
        ("digital", "complex"): DigitalToComplexAdapter,
        ("complex", "optical"): ComplexToOpticalAdapter,
        ("digital", "ternary"): DigitalToTernaryAdapter,
        ("digital", "sparse"): DigitalToSparseAdapter,
        ("digital", "neuromorphic"): DigitalToNeuromorphicAdapter,
        ("digital", "quantum"): DigitalToQuantumAdapter,
        ("digital", "memristive"): DigitalToMemristiveAdapter,
        ("digital", "analog"): DigitalToAnalogAdapter,
    }

    key = (source_type, target_type)
    if key not in adapter_map:
        raise AdapterNotFoundError(source_type, target_type, list(adapter_map.keys()))

    adapter_class = adapter_map[key]
    return adapter_class(source_substrate, config, **kwargs)


__all__ = [
    "AdapterNotFoundError",
    "ComplexToOpticalAdapter",
    "DigitalToAnalogAdapter",
    "DigitalToComplexAdapter",
    "DigitalToMemristiveAdapter",
    "DigitalToNeuromorphicAdapter",
    "DigitalToQuantumAdapter",
    "DigitalToSparseAdapter",
    "DigitalToTernaryAdapter",
    "NeuromorphicAdapterConfig",
    "QuantumAdapterConfig",
    "SparseAdapterConfig",
    "SubstrateAdapter",
    "TernaryAdapterConfig",
    "create_substrate_adapter",
]
