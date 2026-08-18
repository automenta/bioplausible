"""Hardware-faithful variants of :class:`LoopedMLP` (plan §17, §2, REFACTOR7 §4).

These facades approximate non-Von-Neumann substrates so that the frontier
pipeline's ``cost_of_plausibility`` can be measured *hardware-aware* rather
than on an idealized digital GPU:

- :class:`QuantizedLoopedMLP` — FPGA / bit-precision: every hidden-layer
  activation is rounded to a signed ``bits``-bit representation each settle
  step.
- :class:`NoisyLoopedMLP` — analog/photonic: a Gaussian noise draw (shot +
  thermal) is added to every hidden-layer pre-activation each settle step.
- :class:`SpikingLoopedMLP` — neuromorphic: LIF-style spike-and-reset on the
  hidden activations each settle step.
- :class:`OpticalLoopedMLP` — photonic/optical: phase- and detector-noise on
  the hidden activations each settle step.
- :class:`CrossbarLoopedMLP` — analog crossbar: ADC-quantised conductance
  readout with IR-drop attenuation on the hidden activations each step.
- :class:`QuantumLoopedMLP` — variational/quantum: measurement (shot) noise
  on the hidden activations each settle step.

All keep float32 gradients (a surrogate for analog accumulation / high-
precision integration) and otherwise share the consolidated layered eqprop
engine of their parent (:class:`EquilibriumMLP`), so they drop into
``CoreTrainer`` and the probe driver unchanged. Living here (instead of
``validation/tracks``) keeps them importable from ``core.trainer`` without a
``validation -> core`` dependency.
"""

import torch

from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model

from .looped_mlp import LoopedMLP

__all__ = [
    "CrossbarLoopedMLP",
    "NoisyLoopedMLP",
    "OpticalLoopedMLP",
    "QuantizedLoopedMLP",
    "QuantumLoopedMLP",
    "SpikingLoopedMLP",
]


@register_model(
    "quantized_looped_mlp",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=[
        "eqprop",
        "looped_mlp",
        "fpga",
        "hardware",
        "quantization",
        status_tag("experimental"),
    ],
    extra={"parity_threshold": 0.05},
)
class QuantizedLoopedMLP(LoopedMLP):
    """Approximate an FPGA substrate with fixed bit-precision hidden states.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        bits: Signed fixed-point width (e.g. ``8`` for INT8). Each hidden
            activation is rounded to ``[-2**(bits-1)+1, 2**(bits-1)-1]`` every
            settle step.
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments
            (``num_layers``, ``max_steps``, ``use_spectral_norm``, ...).

    Gradients stay float: this simulates high-precision accumulators with a
    quantized interaction (surrogate-gradient style).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, bits: int = 8, **kwargs
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.bits = int(bits)
        self.scale = 2 ** (self.bits - 1) - 1  # e.g. signed INT8 range [-127, 127]

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * self.scale).clamp(-self.scale, self.scale) / self.scale

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Settle one step with quantised hidden states (FPGA fidelity)."""
        # Quantise the prior hidden activations before they feed the bottom-up,
        # self-recurrent and top-down terms — this matches the original
        # single-hidden facade's "quantise h before W Recurrence" rule.
        quantised = [activations[0]]
        for i in range(1, len(activations)):
            quantised.append(self._quantize(activations[i]))
        return super().forward_dynamics(quantised, beta=beta, target=target)


@register_model(
    "noisy_looped_mlp",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=[
        "eqprop",
        "looped_mlp",
        "analog",
        "photonic",
        "hardware",
        "noise",
        status_tag("experimental"),
    ],
    extra={"parity_threshold": 0.05},
)
class NoisyLoopedMLP(LoopedMLP):
    """Approximate an analog/photonic substrate with continuous noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        noise_level: Std-dev of the per-step Gaussian noise (as a fraction of
            the activation scale).
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments.

    Attractor dynamics correct for the injected noise continuously, so the
    equilibrium is reached despite a noisy interaction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        noise_level: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.noise_level = float(noise_level)

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Inject Gaussian noise into the bottom-up term of each hidden layer."""
        new_acts = super().forward_dynamics(activations, beta=beta, target=target)
        # Add fresh noise to every hidden activation (layers 1..L), analogue
        # of the original single-hidden facade's "noise to the recurrent
        # pre-activation" rule. The output layer (last entry) stays noise-free.
        with torch.no_grad():
            for i in range(1, len(new_acts) - 1):
                new_acts[i] = (
                    new_acts[i] + torch.randn_like(new_acts[i]) * self.noise_level
                )
        return new_acts


@register_model(
    "spiking_looped_mlp",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=[
        "eqprop",
        "looped_mlp",
        "neuromorphic",
        "spiking",
        "hardware",
        status_tag("experimental"),
    ],
    extra={"parity_threshold": 0.05},
)
class SpikingLoopedMLP(LoopedMLP):
    """Approximate a neuromorphic substrate with LIF-style spike-and-reset.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        spike_threshold: Membrane voltage at which a neuron emits a spike and
            is reset to zero.
        refractory_period: Number of settle steps a neuron stays clamped at
            zero after spiking.
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments.

    Each settle step, hidden activations above ``spike_threshold`` are reset
    to zero and held there for ``refractory_period`` steps (surrogate
    gradients: the reset is applied on detached values so learning still
    flows through the non-spiking pathway).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        spike_threshold: float = 1.0,
        refractory_period: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.spike_threshold = float(spike_threshold)
        self.refractory_period = float(refractory_period)
        self._refractory_counts: list[torch.Tensor] | None = None

    def _reset_refractory(self, activations) -> None:
        """(Re)allocate refractory counters sized to the current activations."""
        if activations is None or len(activations) < 2:
            return
        hidden = activations[1].shape[-1]
        batch = activations[1].shape[0]
        device, dtype = activations[1].device, activations[1].dtype
        self._refractory_counts = [
            torch.zeros(batch, hidden, device=device, dtype=dtype)
            for _ in range(len(activations) - 1)
        ]

    def _refractory_for(self, activations) -> list[torch.Tensor]:
        """Return (and lazily (re)allocate) batch-shaped refractory counters."""
        expected = len(activations) - 1
        if self._refractory_counts is None or len(self._refractory_counts) != expected:
            self._reset_refractory(activations)
            return self._refractory_counts or []
        counts = self._refractory_counts
        if counts[0].shape != activations[1].shape:
            self._reset_refractory(activations)
            return self._refractory_counts or []
        return counts

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Settle one step with neuromorphic spike-and-reset on hidden states."""
        counts_list = self._refractory_for(activations)

        new_acts = super().forward_dynamics(activations, beta=beta, target=target)
        with torch.no_grad():
            for i in range(1, len(new_acts) - 1):
                counts = counts_list[i - 1]
                v = new_acts[i]
                spiked = (v > self.spike_threshold).to(v.dtype)
                # Reset neurons that spiked this step or are still refractory.
                reset = (spiked > 0) | (counts > 0)
                new_acts[i] = torch.where(reset, torch.zeros_like(v), v)
                # Refractory: spiking neurons enter refractory_period; others
                # decay one count toward zero.
                counts.copy_(
                    torch.where(spiked > 0, self.refractory_period, counts - 1)
                )
                counts.clamp_min_(0)
        return new_acts


@register_model(
    "optical_looped_mlp",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=[
        "eqprop",
        "looped_mlp",
        "optical",
        "photonic",
        "hardware",
        status_tag("experimental"),
    ],
    extra={"parity_threshold": 0.05},
)
class OpticalLoopedMLP(LoopedMLP):
    """Approximate an optical substrate with phase- and detector-noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        phase_noise: Std-dev of the phase-noise applied to hidden amplitudes.
        detector_noise: Std-dev of the additive detector/readout noise.
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        phase_noise: float = 0.01,
        detector_noise: float = 0.005,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.phase_noise = float(phase_noise)
        self.detector_noise = float(detector_noise)

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Add phase + detector noise to the hidden amplitudes each step."""
        new_acts = super().forward_dynamics(activations, beta=beta, target=target)
        with torch.no_grad():
            for i in range(1, len(new_acts) - 1):
                a = new_acts[i]
                phase = a * torch.randn_like(a) * self.phase_noise
                detection = torch.randn_like(a) * self.detector_noise
                new_acts[i] = a + phase + detection
        return new_acts


@register_model(
    "crossbar_looped_mlp",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=[
        "eqprop",
        "looped_mlp",
        "crossbar",
        "memristor",
        "analog",
        "hardware",
        status_tag("experimental"),
    ],
    extra={"parity_threshold": 0.05},
)
class CrossbarLoopedMLP(LoopedMLP):
    """Approximate an analog crossbar with ADC-quantised conductance readout.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        adc_bits: ADC resolution for the conductance readout.
        ir_drop_factor: Fraction of signal lost to IR drop along the array.
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        adc_bits: int = 8,
        ir_drop_factor: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.adc_bits = int(adc_bits)
        self.ir_drop_factor = float(ir_drop_factor)
        self._adc_levels = 2 ** (self.adc_bits - 1) - 1

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Quantise hidden states to ADC levels and apply IR-drop attenuation."""
        new_acts = super().forward_dynamics(activations, beta=beta, target=target)
        with torch.no_grad():
            for i in range(1, len(new_acts) - 1):
                a = new_acts[i]
                scale = a.abs().max().clamp_min(1e-8)
                q = (
                    torch.round(a / scale * self._adc_levels).clamp(
                        -self._adc_levels, self._adc_levels
                    )
                    / self._adc_levels
                    * scale
                )
                new_acts[i] = q * (1 - self.ir_drop_factor)
        return new_acts


@register_model(
    "quantum_looped_mlp",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=[
        "eqprop",
        "looped_mlp",
        "quantum",
        "hardware",
        status_tag("experimental"),
    ],
    extra={"parity_threshold": 0.05},
)
class QuantumLoopedMLP(LoopedMLP):
    """Approximate a variational/quantum substrate with measurement shot-noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        shot_noise: Number of measurement shots; noise scales as
            ``1 / sqrt(shot_noise)``.
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        shot_noise: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, hidden_dim, output_dim, **kwargs)
        self.shot_noise = int(shot_noise)

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Add measurement shot-noise to hidden amplitudes each step."""
        new_acts = super().forward_dynamics(activations, beta=beta, target=target)
        noise_scale = 1.0 / (self.shot_noise**0.5) if self.shot_noise > 0 else 0.0
        with torch.no_grad():
            for i in range(1, len(new_acts) - 1):
                new_acts[i] = new_acts[i] + torch.randn_like(new_acts[i]) * noise_scale
        return new_acts
