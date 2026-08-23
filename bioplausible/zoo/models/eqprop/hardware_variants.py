"""Hardware-faithful variants of :class:`EquilibriumMLP` (plan §17, §2, REFACTOR7 §4).

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

from bioplausible.config.unified import ModelConfig
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import LocalityLevel, register_model

from ._energy import EquilibriumMLP

__all__ = [
    "CrossbarLoopedMLP",
    "NoisyLoopedMLP",
    "OpticalLoopedMLP",
    "QuantizedLoopedMLP",
    "QuantumLoopedMLP",
    "SpikingLoopedMLP",
]


def _make_config(
    name: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    **extra,
) -> ModelConfig:
    """Create a ModelConfig for hardware variant constructors."""
    return ModelConfig(
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple([hidden_dim] * max(num_layers, 1)),
        num_layers=max(num_layers, 1),
        learning_rate=extra.pop("learning_rate", 0.01),
        beta=extra.pop("beta", 0.5),
        max_steps=extra.pop("max_steps", 30),
        convergence_threshold=extra.pop("convergence_threshold", 1e-4),
        convergence_start=extra.pop("convergence_start", 5),
        use_spectral_norm=extra.pop("use_spectral_norm", True),
        spectral_norm_power_iterations=extra.pop("spectral_norm_power_iterations", 5),
        activation=extra.pop("activation", "tanh"),
        lipschitz_mode=extra.pop("lipschitz_mode", "power_iteration"),
        output_scaling_mode=extra.pop("output_scaling_mode", "uniform"),
        dropout=extra.pop("dropout", 0.0),
        neurons_per_tile=extra.pop("neurons_per_tile", 48),
        tiles_per_layer=extra.pop("tiles_per_layer", 4),
        algorithm=extra.pop("algorithm", "ep"),
        mode=extra.pop("mode", "ep"),
        inference_steps=extra.pop("inference_steps", 10),
        step_size=extra.pop("step_size", 0.1),
        input_channels=extra.pop("input_channels", 3),
        input_size=extra.pop("input_size", 32),
        conv_channels=extra.pop("conv_channels", (32, 64, 128)),
        kernel_sizes=extra.pop("kernel_sizes", (3, 3, 3)),
        use_pooling=extra.pop("use_pooling", True),
        pooling_size=extra.pop("pooling_size", 2),
        attention_heads=extra.pop("attention_heads", 4),
        use_positional_encoding=extra.pop("use_positional_encoding", True),
        use_temporal_attention=extra.pop("use_temporal_attention", True),
        seq_len=extra.pop("seq_len", 64),
        hidden_dim=extra.pop("hidden_dim", hidden_dim),
        obs_dim=extra.pop("obs_dim", 8),
        action_dim=extra.pop("action_dim", 4),
        action_type=extra.pop("action_type", "discrete"),
        log_std_init=extra.pop("log_std_init", 0.0),
        log_std_min=extra.pop("log_std_min", -20.0),
        log_std_max=extra.pop("log_std_max", 2.0),
        entropy_coef=extra.pop("entropy_coef", 0.01),
        value_coef=extra.pop("value_coef", 0.5),
        max_grad_norm=extra.pop("max_grad_norm", 0.5),
        node_features=extra.pop("node_features", 10),
        aggregation=extra.pop("aggregation", "mean"),
        readout=extra.pop("readout", "mean"),
        extra=extra,
    )


@register_model(
    "quantized_looped_mlp",
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
class QuantizedLoopedMLP(EquilibriumMLP):
    """Approximate an FPGA substrate with fixed bit-precision hidden states.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        bits: Signed fixed-point width (e.g. ``8`` for INT8). Each hidden
            activation is rounded to ``[-2**(bits-1)+1, 2**(bits-1)-1]`` every
            settle step.
        **kwargs: Remaining :class:`EquilibriumMLP` constructor arguments
            (``num_layers``, ``max_steps``, ``use_spectral_norm``, ...).

    Gradients stay float: this simulates high-precision accumulators with a
    quantized interaction (surrogate-gradient style).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, bits: int = 8, **kwargs
    ) -> None:
        config = _make_config(
            "quantized_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            **kwargs,
        )
        super().__init__(config=config)
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
class NoisyLoopedMLP(EquilibriumMLP):
    """Approximate an analog/photonic substrate with continuous noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        noise_level: Std-dev of the per-step Gaussian noise (as a fraction of
            the activation scale).
        **kwargs: Remaining :class:`EquilibriumMLP` constructor arguments.

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
        config = _make_config(
            "noisy_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            **kwargs,
        )
        super().__init__(config=config)
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
class SpikingLoopedMLP(EquilibriumMLP):
    """Approximate a neuromorphic substrate with LIF-style spike-and-reset.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        spike_threshold: Membrane voltage at which a neuron emits a spike and
            is reset to zero.
        refractory_period: Number of settle steps a neuron stays clamped at
            zero after spiking.
        **kwargs: Remaining :class:`EquilibriumMLP` constructor arguments.

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
        config = _make_config(
            "spiking_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            **kwargs,
        )
        super().__init__(config=config)
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
class OpticalLoopedMLP(EquilibriumMLP):
    """Approximate an optical substrate with phase- and detector-noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        phase_noise: Std-dev of the phase-noise applied to hidden amplitudes.
        detector_noise: Std-dev of the additive detector/readout noise.
        **kwargs: Remaining :class:`EquilibriumMLP` constructor arguments.
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
        config = _make_config(
            "optical_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            **kwargs,
        )
        super().__init__(config=config)
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
class CrossbarLoopedMLP(EquilibriumMLP):
    """Approximate an analog crossbar with ADC-quantised conductance readout.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        adc_bits: ADC resolution for the conductance readout.
        ir_drop_factor: Fraction of signal lost to IR drop along the array.
        **kwargs: Remaining :class:`EquilibriumMLP` constructor arguments.
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
        config = _make_config(
            "crossbar_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            **kwargs,
        )
        super().__init__(config=config)
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
class QuantumLoopedMLP(EquilibriumMLP):
    """Approximate a variational/quantum substrate with measurement shot-noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Equilibrium state width (per hidden layer).
        output_dim: Output logits width.
        shot_noise: Number of measurement shots; noise scales as
            ``1 / sqrt(shot_noise)``.
        **kwargs: Remaining :class:`EquilibriumMLP` constructor arguments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        shot_noise: int = 1000,
        **kwargs,
    ) -> None:
        config = _make_config(
            "quantum_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            **kwargs,
        )
        super().__init__(config=config)
        self.shot_noise = int(shot_noise)

    def forward_dynamics(self, activations, beta: float = 0.0, target=None):
        """Add measurement shot-noise to hidden amplitudes each step."""
        new_acts = super().forward_dynamics(activations, beta=beta, target=target)
        noise_scale = 1.0 / (self.shot_noise**0.5) if self.shot_noise > 0 else 0.0
        with torch.no_grad():
            for i in range(1, len(new_acts) - 1):
                new_acts[i] = new_acts[i] + torch.randn_like(new_acts[i]) * noise_scale
        return new_acts
