"""Hardware-faithful variants of :class:`LoopedMLP` (plan §17, §2).

These facades approximate non-Von-Neumann substrates so that the frontier
pipeline's ``cost_of_plausibility`` can be measured *hardware-aware* rather
than on an idealized digital GPU:

- :class:`QuantizedLoopedMLP` — FPGA / bit-precision: every hidden-layer
  activation is rounded to a signed ``bits``-bit representation each settle
  step.
- :class:`NoisyLoopedMLP` — analog/photonic: a Gaussian noise draw (shot +
  thermal) is added to every hidden-layer pre-activation each settle step.

Both keep float32 gradients (a surrogate for analog accumulation / high-
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

__all__ = ["NoisyLoopedMLP", "QuantizedLoopedMLP"]


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
