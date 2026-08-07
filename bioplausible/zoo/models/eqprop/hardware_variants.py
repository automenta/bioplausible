"""Hardware-faithful variants of :class:`LoopedMLP` (plan §17, §2).

These facades approximate non-Von-Neumann substrates so that the frontier
pipeline's ``cost_of_plausibility`` can be measured *hardware-aware* rather
than on an idealized digital GPU:

- :class:`QuantizedLoopedMLP` — FPGA / bit-precision: the hidden state is
  rounded to a signed ``bits``-bit representation each settle step.
- :class:`NoisyLoopedMLP` — analog/photonic: a Gaussian noise draw (shot +
  thermal) is added to the recurrent interaction every settle step.

Both keep float32 gradients (a surrogate for analog accumulation / high
precision integration) and otherwise share the equilibrium loop and training
path of their parent, so they drop into ``CoreTrainer`` and the probe driver
unchanged. Living here (instead of ``validation/tracks``) keeps them
importable from ``core.trainer`` without a `validation -> core` dependency.
"""

import torch

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
    tags=["eqprop", "looped_mlp", "fpga", "hardware", "quantization"],
    extra={"parity_threshold": 0.05},
)
class QuantizedLoopedMLP(LoopedMLP):
    """Approximate an FPGA substrate with fixed bit-precision hidden states.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Recurrent (equilibrium) state width.
        output_dim: Output logits width.
        bits: Signed fixed-point width (e.g. ``8`` for INT8). The hidden state
            is rounded to ``[-2**(bits-1)+1, 2**(bits-1)-1]`` each step.
        **kwargs: Remaining :class:`LoopedMLP` constructor arguments
            (``max_steps``, ``use_spectral_norm``, ...).

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

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_q = self._quantize(h)
        pre_act = x_transformed + self.W_rec(h_q)
        return torch.tanh(pre_act)


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
    tags=["eqprop", "looped_mlp", "analog", "photonic", "hardware", "noise"],
    extra={"parity_threshold": 0.05},
)
class NoisyLoopedMLP(LoopedMLP):
    """Approximate an analog/photonic substrate with continuous noise.

    Args:
        input_dim: Input dimensionality.
        hidden_dim: Recurrent (equilibrium) state width.
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

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(h) * self.noise_level
        pre_act = x_transformed + self.W_rec(h) + noise
        return torch.tanh(pre_act)
