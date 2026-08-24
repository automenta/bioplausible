"""Backprop Baseline Kernel Backend.

Fused BPTT kernel for the standard autograd family. Unlike the other
bio-plausible backends this is the *reference* the parity gates compare
against — it computes exact backpropagation gradients through the layer
stack (manual chain rule, no autograd graph) so a fused/settled kernel can be
verified to match within tolerance.

Memory complexity is O(L) because backprop stores an activation per layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from computronium.acceleration.contrastive_primitives import batched_outer_product
from computronium.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)

_ACTIVATIONS = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
}


class BackpropKernelBackend:
    """Fused backpropagation kernel backend (BPTT/reference baseline).

    Implements an explicit forward/backward chain-rule pass over a stack of
    ``nn.Linear`` layers. The backward pass computes each layer's weight and
    bias gradients from the back-propagated error and the cached activations,
    matching ``torch.autograd`` exactly (up to floating-point order).
    """

    name = AlgorithmFamily.BACKPROP
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = True
    requires_settle = False
    memory_complexity = "O(L)"
    locality_level = LocalityLevel.GLOBAL

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[nn.Linear] = []
        self._activation: nn.Module = nn.ReLU()
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def initialize(self, config: KernelConfig) -> None:
        """Initialize backend with configuration."""
        self._config = config
        is_cuda = config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
        self._device = torch.device("cuda" if is_cuda else "cpu")
        self._dtype = config.dtype
        activation_name = config.extra.get("activation", "relu")
        self._activation = _ACTIVATIONS.get(str(activation_name), nn.ReLU())

    def set_model_ref(self, layers: list[nn.Linear]) -> None:
        """Set reference to the model's linear layer stack."""
        self._layers = layers

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass returning output and per-layer activations.

        Returns:
            ``(output, activations)`` where ``activations = [x, h1, ..., out]``.
        """
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        activations: list[Tensor] = [x]
        h = x
        for i, layer in enumerate(self._layers):
            h = layer(h)
            if i < len(self._layers) - 1:
                h = self._activation(h)
            activations.append(h)
        return activations[-1], activations

    def backward(
        self,
        activations: list[Tensor],
        error: Tensor,
    ) -> dict[str, Tensor]:
        """Manual backprop: compute weight and bias gradients.

        Args:
            activations: ``[x, h1, ..., out]`` from :meth:`forward`.
            error: Output error (``output - target``) ``[B, D_out]``.

        Returns:
            Dict mapping ``layers.<i>.weight`` / ``layers.<i>.bias`` to gradients.
        """
        weight_grads: dict[str, Tensor] = {}
        bias_grads: dict[str, Tensor] = {}
        propagated = error

        for i in reversed(range(len(self._layers))):
            h_prev = activations[i]

            if i < len(self._layers) - 1:
                h_curr = activations[i + 1]
                propagated = propagated * _activation_deriv(h_curr, self._activation)

            weight_grads[f"layers.{i}.weight"] = batched_outer_product(
                h_prev, propagated
            )
            if self._layers[i].bias is not None:
                bias_grads[f"layers.{i}.bias"] = propagated.mean(dim=0)

            propagated = propagated @ self._layers[i].weight.data

        result = dict(weight_grads)
        result.update(bias_grads)
        return result

    def update_weights(self, gradients: dict[str, Tensor], lr: float) -> None:
        """Apply weight updates in-place."""
        with torch.no_grad():
            for name, grad in gradients.items():
                layer_idx = int(name.split(".")[1])
                if "weight" in name:
                    self._layers[layer_idx].weight.sub_(lr * grad)
                elif "bias" in name and self._layers[layer_idx].bias is not None:
                    self._layers[layer_idx].bias.sub_(lr * grad)

    def get_memory_stats(self) -> dict[str, float]:
        """Return memory usage stats (activations stored per layer: O(L))."""
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        return {
            "params_mb": total_params * 4 / 1e6,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        """Backprop has no settling dynamics."""
        return None


def _activation_deriv(h: Tensor, activation: nn.Module) -> Tensor:
    if isinstance(activation, nn.SiLU):
        sig = torch.sigmoid(h)
        return sig * (1 + h * (1 - sig))
    if isinstance(activation, nn.Tanh):
        return 1 - h**2
    if isinstance(activation, nn.GELU):
        cdf = 0.5 * (1 + torch.erf(h / 1.4142))
        pdf = torch.exp(-(h**2) / 2) / 2.5066
        return cdf + h * pdf
    return (h > 0).to(h.dtype)


# Register backend for all HardwareTargets
for hw in HardwareTarget:
    KernelRegistry.register(AlgorithmFamily.BACKPROP, hw, BackpropKernelBackend)

__all__ = ["BackpropKernelBackend"]
