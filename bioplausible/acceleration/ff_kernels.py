"""Forward-Forward / PEPITA Kernel Backend.

Fused kernels for Forward-Forward goodness and PEPITA error-modulated updates.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
    pepita_error_modulation,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class FFKernelBackend:
    """Forward-Forward kernel backend.

    Implements the Forward-Forward algorithm with positive/negative passes
    and goodness-based contrastive updates.
    """

    name = AlgorithmFamily.FF
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = False
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.FORWARD_ONLY

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._threshold: float = 1.0
        self._num_layers: int = 0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._activation: torch.nn.Module = torch.nn.ReLU()

    def initialize(self, config: KernelConfig) -> None:
        """Initialize backend with configuration."""
        self._config = config
        is_cuda = config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
        self._device = torch.device("cuda" if is_cuda else "cpu")
        self._dtype = config.dtype

        extra = config.extra
        self._threshold = extra.get("threshold", 1.0)
        self._num_layers = extra.get("num_layers", 3)
        self._input_dim = extra.get("input_dim", 784)
        self._output_dim = extra.get("output_dim", 10)
        self._activation = _get_activation(extra.get("activation", "relu"))

    def set_model_ref(
        self,
        layers: list[torch.nn.Linear],
        activation: torch.nn.Module | None = None,
    ) -> None:
        """Set reference to model layers."""
        self._layers = layers
        if activation is not None:
            self._activation = activation

    def forward_positive(self, x: Tensor, y: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Positive pass with label information.

        Args:
            x: Input [B, D_in]
            y: Labels [B]

        Returns:
            (output, activations) with label information embedded
        """
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Embed label into input (concatenate or add)
        # Standard FF: concatenate one-hot label to input (output_dim classes)
        y_onehot = torch.nn.functional.one_hot(
            y, num_classes=self._output_dim
        ).float().to(device=self._device, dtype=self._dtype)
        x_pos = torch.cat([x, y_onehot], dim=1)

        return self._forward_layers(x_pos)

    def forward_negative(self, x: Tensor, y: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Negative pass with wrong label information."""
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Use incorrect label (shift by 1)
        y_wrong = (y + 1) % self._output_dim
        y_onehot = torch.nn.functional.one_hot(
            y_wrong, num_classes=self._output_dim
        ).float().to(device=self._device, dtype=self._dtype)
        x_neg = torch.cat([x, y_onehot], dim=1)

        return self._forward_layers(x_neg)

    def _forward_layers(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward through all layers."""
        activations: list[Tensor] = [x]
        h = x

        for i, layer in enumerate(self._layers):
            h = layer(h)
            if i < len(self._layers) - 1:
                h = self._activation(h)
            activations.append(h)

        return activations[-1], activations

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Standard forward (for inference)."""
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self._forward_layers(x)

    def compute_goodness(
        self,
        pos_acts: list[Tensor],
        neg_acts: list[Tensor],
    ) -> dict[int, Tensor]:
        """Compute goodness per layer.

        Goodness = ||pos||^2 - ||neg||^2 - threshold

        Returns:
            Dict mapping layer_idx -> goodness per sample [B]
        """
        goodness: dict[int, Tensor] = {}
        for i in range(1, len(pos_acts)):  # Skip input layer
            pos_norm = pos_acts[i].pow(2).sum(dim=1)
            neg_norm = neg_acts[i].pow(2).sum(dim=1)
            goodness[i - 1] = pos_norm - neg_norm - self._threshold
        return goodness

    def backward(
        self,
        pos_activations: list[Tensor],
        neg_activations: list[Tensor],
    ) -> dict[str, Tensor]:
        """FF backward: contrastive update based on goodness.

        Layer update: Delta W = lr * (pos_acts.T @ pos_acts - neg_acts.T @ neg_acts) / B

        Returns:
            Weight deltas for all layers
        """
        weight_deltas: dict[str, Tensor] = {}

        for i in range(len(self._layers)):
            pos_pre = pos_activations[i]
            pos_post = pos_activations[i + 1]
            neg_pre = neg_activations[i]
            neg_post = neg_activations[i + 1]

            # FF contrastive update
            pos_grad = batched_outer_product(pos_pre, pos_post)
            neg_grad = batched_outer_product(neg_pre, neg_post)
            delta = contrastive_delta(neg_grad, pos_grad, beta=1.0)  # pos - neg

            weight_deltas[f"layers.{i}.weight"] = delta

            if self._layers[i].bias is not None:
                pos_bias = pos_post.mean(dim=0)
                neg_bias = neg_post.mean(dim=0)
                weight_deltas[f"layers.{i}.bias"] = pos_bias - neg_bias

        return weight_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float) -> None:
        """Apply weight updates."""
        with torch.no_grad():
            for name, grad in gradients.items():
                if "weight" in name:
                    layer_idx = int(name.split(".")[1])
                    self._layers[layer_idx].weight.add_(lr * grad)
                elif "bias" in name:
                    layer_idx = int(name.split(".")[1])
                    if self._layers[layer_idx].bias is not None:
                        self._layers[layer_idx].bias.add_(lr * grad)

    def get_memory_stats(self) -> dict[str, float]:
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        return {
            "params_mb": total_params * 4 / 1e6,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return None


class PEPITAKernelBackend:
    """PEPITA kernel backend.

    Error-modulated forward-forward with fixed feedback matrix.
    """

    name = AlgorithmFamily.PEPITA
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = False
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.FORWARD_ONLY

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._feedback_matrix: Tensor | None = None
        self._scale: float = 1.0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._activation: torch.nn.Module = torch.nn.ReLU()

    def initialize(self, config: KernelConfig) -> None:
        """Initialize backend with configuration."""
        self._config = config
        is_cuda = config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
        self._device = torch.device("cuda" if is_cuda else "cpu")
        self._dtype = config.dtype

        extra = config.extra
        self._scale = extra.get("feedback_matrix_scale", 1.0)
        self._activation = _get_activation(extra.get("activation", "relu"))

        # Create fixed random feedback matrix
        input_dim = extra.get("input_dim", 784)
        output_dim = extra.get("output_dim", 10)
        self._feedback_matrix = (
            torch.randn(input_dim, output_dim, device=self._device, dtype=self._dtype)
            * 0.1
        )

    def set_model_ref(
        self,
        layers: list[torch.nn.Linear],
        activation: torch.nn.Module | None = None,
    ) -> None:
        self._layers = layers
        if activation is not None:
            self._activation = activation

    def forward_standard(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Standard forward pass."""
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

    def forward_error_modulated(
        self, x: Tensor, error: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """Error-modulated forward pass (PEPITA).

        Error is backprojected through fixed feedback matrix and added to input.
        """
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Backproject error to input space
        error_input = error @ self._feedback_matrix.T * self._scale
        x_modulated = x + error_input

        return self._forward_layers(x_modulated)

    def _forward_layers(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
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
        standard_activations: list[Tensor],
        error_activations: list[Tensor],
        error: Tensor,
    ) -> dict[str, Tensor]:
        """PEPITA backward: error-modulated contrastive update.

        For every layer, Delta W = scale * (a_err.T @ pre - a_std.T @ pre) / B,
        i.e. the difference in pre-synaptic x post-synaptic correlation between
        the error-modulated and standard passes (mirrors the reference's
        ``layer.weight -= lr * delta_a.T @ inp / B``).
        """
        weight_deltas: dict[str, Tensor] = {}

        for i in range(len(self._layers)):
            std_pre = standard_activations[i]
            std_post = standard_activations[i + 1]
            err_pre = error_activations[i]
            err_post = error_activations[i + 1]

            std_grad = batched_outer_product(std_pre, std_post)
            err_grad = batched_outer_product(err_pre, err_post)
            delta = self._scale * contrastive_delta(std_grad, err_grad, beta=1.0)

            weight_deltas[f"layers.{i}.weight"] = delta

            if self._layers[i].bias is not None:
                weight_deltas[f"layers.{i}.bias"] = self._scale * (
                    err_post.mean(dim=0) - std_post.mean(dim=0)
                )

        return weight_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float) -> None:
        with torch.no_grad():
            for name, grad in gradients.items():
                if "weight" in name:
                    layer_idx = int(name.split(".")[1])
                    self._layers[layer_idx].weight.add_(lr * grad)
                elif "bias" in name:
                    layer_idx = int(name.split(".")[1])
                    if self._layers[layer_idx].bias is not None:
                        self._layers[layer_idx].bias.add_(lr * grad)

    def get_memory_stats(self) -> dict[str, float]:
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        fb_mb = 0.0
        if self._feedback_matrix is not None:
            fb_mb = self._feedback_matrix.numel() * 4 / 1e6
        return {
            "params_mb": total_params * 4 / 1e6,
            "feedback_mb": fb_mb,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return None


def _get_activation(name: str) -> torch.nn.Module:
    activations = {
        "relu": torch.nn.ReLU(),
        "silu": torch.nn.SiLU(),
        "tanh": torch.nn.Tanh(),
        "gelu": torch.nn.GELU(),
    }
    return activations.get(name.lower(), torch.nn.ReLU())


# Register backends
KernelRegistry.register(AlgorithmFamily.FF, HardwareTarget.CPU, FFKernelBackend)
KernelRegistry.register(AlgorithmFamily.FF, HardwareTarget.CUDA, FFKernelBackend)
KernelRegistry.register(AlgorithmFamily.FF, HardwareTarget.TRITON, FFKernelBackend)

KernelRegistry.register(AlgorithmFamily.PEPITA, HardwareTarget.CPU, PEPITAKernelBackend)
KernelRegistry.register(
    AlgorithmFamily.PEPITA, HardwareTarget.CUDA, PEPITAKernelBackend
)
KernelRegistry.register(
    AlgorithmFamily.PEPITA, HardwareTarget.TRITON, PEPITAKernelBackend
)


__all__ = ["FFKernelBackend", "PEPITAKernelBackend"]
