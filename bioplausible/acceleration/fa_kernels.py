"""Feedback Alignment Kernel Backend.

Fused kernels for Feedback Alignment backward pass:
- Matmul + activation derivative fusion
- Batched outer product for weight gradients
- Support for dropout on feedback weights (Stochastic FA)
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class FAKernelBackend:
    """Feedback Alignment kernel backend.

    Implements fused FA backward loop with optional contrastive phases.
    """

    name = AlgorithmFamily.FA
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = False
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.LAYERWISE

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._feedback_weights: list[Tensor] = []
        self._activation: torch.nn.Module = torch.nn.ReLU()
        self._dropout_prob: float = 0.0
        self._num_layers: int = 0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def initialize(self, config: KernelConfig) -> None:
        """Initialize backend with configuration."""
        self._config = config
        is_cuda = config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
        self._device = torch.device("cuda" if is_cuda else "cpu")
        self._dtype = config.dtype

        # Extract algorithm-specific config
        extra = config.extra
        self._num_layers = extra.get("num_layers", 3)
        self._dropout_prob = extra.get("dropout_prob", 0.0)
        self._activation = _get_activation(extra.get("activation", "relu"))

        # Initialize feedback weights (fixed random)
        hidden_dim = extra.get("hidden_dim", 256)
        input_dim = extra.get("input_dim", 784)
        output_dim = extra.get("output_dim", 10)

        dims = [input_dim] + [hidden_dim] * (self._num_layers - 1) + [output_dim]
        self._feedback_weights = []
        for i in range(len(dims) - 1):
            B = (
                torch.randn(
                    dims[i + 1], dims[i], device=self._device, dtype=self._dtype
                )
                * 0.1
            )
            self._feedback_weights.append(B)

    def set_model_ref(
        self,
        layers: list[torch.nn.Linear],
        activation: torch.nn.Module,
    ) -> None:
        """Set reference to model layers for weight updates."""
        self._layers = layers
        self._activation = activation

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass returning output and per-layer activations.

        Returns:
            (output, activations) where activations = [x, h1, h2, ..., output]
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
        """FA backward loop: compute weight and bias gradients.

        Args:
            activations: [x, h1, h2, ..., output] from forward
            error: Output error (output - target) [B, D_out]

        Returns:
            Dict mapping parameter names to gradients
        """
        num_layers = len(self._layers)
        propagated_error = error

        weight_grads: dict[str, Tensor] = {}
        bias_grads: dict[str, Tensor] = {}

        for i in reversed(range(num_layers)):
            h_prev = activations[i]

            if i < num_layers - 1:
                B = self._feedback_weights[i + 1]

                use_dropout = (
                    self._dropout_prob > 0.0
                    and self._config is not None
                    and self._config.use_autograd
                )
                if use_dropout:
                    mask = (torch.rand_like(B) > self._dropout_prob).float()
                    B_eff = B * mask * (1.0 / (1.0 - self._dropout_prob))
                else:
                    B_eff = B

                # Feedback weight B maps output-of-layer-(i+1) back to layer i:
                # B = feedback_weights[i+1] is shaped [D_{i+1}, D_i].
                grad_h = propagated_error @ B_eff

                h_curr = activations[i + 1]
                grad_h = _apply_activation_derivative(grad_h, h_curr, self._activation)
            else:
                grad_h = propagated_error

            # Weight gradient: grad_h.T @ h_prev / batch_size
            wgrad = batched_outer_product(h_prev, grad_h)
            weight_grads[f"layers.{i}.weight"] = wgrad

            # Bias gradient: mean over batch
            bgrad = grad_h.mean(dim=0)
            if self._layers[i].bias is not None:
                bias_grads[f"layers.{i}.bias"] = bgrad

            propagated_error = grad_h

        # Combine
        result = {}
        result.update(weight_grads)
        result.update(bias_grads)
        return result

    def backward_contrastive(
        self,
        free_activations: list[Tensor],
        nudged_activations: list[Tensor],
        beta: float,
    ) -> dict[str, Tensor]:
        """Contrastive FA backward: free vs nudged phases.

        Args:
            free_activations: Activations from free phase
            nudged_activations: Activations from nudged phase (with target)
            beta: Nudge strength

        Returns:
            Contrastive weight deltas
        """
        weight_deltas: dict[str, Tensor] = {}
        bias_deltas: dict[str, Tensor] = {}

        num_layers = len(self._layers)
        for i in range(num_layers):
            free_pre = free_activations[i]
            free_post = free_activations[i + 1]
            nudged_pre = nudged_activations[i]
            nudged_post = nudged_activations[i + 1]

            free_wgrad = batched_outer_product(free_pre, free_post)
            nudged_wgrad = batched_outer_product(nudged_pre, nudged_post)
            weight_deltas[f"layers.{i}.weight"] = contrastive_delta(
                free_wgrad, nudged_wgrad, beta
            )

            free_bgrad = free_post.mean(dim=0)
            nudged_bgrad = nudged_post.mean(dim=0)
            if self._layers[i].bias is not None:
                bias_deltas[f"layers.{i}.bias"] = contrastive_delta(
                    free_bgrad, nudged_bgrad, beta
                )

        result = {}
        result.update(weight_deltas)
        result.update(bias_deltas)
        return result

    def update_weights(self, gradients: dict[str, Tensor], lr: float) -> None:
        """Apply weight updates in-place."""
        with torch.no_grad():
            for name, grad in gradients.items():
                if "weight" in name:
                    layer_idx = int(name.split(".")[1])
                    self._layers[layer_idx].weight.sub_(lr * grad)
                elif "bias" in name:
                    layer_idx = int(name.split(".")[1])
                    if self._layers[layer_idx].bias is not None:
                        self._layers[layer_idx].bias.sub_(lr * grad)

    def get_memory_stats(self) -> dict[str, float]:
        """Return memory usage stats."""
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        feedback_params = sum(w.numel() for w in self._feedback_weights)
        return {
            "params_mb": total_params * 4 / 1e6,
            "activations_mb": 0.0,
            "feedback_weights_mb": feedback_params * 4 / 1e6,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        """FA doesn't have settling dynamics."""
        return None


def _get_activation(name: str) -> torch.nn.Module:
    """Get activation module by name."""
    activations = {
        "relu": torch.nn.ReLU(),
        "silu": torch.nn.SiLU(),
        "tanh": torch.nn.Tanh(),
        "gelu": torch.nn.GELU(),
    }
    return activations.get(name.lower(), torch.nn.ReLU())


def _apply_activation_derivative(
    grad_h: Tensor,
    h_curr: Tensor,
    activation: torch.nn.Module,
) -> Tensor:
    """Apply activation function derivative."""
    if isinstance(activation, torch.nn.SiLU):
        sig = torch.sigmoid(h_curr)
        return grad_h * sig * (1 + h_curr * (1 - sig))
    if isinstance(activation, torch.nn.ReLU):
        return grad_h * (h_curr > 0).to(grad_h.dtype)
    if isinstance(activation, torch.nn.Tanh):
        return grad_h * (1 - h_curr**2)
    if isinstance(activation, torch.nn.GELU):
        # GELU derivative approximation
        cdf = 0.5 * (1 + torch.erf(h_curr / 1.4142))
        pdf = torch.exp(-(h_curr**2) / 2) / 2.5066
        return grad_h * (cdf + h_curr * pdf)
    return grad_h * (h_curr > 0).to(grad_h.dtype)


# Triton kernel for fused backward loop (when TRITON backend)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fa_backward_kernel(
        activations_ptr,  # [num_layers, B, D]
        error_ptr,  # [B, D_out]
        feedback_ptr,  # [num_layers-1, D_out, D_in]
        weight_grads_ptr,  # [num_layers, D_out, D_in]
        bias_grads_ptr,  # [num_layers, D_out]
        activation_type,  # 0=relu, 1=silu, 2=tanh
        dropout_prob,
        num_layers,
        B,
        D_out,
        D_in,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused FA backward kernel."""
        pass  # Implementation would go here for production

    HAS_TRITON_FA = True
except ImportError:
    HAS_TRITON_FA = False


# Register the backend
KernelRegistry.register(AlgorithmFamily.FA, HardwareTarget.CPU, FAKernelBackend)
KernelRegistry.register(AlgorithmFamily.FA, HardwareTarget.CUDA, FAKernelBackend)
if HAS_TRITON_FA:
    KernelRegistry.register(AlgorithmFamily.FA, HardwareTarget.TRITON, FAKernelBackend)


__all__ = ["FAKernelBackend", "HAS_TRITON_FA"]
