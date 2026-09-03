"""Feedback Alignment Kernel Backend.

Fused kernels for Feedback Alignment backward pass:
- Matmul + activation derivative fusion
- Batched outer product for weight gradients
- Support for dropout on feedback weights (Stochastic FA)
"""

from __future__ import annotations

import torch
from torch import Tensor

from computronium.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
)
from computronium.acceleration.kernel_backend import (
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
        is_cuda = config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)  # ruff: ignore[literal-membership]
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

        # Flatten input_dim if it's a tuple (spatial format like (C, H, W))
        if isinstance(input_dim, tuple):
            import math

            input_dim = math.prod(input_dim)

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
        activation: torch.nn.Module | None = None,
    ) -> None:
        """Set reference to model layers for weight updates.

        Feedback weights are (re)built from the bound layers' actual shapes so
        the backend stays consistent with the real model depth regardless of
        the ``num_layers`` hint in the kernel config. They are built **once**
        (only when empty) — FA's feedback weights are fixed across steps, so a
        per-step rebuild would inject fresh randomness into every update.
        """
        self._layers = layers
        if activation is not None:
            self._activation = activation
        if self._layers and not self._feedback_weights:
            device = self._layers[0].weight.device
            dtype = self._layers[0].weight.dtype
            self._feedback_weights = [
                torch.randn(
                    layer.out_features, layer.in_features, device=device, dtype=dtype
                )
                * 0.1
                for layer in self._layers
            ]

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


# Triton kernels for fused FA operations
try:  # ruff: ignore[too-many-statements-in-try-clause]
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _fa_feedback_projection_kernel(
        error_ptr,
        feedback_ptr,
        out_ptr,
        B,
        D_in,
        D_out,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused feedback weight projection: error @ B.T"""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_b = offs_b < B
        mask_d = offs_d < D_in

        # Accumulate error @ B.T
        acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        for k in range(0, D_out, BLOCK_D):
            offs_k = k + tl.arange(0, BLOCK_D)
            mask_k = offs_k < D_out

            # Load error tile [BLOCK_B, BLOCK_D]
            error_tile = tl.load(
                error_ptr + offs_b[:, None] * D_out + offs_k[None, :],
                mask=mask_b[:, None] & mask_k[None, :],
                other=0.0,
            )

            # Load feedback tile [BLOCK_D, BLOCK_D]
            fb_tile = tl.load(
                feedback_ptr + offs_d[:, None] * D_out + offs_k[None, :],
                mask=mask_d[:, None] & mask_k[None, :],
                other=0.0,
            )

            acc += tl.dot(error_tile, tl.trans(fb_tile), input_precision="ieee")

        tl.store(
            out_ptr + offs_b[:, None] * D_in + offs_d[None, :],
            acc,
            mask=mask_b[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _fa_activation_derivative_kernel(
        grad_ptr,
        h_ptr,
        out_ptr,
        n_elements,
        activation_type,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply activation derivative in-place."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        grad = tl.load(grad_ptr + offs, mask=mask)
        h = tl.load(h_ptr + offs, mask=mask)

        if activation_type == 0:  # ReLU
            deriv = (h > 0).to(tl.float32)
        elif activation_type == 1:  # SiLU
            sig = libdevice.sigmoid(h)
            deriv = sig * (1.0 + h * (1.0 - sig))
        elif activation_type == 2:  # Tanh
            deriv = 1.0 - h * h
        elif activation_type == 3:  # GELU (approximate)
            cdf = 0.5 * (1.0 + libdevice.erf(h * 0.7071067811865475))
            pdf = libdevice.exp(-h * h * 0.5) * 0.3989422804014327
            deriv = cdf + h * pdf
        else:
            deriv = (h > 0).to(tl.float32)

        out = grad * deriv
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _fa_batched_outer_kernel(
        pre_ptr,
        post_ptr,
        grad_ptr,
        B,
        D_in,
        D_out,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Fused batched outer product for weight gradients."""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            pre = tl.load(
                pre_ptr + b * D_in + offs_in[None, :],
                mask=mask_in[None, :],
                other=0.0,
            )
            post = tl.load(
                post_ptr + b * D_out + offs_out[:, None],
                mask=mask_out[:, None],
                other=0.0,
            )
            acc += tl.dot(tl.trans(post), pre)

        acc = acc / B  # ruff: ignore[non-augmented-assignment]
        tl.store(
            grad_ptr + offs_out[:, None] * D_in + offs_in[None, :],
            acc,
            mask=mask_out[:, None] & mask_in[None, :],
        )

    HAS_TRITON_FA = True
except ImportError:
    HAS_TRITON_FA = False


# Register the backend for all HardwareTargets
for hw in HardwareTarget:
    KernelRegistry.register(AlgorithmFamily.FA, hw, FAKernelBackend)


__all__ = ["HAS_TRITON_FA", "FAKernelBackend"]
