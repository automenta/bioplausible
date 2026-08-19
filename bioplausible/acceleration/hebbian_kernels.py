"""Hebbian / 3-Factor Kernel Backend.

Batched outer product kernels for Hebbian and 3-factor learning rules.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
    contrastive_hebbian_update,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class HebbianKernelBackend:
    """Hebbian kernel backend with Oja's rule and 3-factor modulation."""

    name = AlgorithmFamily.HEBBIAN
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = False
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.LOCAL

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._use_oja: bool = True
        self._learning_rate: float = 0.01
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def initialize(self, config: KernelConfig) -> None:
        """Initialize backend with configuration."""
        self._config = config
        self._device = torch.device(
            "cuda"
            if config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
            else "cpu"
        )
        self._dtype = config.dtype

        extra = config.extra
        self._use_oja = extra.get("use_oja", True)
        self._learning_rate = extra.get("learning_rate", 0.01)

    def set_model_ref(self, layers: list[torch.nn.Linear]) -> None:
        """Set reference to model layers."""
        self._layers = layers

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass returning output and activations."""
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        activations: list[Tensor] = [x]
        h = x

        for i, layer in enumerate(self._layers):
            h = layer(h)
            if hasattr(self, "_activation") and i < len(self._layers) - 1:
                h = self._activation(h)
            activations.append(h)

        return activations[-1], activations

    def hebbian_update(
        self,
        pre: Tensor,
        post: Tensor,
        layer_idx: int,
    ) -> dict[str, Tensor]:
        """Single Hebbian update with Oja's rule.

        Delta W = lr * (post.T @ pre / B - post^2 @ W)

        Args:
            pre: Pre-synaptic activations [B, D_in]
            post: Post-synaptic activations [B, D_out]
            layer_idx: Layer index

        Returns:
            Dict with weight and bias gradients
        """
        layer = self._layers[layer_idx]

        # Hebbian term: post.T @ pre / B
        hebbian = batched_outer_product(pre, post)

        # Oja's subtraction term
        if self._use_oja:
            post_sq = post.pow(2).mean(dim=0, keepdim=True).T  # [D_out, 1]
            weight = layer.weight.data
            oja_term = post_sq * weight
            delta = self._learning_rate * (hebbian - oja_term)
        else:
            delta = self._learning_rate * hebbian

        result = {f"layers.{layer_idx}.weight": delta}

        if layer.bias is not None:
            bias_delta = self._learning_rate * post.mean(dim=0)
            result[f"layers.{layer_idx}.bias"] = bias_delta

        return result

    def backward(
        self,
        activations: list[Tensor],
        modulator: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full backward pass through all layers.

        Args:
            activations: [x, h1, h2, ..., output]
            modulator: Optional 3rd factor modulator [B, D_out] (for 3-factor)

        Returns:
            Dict mapping parameter names to gradients
        """
        all_grads: dict[str, Tensor] = {}

        for i in range(len(self._layers)):
            pre = activations[i]
            post = activations[i + 1]

            # Apply 3-factor modulation if provided
            if modulator is not None and i == len(self._layers) - 1:
                # Modulate output layer by error signal
                post_mod = post * modulator
                grads = self.hebbian_update(pre, post_mod, i)
            else:
                grads = self.hebbian_update(pre, post, i)

            all_grads.update(grads)

        return all_grads

    def backward_contrastive(
        self,
        free_activations: list[Tensor],
        nudged_activations: list[Tensor],
        beta: float,
        modulator: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Contrastive Hebbian update for all layers."""
        all_deltas: dict[str, Tensor] = {}

        for i in range(len(self._layers)):
            src_free = free_activations[i]
            dst_free = free_activations[i + 1]
            src_nudged = nudged_activations[i]
            dst_nudged = nudged_activations[i + 1]

            # Apply modulator to nudged phase if provided (3-factor)
            if modulator is not None and i == len(self._layers) - 1:
                dst_nudged = dst_nudged * modulator

            delta = contrastive_hebbian_update(
                src_free, dst_free, src_nudged, dst_nudged, self._learning_rate, beta
            )
            all_deltas[f"layers.{i}.weight"] = delta

            # Bias
            if self._layers[i].bias is not None:
                free_bias = dst_free.mean(dim=0)
                nudged_bias = dst_nudged.mean(dim=0)
                all_deltas[f"layers.{i}.bias"] = (
                    contrastive_delta(free_bias, nudged_bias, beta)
                    * self._learning_rate
                )

        return all_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float = 1.0) -> None:
        """Apply weight updates in-place (lr already baked into gradients)."""
        with torch.no_grad():
            for name, grad in gradients.items():
                if "weight" in name:
                    layer_idx = int(name.split(".")[1])
                    self._layers[layer_idx].weight.add_(grad)
                elif "bias" in name:
                    layer_idx = int(name.split(".")[1])
                    if self._layers[layer_idx].bias is not None:
                        self._layers[layer_idx].bias.add_(grad)

    def get_memory_stats(self) -> dict[str, float]:
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        return {
            "params_mb": total_params * 4 / 1e6,
            "activations_mb": 0.0,  # O(1) - no activation storage needed
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return None


class ThreeFactorKernelBackend(HebbianKernelBackend):
    """Three-Factor Hebbian kernel backend.

    Delta W = lr * M * pre * post
    where M is a neuromodulatory signal.
    """

    name = AlgorithmFamily.HEBBIAN  # Same family, different implementation
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = False
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.LOCAL

    def __init__(self) -> None:
        super().__init__()
        self._out_layer: torch.nn.Linear | None = None

    def set_model_ref(
        self,
        layers: list[torch.nn.Linear],
        out_layer: torch.nn.Linear,
    ) -> None:
        self._layers = layers
        self._out_layer = out_layer

    def backward(
        self,
        activations: list[Tensor],
        output_modulator: Tensor,
    ) -> dict[str, Tensor]:
        """3-factor backward: backproject output modulator to hidden layers.

        Args:
            activations: [x, h1, h2, ..., output]
            output_modulator: Error signal at output [B, D_out] (continuous)

        Returns:
            Weight deltas for all layers
        """
        all_grads: dict[str, Tensor] = {}

        # Backproject modulator through output weights
        hidden_modulator = output_modulator @ self._out_layer.weight.data
        hidden_modulator = hidden_modulator / max(
            hidden_modulator.abs().max().item(), 1.0
        )

        # Hidden layers
        for i in range(len(self._layers)):
            pre = activations[i]
            post = activations[i + 1]

            if i == len(self._layers) - 1:
                # Last hidden layer gets backprojected modulator
                post_mod = post * hidden_modulator
                grads = self.hebbian_update(pre, post_mod, i)
            else:
                grads = self.hebbian_update(pre, post, i)

            all_grads.update(grads)

        # Output layer: direct error modulation
        if self._out_layer is not None:
            pre = activations[-2]
            error = output_modulator
            delta = self._learning_rate * batched_outer_product(pre, error)
            all_grads["out_layer.weight"] = delta
            if self._out_layer.bias is not None:
                all_grads["out_layer.bias"] = self._learning_rate * error.mean(dim=0)

        return all_grads


# Register backends for all HardwareTargets
for hw in HardwareTarget:
    KernelRegistry.register(AlgorithmFamily.HEBBIAN, hw, HebbianKernelBackend)
# ThreeFactorKernelBackend is a variant; register explicitly if needed by users
# for hw in HardwareTarget:
#     KernelRegistry.register(AlgorithmFamily.HEBBIAN, hw, ThreeFactorKernelBackend)


# Triton kernels for fused Hebbian operations
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _hebbian_update_kernel(
        pre_ptr,
        post_ptr,
        weight_ptr,
        delta_ptr,
        B,
        D_in,
        D_out,
        lr,
        use_oja,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Hebbian weight update with Oja's rule: Delta W = lr * (post.T @ pre / B - post^2 @ W)"""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            pre = tl.load(pre_ptr + b * D_in + offs_in[None, :], mask=mask_in[None, :], other=0.0)
            post = tl.load(post_ptr + b * D_out + offs_out[:, None], mask=mask_out[:, None], other=0.0)
            acc += tl.dot(tl.trans(post), pre)

        acc = acc / B

        if use_oja:
            # Oja's subtraction term: post^2 @ W
            post_sq = tl.zeros((BLOCK_OUT, 1), dtype=tl.float32)
            for b in range(B):
                post = tl.load(post_ptr + b * D_out + offs_out[:, None], mask=mask_out[:, None], other=0.0)
                post_sq += post * post
            post_sq = post_sq / B

            weight = tl.load(weight_ptr + offs_out[:, None] * D_in + offs_in[None, :], mask=mask_out[:, None] & mask_in[None, :], other=0.0)
            acc = acc - post_sq * weight

        delta = lr * acc

        tl.store(delta_ptr + offs_out[:, None] * D_in + offs_in[None, :], delta, mask=mask_out[:, None] & mask_in[None, :])

    @triton.jit
    def _three_factor_hebbian_kernel(
        pre_ptr,
        post_ptr,
        modulator_ptr,
        delta_ptr,
        B,
        D_in,
        D_out,
        lr,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Three-factor Hebbian: Delta W = lr * modulator * (post.T @ pre / B)"""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            pre = tl.load(pre_ptr + b * D_in + offs_in[None, :], mask=mask_in[None, :], other=0.0)
            post = tl.load(post_ptr + b * D_out + offs_out[:, None], mask=mask_out[:, None], other=0.0)
            mod = tl.load(modulator_ptr + b * D_out + offs_out[:, None], mask=mask_out[:, None], other=0.0)
            post_mod = post * mod
            acc += tl.dot(tl.trans(post_mod), pre)

        acc = acc / B
        delta = lr * acc

        tl.store(delta_ptr + offs_out[:, None] * D_in + offs_in[None, :], delta, mask=mask_out[:, None] & mask_in[None, :])

    @triton.jit
    def _contrastive_hebbian_kernel(
        pre_free_ptr,
        post_free_ptr,
        pre_nudged_ptr,
        post_nudged_ptr,
        delta_ptr,
        B,
        D_in,
        D_out,
        lr,
        beta,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Contrastive Hebbian update."""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc_free = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)
        acc_nudged = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            pre_f = tl.load(pre_free_ptr + b * D_in + offs_in[None, :], mask=mask_in[None, :], other=0.0)
            post_f = tl.load(post_free_ptr + b * D_out + offs_out[:, None], mask=mask_out[:, None], other=0.0)
            acc_free += tl.dot(tl.trans(post_f), pre_f)

            pre_n = tl.load(pre_nudged_ptr + b * D_in + offs_in[None, :], mask=mask_in[None, :], other=0.0)
            post_n = tl.load(post_nudged_ptr + b * D_out + offs_out[:, None], mask=mask_out[:, None], other=0.0)
            acc_nudged += tl.dot(tl.trans(post_n), pre_n)

        acc_free = acc_free / B
        acc_nudged = acc_nudged / B

        delta = lr * (acc_nudged - acc_free) / beta

        tl.store(delta_ptr + offs_out[:, None] * D_in + offs_in[None, :], delta, mask=mask_out[:, None] & mask_in[None, :])

    HAS_TRITON_HEBBIAN = True
except ImportError:
    HAS_TRITON_HEBBIAN = False


__all__ = ["HAS_TRITON_HEBBIAN", "HebbianKernelBackend", "ThreeFactorKernelBackend"]
