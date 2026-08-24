"""Predictive Coding Kernel Backend.

Graph-parallel inference + PCN loss kernels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from computronium.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
    predictive_coding_inference_step,
)
from computronium.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class PCKernelBackend:
    """Predictive Coding kernel backend.

    Implements Predictive Coding Network (PCN) with:
    - Graph-parallel inference (all layers settle simultaneously)
    - PCN loss: sum of prediction errors
    - Local weight updates from prediction errors
    """

    name = AlgorithmFamily.PC
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = True
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.LOCAL

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._infer_steps: int = 10
        self._eta_infer: float = 0.1
        self._eta_weight: float = 0.01
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._activation: str = "tanh"
        self._mu: list[Tensor] | None = None  # State estimates
        self._last_settle_telemetry: dict[str, object] | None = None

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
        self._infer_steps = extra.get("infer_steps", 10)
        self._eta_infer = extra.get("eta_infer", 0.1)
        self._eta_weight = extra.get("eta_weight", 0.01)
        self._activation = extra.get("activation", "tanh")

    def set_model_ref(
        self,
        layers: list[torch.nn.Linear],
        activation: str | None = None,
    ) -> None:
        self._layers = layers
        if activation is not None:
            self._activation = activation

    def init_states(self, x: Tensor) -> list[Tensor]:
        """Initialize state estimates (mu) for all layers."""
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        mu = [x]
        h = x

        for i, layer in enumerate(self._layers):
            h = layer(h)
            if i < len(self._layers) - 1:
                h = _apply_activation(h, self._activation)
            mu.append(h)

        self._mu = mu
        return mu

    def settle(
        self,
        x: Tensor,
        y: Tensor | None = None,
        steps: int | None = None,
    ) -> tuple[list[Tensor], dict[str, float]]:
        """Run inference to settle states (minimize prediction error).

        Args:
            x: Input
            y: Target (for clamped phase)
            steps: Number of inference steps

        Returns:
            (settled_states, telemetry)
        """
        if self._mu is None:
            self.init_states(x)

        # Clamp output to target if provided
        if y is not None and self._mu is not None:
            y_onehot = (
                torch.nn.functional
                .one_hot(y, num_classes=self._mu[-1].shape[1])
                .float()
                .to(device=self._device, dtype=self._dtype)
            )
            self._mu[-1] = y_onehot

        infer_steps = steps or self._infer_steps
        W = [layer.weight.data for layer in self._layers]
        b = [
            layer.bias.data if layer.bias is not None else None
            for layer in self._layers
        ]

        telemetry = {"steps": infer_steps, "converged": False, "final_error": 0.0}
        prev_energy = float("inf")

        for step in range(infer_steps):
            self._mu = predictive_coding_inference_step(
                self._mu, x, W, b, self._eta_infer, activation=self._activation
            )

            # Check convergence via energy
            if step % 5 == 0:
                energy = self.compute_energy(x)
                if step > 0 and abs(energy - prev_energy) < 1e-6:
                    telemetry["converged"] = True
                    telemetry["steps"] = step + 1
                    break

        telemetry["final_error"] = self.compute_energy(x)
        self._last_settle_telemetry = telemetry
        return self._mu, telemetry

    def compute_energy(self, x: Tensor) -> float:
        """Compute total prediction error (energy).

        Standard PCN energy: each non-input state is compared to its prediction
        from the layer below, ``E = sum_i 0.5 * ||mu_i - f(mu_{i-1} W_{i-1})||^2``.
        """
        if self._mu is None:
            return 0.0

        energy = 0.0
        W = [layer.weight.data for layer in self._layers]

        for i in range(1, len(self._mu)):
            pred = _apply_activation(self._mu[i - 1] @ W[i - 1].T, self._activation)
            if self._layers[i - 1].bias is not None:
                pred = pred + self._layers[i - 1].bias.data
            energy += 0.5 * (self._mu[i] - pred).pow(2).sum().item()

        return energy

    def backward(
        self,
        x: Tensor,
        free_mu: list[Tensor],
        nudged_mu: list[Tensor],
    ) -> dict[str, Tensor]:
        """Compute weight updates from free and nudged phases.

        Contrastive update: each weight W_{l-1} is driven by the prediction
        error at layer l (``mu_l - f(mu_{l-1} W_{l-1} + b)``); the free and
        nudged errors are contrasted. Matches the inference primitive's
        convention (predict layer i from layer i-1 via ``W[i-1]``).

        Returns:
            Weight deltas for all layers
        """
        weight_deltas: dict[str, Tensor] = {}
        L = len(self._layers)

        for l in range(1, L + 1):
            # Error at layer l predicts mu_l from mu_{l-1} via W[l-1].
            free_pre = free_mu[l - 1]
            free_pred = _apply_activation(
                free_pre @ self._layers[l - 1].weight.data.T, self._activation
            )
            if self._layers[l - 1].bias is not None:
                free_pred = free_pred + self._layers[l - 1].bias.data
            free_error = free_mu[l] - free_pred

            nudged_pre = nudged_mu[l - 1]
            nudged_pred = _apply_activation(
                nudged_pre @ self._layers[l - 1].weight.data.T, self._activation
            )
            if self._layers[l - 1].bias is not None:
                nudged_pred = nudged_pred + self._layers[l - 1].bias.data
            nudged_error = nudged_mu[l] - nudged_pred

            # Weight delta for W[l-1] [D_l, D_{l-1}]
            free_grad = batched_outer_product(free_pre, free_error)
            nudged_grad = batched_outer_product(nudged_pre, nudged_error)
            delta = self._eta_weight * contrastive_delta(
                free_grad, nudged_grad, beta=1.0
            )
            weight_deltas[f"layers.{l - 1}.weight"] = delta

            if self._layers[l - 1].bias is not None:
                weight_deltas[f"layers.{l - 1}.bias"] = self._eta_weight * (
                    nudged_error.mean(dim=0) - free_error.mean(dim=0)
                )

        return weight_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float = 1.0) -> None:
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
        mu_mb = 0.0
        if self._mu is not None:
            mu_mb = sum(m.numel() for m in self._mu) * 4 / 1e6
        return {
            "params_mb": total_params * 4 / 1e6,
            "states_mb": mu_mb,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        """Return the most recent settle loop's telemetry, if any."""
        return self._last_settle_telemetry


def _apply_activation(x: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return torch.relu(x)
    if activation == "silu":
        return torch.nn.functional.silu(x)
    if activation == "tanh":
        return torch.tanh(x)
    if activation == "gelu":
        return torch.nn.functional.gelu(x)
    return torch.tanh(x)


# Triton kernels for fused PC operations
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _pc_prediction_kernel(
        mu_ptr,
        W_ptr,
        b_ptr,
        pred_ptr,
        B,
        D_in,
        D_out,
        activation_type,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute prediction: act(mu @ W.T + b)"""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_b = offs_b < B
        mask_d = offs_d < D_out

        acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        for k in range(0, D_in, BLOCK_D):
            offs_k = k + tl.arange(0, BLOCK_D)
            mask_k = offs_k < D_in

            mu_tile = tl.load(
                mu_ptr + offs_b[:, None] * D_in + offs_k[None, :],
                mask=mask_b[:, None] & mask_k[None, :],
                other=0.0,
            )
            W_tile = tl.load(
                W_ptr + offs_d[:, None] * D_in + offs_k[None, :],
                mask=mask_d[:, None] & mask_k[None, :],
                other=0.0,
            )
            acc += tl.dot(mu_tile, tl.trans(W_tile))

        # Add bias
        if b_ptr is not None:
            bias = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0)
            acc += bias[None, :]

        # Apply activation
        if activation_type == 0:  # ReLU
            pred = tl.maximum(acc, 0.0)
        elif activation_type == 1:  # SiLU
            sig = libdevice.sigmoid(acc)
            pred = acc * sig
        elif activation_type == 2:  # Tanh
            pred = libdevice.tanh(acc)
        elif activation_type == 3:  # GELU
            cdf = 0.5 * (1.0 + libdevice.erf(acc * 0.7071067811865475))
            pred = acc * cdf
        else:
            pred = tl.maximum(acc, 0.0)

        tl.store(
            pred_ptr + offs_b[:, None] * D_out + offs_d[None, :],
            pred,
            mask=mask_b[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _pc_error_update_kernel(
        mu_ptr,
        pred_ptr,
        mu_new_ptr,
        eta_infer,
        activation_type,
        B,
        D,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """PCN inference step: mu = mu - eta * (mu - pred) * act_deriv(mu)"""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_b = offs_b < B
        mask_d = offs_d < D

        mu = tl.load(
            mu_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        pred = tl.load(
            pred_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )

        error = mu - pred

        # Activation derivative
        if activation_type == 0:  # ReLU
            deriv = (mu > 0).to(tl.float32)
        elif activation_type == 1:  # SiLU
            sig = libdevice.sigmoid(mu)
            deriv = sig * (1.0 + mu * (1.0 - sig))
        elif activation_type == 2:  # Tanh
            deriv = 1.0 - mu * mu
        elif activation_type == 3:  # GELU
            cdf = 0.5 * (1.0 + libdevice.erf(mu * 0.7071067811865475))
            pdf = libdevice.exp(-mu * mu * 0.5) * 0.3989422804014327
            deriv = cdf + mu * pdf
        else:
            deriv = (mu > 0).to(tl.float32)

        mu_new = mu - eta_infer * error * deriv

        tl.store(
            mu_new_ptr + offs_b[:, None] * D + offs_d[None, :],
            mu_new,
            mask=mask_b[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _pc_contrastive_update_kernel(
        pre_free_ptr,
        post_free_ptr,
        pre_nudged_ptr,
        post_nudged_ptr,
        delta_ptr,
        B,
        D_in,
        D_out,
        beta,
        lr,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Contrastive weight update for PC."""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc_free = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)
        acc_nudged = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            pre_f = tl.load(
                pre_free_ptr + b * D_in + offs_in[None, :],
                mask=mask_in[None, :],
                other=0.0,
            )
            post_f = tl.load(
                post_free_ptr + b * D_out + offs_out[:, None],
                mask=mask_out[:, None],
                other=0.0,
            )
            acc_free += tl.dot(tl.trans(post_f), pre_f)

            pre_n = tl.load(
                pre_nudged_ptr + b * D_in + offs_in[None, :],
                mask=mask_in[None, :],
                other=0.0,
            )
            post_n = tl.load(
                post_nudged_ptr + b * D_out + offs_out[:, None],
                mask=mask_out[:, None],
                other=0.0,
            )
            acc_nudged += tl.dot(tl.trans(post_n), pre_n)

        acc_free = acc_free / B
        acc_nudged = acc_nudged / B

        delta = lr * (acc_nudged - acc_free) / beta

        tl.store(
            delta_ptr + offs_out[:, None] * D_in + offs_in[None, :],
            delta,
            mask=mask_out[:, None] & mask_in[None, :],
        )

    HAS_TRITON_PC = True
except ImportError:
    HAS_TRITON_PC = False
for hw in HardwareTarget:
    KernelRegistry.register(AlgorithmFamily.PC, hw, PCKernelBackend)


__all__ = ["PCKernelBackend"]
