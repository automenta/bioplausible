"""Predictive Coding Kernel Backend.

Graph-parallel inference + PCN loss kernels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
    predictive_coding_inference_step,
)
from bioplausible.acceleration.kernel_backend import (
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
            pred = _apply_activation(
                self._mu[i - 1] @ W[i - 1].T, self._activation
            )
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
            weight_deltas[f"layers.{l-1}.weight"] = delta

            if self._layers[l - 1].bias is not None:
                weight_deltas[f"layers.{l-1}.bias"] = self._eta_weight * (
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


# Register backend
KernelRegistry.register(AlgorithmFamily.PC, HardwareTarget.CPU, PCKernelBackend)
KernelRegistry.register(AlgorithmFamily.PC, HardwareTarget.CUDA, PCKernelBackend)
KernelRegistry.register(AlgorithmFamily.PC, HardwareTarget.TRITON, PCKernelBackend)


__all__ = ["PCKernelBackend"]
