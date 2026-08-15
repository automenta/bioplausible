"""Target Propagation Kernel Backend.

Inverse network forward + target propagation kernels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class TPKernelBackend:
    """Target Propagation kernel backend.

    Implements Difference Target Propagation (DTP) with inverse networks.
    """

    name = AlgorithmFamily.TP
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = False
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.LAYERWISE

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._forward_layers: list[torch.nn.Linear] = []
        self._inverse_layers: list[torch.nn.Linear] = []
        self._target_lr: float = 0.1
        self._inverse_lr: float = 0.01
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._activation: torch.nn.Module = torch.nn.Tanh()

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
        self._target_lr = extra.get("target_lr", 0.1)
        self._inverse_lr = extra.get("inverse_net_lr", 0.01)
        self._activation = _get_activation(extra.get("activation", "tanh"))

    def set_model_ref(
        self,
        forward_layers: list[torch.nn.Linear],
        inverse_layers: list[torch.nn.Linear],
        activation: torch.nn.Module | None = None,
    ) -> None:
        """Set reference to forward and inverse network layers."""
        self._forward_layers = forward_layers
        self._inverse_layers = inverse_layers
        if activation is not None:
            self._activation = activation

    def forward_forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass through forward network."""
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        activations: list[Tensor] = [x]
        h = x

        for i, layer in enumerate(self._forward_layers):
            h = layer(h)
            if i < len(self._forward_layers) - 1:
                h = self._activation(h)
            activations.append(h)

        return activations[-1], activations

    def forward_inverse(
        self, target: Tensor, layer_idx: int
    ) -> tuple[Tensor, list[Tensor]]:
        """Backward pass through inverse network from target.

        Args:
            target: Target at layer layer_idx+1
            layer_idx: Index of inverse layer to start from

        Returns:
            (computed_target, activations) where computed_target is target for layer_idx
        """
        target = target.to(device=self._device, dtype=self._dtype)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)

        activations: list[Tensor] = [target]
        h = target

        # Go backwards through inverse layers
        for i in range(len(self._inverse_layers) - 1, layer_idx - 1, -1):
            h = self._inverse_layers[i](h)
            if i > 0:
                h = self._activation(h)
            activations.append(h)

        return activations[-1], list(reversed(activations))

    def compute_targets(
        self,
        forward_activations: list[Tensor],
        output_target: Tensor,
    ) -> list[Tensor]:
        """Compute layer-wise targets via inverse network.

        Args:
            forward_activations: [x, h1, h2, ..., output] from forward pass
            output_target: Target at output layer (e.g., one-hot labels)

        Returns:
            List of targets for each layer [target_h1, target_h2, ..., target_output]
        """
        targets: list[Tensor] = [None] * len(self._forward_layers)  # type: ignore
        targets[-1] = output_target

        # Backpropagate target through inverse network
        current_target = output_target
        for i in range(len(self._inverse_layers) - 1, -1, -1):
            # Target for layer i = inverse_layer_i(target_{i+1})
            inverse_layer = self._inverse_layers[i]
            with torch.no_grad():
                current_target = inverse_layer(current_target)
                if i > 0:
                    current_target = self._activation(current_target)
            targets[i] = current_target

        return targets  # type: ignore

    def backward(
        self,
        forward_activations: list[Tensor],
        targets: list[Tensor],
    ) -> dict[str, Tensor]:
        """Compute weight updates for forward and inverse networks.

        Forward: Delta W_f = lr * (target - activation) @ prev_activation.T
        Inverse: Delta W_g = lr * (activation - target) @ next_target.T

        Returns:
            Dict with forward and inverse weight updates
        """
        updates: dict[str, Tensor] = {}

        # Forward network updates
        for i in range(len(self._forward_layers)):
            pre = forward_activations[i]
            post = forward_activations[i + 1]
            target = targets[i]

            # Difference target: target - post
            diff = target - post
            delta = self._target_lr * batched_outer_product(pre, diff)
            updates[f"forward.{i}.weight"] = delta

            if self._forward_layers[i].bias is not None:
                updates[f"forward.{i}.bias"] = self._target_lr * diff.mean(dim=0)

        # Inverse network updates
        # Inverse tries to reconstruct pre from post
        for i in range(len(self._inverse_layers)):
            # Inverse input is target at layer i+1
            inv_input = targets[i + 1] if i + 1 < len(targets) else targets[-1]
            # Inverse target is forward activation at layer i
            inv_target = forward_activations[i]

            with torch.no_grad():
                inv_output = self._inverse_layers[i](inv_input)
                if i < len(self._inverse_layers) - 1:
                    inv_output = self._activation(inv_output)

            diff = inv_target - inv_output
            delta = self._inverse_lr * batched_outer_product(inv_input, diff)
            updates[f"inverse.{i}.weight"] = delta

            if self._inverse_layers[i].bias is not None:
                updates[f"inverse.{i}.bias"] = self._inverse_lr * diff.mean(dim=0)

        return updates

    def update_weights(self, gradients: dict[str, Tensor], lr: float = 1.0) -> None:
        """Apply weight updates (lr already baked in)."""
        with torch.no_grad():
            for name, grad in gradients.items():
                parts = name.split(".")
                net_type = parts[0]  # "forward" or "inverse"
                layer_idx = int(parts[1])
                param_type = parts[2]  # "weight" or "bias"

                if net_type == "forward":
                    layer = self._forward_layers[layer_idx]
                else:
                    layer = self._inverse_layers[layer_idx]

                if param_type == "weight":
                    layer.weight.add_(grad)
                elif param_type == "bias" and layer.bias is not None:
                    layer.bias.add_(grad)

    def get_memory_stats(self) -> dict[str, float]:
        fwd_params = sum(
            p.numel() for layer in self._forward_layers for p in layer.parameters()
        )
        inv_params = sum(
            p.numel() for layer in self._inverse_layers for p in layer.parameters()
        )
        return {
            "forward_params_mb": fwd_params * 4 / 1e6,
            "inverse_params_mb": inv_params * 4 / 1e6,
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
    return activations.get(name.lower(), torch.nn.Tanh())


# Register backend
KernelRegistry.register(AlgorithmFamily.TP, HardwareTarget.CPU, TPKernelBackend)
KernelRegistry.register(AlgorithmFamily.TP, HardwareTarget.CUDA, TPKernelBackend)
KernelRegistry.register(AlgorithmFamily.TP, HardwareTarget.TRITON, TPKernelBackend)


__all__ = ["TPKernelBackend"]
