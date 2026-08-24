"""Target Propagation Kernel Backend.

Inverse network forward + target propagation kernels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from computronium.acceleration.contrastive_primitives import (
    batched_outer_product,
)
from computronium.acceleration.kernel_backend import (
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

        # Inverse layers are ordered output->input: apply them forward from the
        # given layer index to the start.
        for i in range(layer_idx, len(self._inverse_layers)):
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
            List of targets, one per forward layer, where ``targets[-1]`` is the
            output target and ``targets[i]`` is the target for the post-activation
            of forward layer ``i``.
        """
        L = len(self._forward_layers)
        targets: list[Tensor] = [None] * L  # type: ignore
        targets[-1] = output_target

        # Propagate target backward through inverse layers (ordered output->input).
        # inverse_layers[k] maps a target at layer k+1 to a target at layer k.
        current_target = output_target
        for k in range(L - 1):
            inverse_layer = self._inverse_layers[k]
            with torch.no_grad():
                current_target = inverse_layer(current_target)
                if k > 0:
                    current_target = self._activation(current_target)
            targets[L - 2 - k] = current_target

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
        # inverse_layers[i] reconstructs forward activation i+1 from the target
        # at layer i+1.
        for i in range(len(self._inverse_layers)):
            # Inverse input is target at layer i+1
            inv_input = targets[i + 1] if i + 1 < len(targets) else targets[-1]
            # Inverse target is the forward activation at layer i+1
            inv_target = forward_activations[i + 1]

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

    def kernel_train_step(
        self,
        model: torch.nn.Module,
        config: KernelConfig | None,
        x: Tensor,
        y: Tensor,
        optimizer: object | None = None,
    ) -> dict[str, object] | None:
        """Bespoke Difference-Target-Propagation step (REFACTOR7 bespoke seam).

        DTP's dynamics (per-layer output-layer autograd update, target
        propagation through the trained inverse nets, then forward/inverse
        net fitting) don't fit the uniform ``forward → backward(acts, error)
        → update_weights`` contract, so the dispatch seam delegates here when
        present. It mirrors the reference ``DifferenceTargetProp.train_step``
        exactly: the output layer is updated first via its own Adam, a
        difference target ``h - target_lr * dL/dh`` is built from the output
        layer, then propagated backward through the inverse nets while each
        forward (and inverse, for cycle consistency) net is fitted with its
        own Adam.

        Args:
            model: The bound ``DifferenceTargetProp`` model (``layers``
                ModuleList of ``DTPLayer`` with ``forward_net``/``inverse_net``
                Sequentials + per-net ``opt_f``/``opt_g``, plus ``out_layer``,
                ``out_opt``, ``criterion``, ``target_lr``).
            config: KernelConfig (kept for seam parity with other bespoke
                backends; the reference dynamics read the model).
            x: Input batch.
            y: Target labels.
            optimizer: Ignored — DTP owns per-layer optimizers on the model.

        Returns:
            ``{"loss", "accuracy", "logits"}`` or ``None`` when the model does
            not expose the DTP surface (caller falls through to ``train_step``).
        """
        layers = getattr(model, "layers", None)
        out_layer = getattr(model, "out_layer", None)
        if not layers or out_layer is None:
            return None

        self._forward_layers = [layer.forward_net[0] for layer in layers]
        self._inverse_layers = [layer.inverse_net[0] for layer in layers]

        out_opt = getattr(model, "out_opt", None)
        criterion = getattr(model, "criterion", None) or torch.nn.CrossEntropyLoss()
        target_lr = float(getattr(model, "target_lr", self._target_lr))

        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(device=self._device, dtype=self._dtype)
        y = y.to(device=self._device)

        hs: list[Tensor] = [x]
        h = x
        for layer in layers:
            h = layer.forward_net(h)
            hs.append(h)
        out = out_layer(h)

        loss = criterion(out, y)

        if out_opt is not None:
            out_opt.zero_grad()
            loss.backward()
            out_opt.step()

        t = h.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out_t = out_layer(t)
            loss_t = criterion(out_t, y)
            grad_t = torch.autograd.grad(loss_t, t)[0]
        with torch.no_grad():
            t_target = h - target_lr * grad_t

        targets: list[Tensor] = [t_target]

        for i in reversed(range(len(layers))):
            layer = layers[i]
            if i > 0:
                h_prev = hs[i].detach()
                h_curr = hs[i + 1].detach()
                t_curr = targets[-1]
                with torch.no_grad():
                    t_prev = (
                        h_prev - layer.inverse_net(h_curr) + layer.inverse_net(t_curr)
                    )
                    targets.append(t_prev)

            t_curr = targets[-len(targets)]
            h_prev_det = hs[i].detach()
            layer.opt_f.zero_grad()
            pred_h = layer.forward_net(h_prev_det)
            loss_f = torch.nn.functional.mse_loss(pred_h, t_curr)
            loss_f.backward()

            if i > 0:
                layer.opt_g.zero_grad()
                inv_out = layer.inverse_net(pred_h.detach())
                loss_g = torch.nn.functional.mse_loss(inv_out, h_prev_det)
                loss_g.backward()
                layer.opt_g.step()

            layer.opt_f.step()

        accuracy = (out.argmax(dim=1) == y).float().mean().item()
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "logits": out.detach(),
        }

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


# Register backend for all HardwareTargets
for hw in HardwareTarget:
    KernelRegistry.register(AlgorithmFamily.TP, hw, TPKernelBackend)


__all__ = ["TPKernelBackend"]
