"""Equilibrium Propagation model variants."""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.acceleration.kernels import HAS_CUPY, EqPropKernel
from bioplausible.acceleration.triton_kernels import TritonEqPropOps
from bioplausible.core.registry import Domain, LocalityLevel

from ....acceleration import compile_settling_loop
from ...base import BioModel, ModelConfig, register_model
from ...utils import spectral_conv2d, spectral_linear
from ..base import EqPropModel



@register_model(
    "eqprop_mlp",
    domains=[Domain.VISION, Domain.LM, Domain.RL, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    tags=["eqprop", "looped_mlp", "equilibrium"],
)
class LoopedMLP(EqPropModel):
    """
    A recurrent MLP that iterates to a fixed-point equilibrium.

    The key insight: By constraining Lipschitz constant L < 1 via spectral norm,
    the network is guaranteed to converge to a unique fixed point.

    Architecture:
        h_{t+1} = tanh(W_in @ x + W_rec @ h_t)
        output = W_out @ h*  (where h* is the fixed point)

    This model can be trained using:
    1. BPTT (Backpropagation Through Time): With EqPropTrainer(use_kernel=False)
    2. EqProp (Equilibrium Propagation): Using EqPropTrainer(use_kernel=True).
       Note: For EqProp kernel mode, the weights are managed by the kernel (NumPy/CuPy),
       not this PyTorch module. This module is primarily for BPTT or inference/visualization.

    Example:
        >>> model = LoopedMLP(784, 256, 10, use_spectral_norm=True)
        >>> x = torch.randn(32, 784)
        >>> output = model(x, steps=30)  # [32, 10]
        >>> L = model.compute_lipschitz()  # Should be < 1.0
    """

    def __init__(
        self,
        input_dim: int | tuple,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        backend: str = "pytorch",
        num_layers: int = 2,
    ) -> None:
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

        if backend == "auto":
            backend = "kernel" if torch.cuda.is_available() and HAS_CUPY else "pytorch"

        self.backend = backend
        self._engine = None

        if self.backend == "kernel":
            use_gpu = HAS_CUPY and torch.cuda.is_available()
            self._engine = EqPropKernel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                max_steps=max_steps,
                use_spectral_norm=use_spectral_norm,
                use_gpu=use_gpu,
                architecture="rnn",
            )

        self._init_weights()

    def __repr__(self) -> str:
        backend_str = f", backend={self.backend}" if self.backend != "pytorch" else ""
        return (
            f"LoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm}{backend_str})"
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=True,
            max_steps=20,
        ).to(device)

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

    def _init_weights(self) -> None:
        for layer in [self.W_in, self.W_rec, self.W_out]:
            self._initialize_single_layer(layer)

    def _initialize_single_layer(self, layer: nn.Module) -> None:
        actual_layer = self._get_actual_layer(layer)
        if hasattr(actual_layer, "weight"):
            nn.init.xavier_uniform_(actual_layer.weight, gain=0.5)
            if actual_layer.bias is not None:
                nn.init.zeros_(actual_layer.bias)

    def _get_actual_layer(self, layer: nn.Module) -> nn.Module:
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            return layer.parametrizations.weight.original
        return layer

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.float()

        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_in)
            b = self.W_in.bias
            return torch.nn.functional.linear(x, w, b)
        return self.W_in(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        if TritonEqPropOps.is_available():
            pre_act = x_transformed + self.W_rec(h)
            return TritonEqPropOps.step(h, pre_act, alpha=1.0)

        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_rec)
            b = self.W_rec.bias
            rec = torch.nn.functional.linear(h, w, b)
            return torch.tanh(x_transformed + rec)

        return torch.tanh(x_transformed + self.W_rec(h))

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        if not self.training:
            w = self._get_spectral_normalized_weight(self.W_out)
            b = self.W_out.bias
            return torch.nn.functional.linear(h, w, b)
        return self.W_out(h)

    def get_hebbian_pairs(self, h, x):
        return [(self.W_in, x, h), (self.W_rec, h, h)]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = y

            if x_np.ndim > 2:
                x_np = x_np.reshape(x_np.shape[0], -1)

            metrics = self._engine.train_step(x_np, y_np)
            return metrics

        return super().train_step(x, y)

    def forward(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, dict[str, Any]]
    ):
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if x_np.ndim > 2:
                x_np = x_np.reshape(x_np.shape[0], -1)

            h_star, _, _ = self._engine.solve_equilibrium(x_np)
            logits_np = self._engine.compute_output(h_star)

            logits = torch.from_numpy(logits_np).to(x.device)

            if return_trajectory or return_dynamics:
                return logits, {} if return_dynamics else []

            return logits

        return super().forward(x, steps, return_trajectory, return_dynamics)


@register_model(
    "backprop_mlp",
    family="backprop",
    tags=["backprop", "mlp"],
)
class BackpropMLP(nn.Module):
    """Standard feedforward MLP for comparison (no equilibrium dynamics)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        layers = []
        if input_dim is None:
            input_dim = 64

        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        if num_layers <= 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.float()
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        if x.size(1) != self.net[0].in_features:
            raise ValueError(
                f"Input feature dimension mismatch. "
                f"Expected {self.net[0].in_features} but got {x.size(1)}."
            )

        return self.net(x)

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        ).to(device)


