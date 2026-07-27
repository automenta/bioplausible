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
    "modern_conv_eqprop",
    family="eqprop",
    tags=["eqprop", "conv"],
)
class ModernConvEqProp(EqPropModel):
    """
    Multi-stage ConvEqProp with equilibrium settling.

    Architecture:
        Input: 3x32x32 (CIFAR-10)
        Stage 1: Conv 3->64, no pooling (32x32)
        Stage 2: Conv 64->128, stride 2 (16x16)
        Stage 3: Conv 128->256, stride 2 (8x8)
        Equilibrium: Recurrent conv at 256 channels
        Output: Global pool -> Linear(256, 10)
    """

    def __init__(
        self,
        eq_steps: int = 15,
        gamma: float = 0.5,
        hidden_channels: int = 64,
        use_spectral_norm: bool = True,
        gradient_method: str = "bptt",
        input_dim: int = 0,
        output_dim: int = 10,
        **kwargs,
    ):
        self.gamma = gamma
        self.base_hidden_channels = hidden_channels

        self.input_channels = 3
        self.input_dims = (32, 32)
        flat_input_dim = 0

        if "input_channels" in kwargs:
            self.input_channels = kwargs["input_channels"]

        if isinstance(input_dim, tuple):
            flat_input_dim = math.prod(input_dim)
            if len(input_dim) == 3:
                self.input_channels = input_dim[0]
                self.input_dims = (input_dim[1], input_dim[2])
            elif len(input_dim) == 1:
                self.input_channels = input_dim[0]
        elif isinstance(input_dim, int) and input_dim > 0:
            flat_input_dim = input_dim

        if isinstance(input_dim, int) and input_dim == 64:
            self.input_channels = 1
            self.input_dims = (8, 8)
        if isinstance(input_dim, int) and input_dim == 784:
            self.input_channels = 1
            self.input_dims = (28, 28)

        self.output_dim_val = output_dim

        super().__init__(
            input_dim=flat_input_dim,
            hidden_dim=hidden_channels * 4,
            output_dim=self.output_dim_val,
            max_steps=eq_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        hidden_channels = self.base_hidden_channels

        self.stage1 = nn.Sequential(
            spectral_conv2d(
                self.input_channels,
                hidden_channels,
                3,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.Tanh(),
        )

        in_dim_0 = getattr(self, "input_dims", (32, 32))[0]

        self.stage2 = nn.Sequential(
            spectral_conv2d(
                hidden_channels,
                hidden_channels * 2,
                3,
                stride=2 if in_dim_0 >= 16 else 1,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.Tanh(),
        )

        self.stage3 = nn.Sequential(
            spectral_conv2d(
                hidden_channels * 2,
                hidden_channels * 4,
                3,
                stride=2 if in_dim_0 >= 32 else 1,
                padding=1,
                use_sn=self.use_spectral_norm,
            ),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.Tanh(),
        )

        self.eq_conv = spectral_conv2d(
            hidden_channels * 4,
            hidden_channels * 4,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.eq_norm = nn.GroupNorm(8, hidden_channels * 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels * 4, self.output_dim_val)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "parametrizations"):
                    weight = m.parametrizations.weight.original
                else:
                    weight = m.weight
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="tanh")
                weight.data.mul_(0.5)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        with torch.no_grad():
            h_trans = self._transform_input(x[:1])
            H_out, W_out = h_trans.shape[2], h_trans.shape[3]

        return torch.zeros(
            B, self.hidden_dim, H_out, W_out, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            B = x.size(0)
            area = x.size(1) // self.input_channels
            S = int(math.sqrt(area))
            x = x.view(B, self.input_channels, S, S)

        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        return h

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_norm = self.eq_norm(h)

        if TritonEqPropOps.is_available() and h.is_cuda:
            pre_act = self.eq_conv(h_norm) + x_transformed
            return TritonEqPropOps.step(h, pre_act, alpha=self.gamma)

        h_next = torch.tanh(self.eq_conv(h_norm) + x_transformed)
        return torch.lerp(h, h_next, self.gamma)

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        features = self.pool(h).flatten(1)
        return self.fc(features)

    def get_hebbian_pairs(self, h, x):
        if not hasattr(self, "feedforward_net"):
            self.feedforward_net = nn.Sequential(self.stage1, self.stage2, self.stage3)

        h_norm = self.eq_norm(h)

        return [(self.eq_conv, h_norm, h), (self.feedforward_net, x, h)]

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
            eq_steps=30,
            hidden_channels=hidden_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs,
        ).to(device)


class SimpleConvEqProp(EqPropModel):
    """
    Simplified single-stage ConvEqProp for comparison.
    Refactored to use EqPropModel.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        eq_steps: int = 20,
        gamma: float = 0.5,
        use_spectral_norm: bool = True,
        gradient_method: str = "bptt",
        input_channels: int = 3,
        output_dim: int = 10,
        pool_output: bool = True,
    ):
        self.hidden_channels = hidden_channels
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm
        self.input_channels_count = input_channels
        self.output_dim_val = output_dim
        self.pool_output = pool_output

        super().__init__(
            input_dim=0,
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            max_steps=eq_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        self.embed = spectral_conv2d(
            self.input_channels_count,
            self.hidden_channels,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        self.W_rec = spectral_conv2d(
            self.hidden_channels,
            self.hidden_channels,
            3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.norm = nn.GroupNorm(8, self.hidden_channels)

        if self.pool_output:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_channels, self.output_dim_val),
            )
        else:
            self.head = spectral_conv2d(
                self.hidden_channels,
                self.output_dim_val,
                kernel_size=1,
                use_sn=self.use_spectral_norm,
            )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_norm = self.norm(h)

        if TritonEqPropOps.is_available() and h.is_cuda:
            pre_act = self.W_rec(h_norm) + x_transformed
            return TritonEqPropOps.step(h, pre_act, alpha=self.gamma)

        h_next = torch.tanh(self.W_rec(h_norm) + x_transformed)
        return torch.lerp(h, h_next, self.gamma)

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def get_hebbian_pairs(self, h, x):
        h_norm = self.norm(h)
        return [(self.W_rec, h_norm, h), (self.embed, x, h)]


