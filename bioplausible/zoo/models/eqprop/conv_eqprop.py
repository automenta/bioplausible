"""Equilibrium Propagation model variants."""

import torch
from torch import nn

from bioplausible.acceleration.triton_kernels import TritonEqPropOps
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import register_model

from ....acceleration import compile_settling_loop
from ...utils import spectral_conv2d
from ..base import EqPropModel

__all__ = [
    "ConvEqProp",
]


@register_model(
    "conv_eqprop",
    family="eqprop",
    tags=["eqprop", "conv", status_tag("broken")],
)
class ConvEqProp(EqPropModel):
    """
    Convolutional Equilibrium Propagation Model.

    Status: ``broken`` — ``num_layers`` is a phantom knob on this model
    (silently dropped at construction; the architecture is a single conv loop
    regardless of the requested depth). See ``docs/phantom_knob_audit.md``.
    Use ``modern_conv_eqprop`` for a depth-aware conv EqProp once it clears
    its own audit, or ``eqprop`` / ``backprop_mlp`` for conv-unaware depth.

    Uses ResNet-like loop structure with spectral normalization.
    Suitable for image classification tasks (MNIST, CIFAR-10).

    Example:
        >>> model = ConvEqProp(1, 32, 10)  # MNIST
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x, steps=25)  # [32, 10]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dim: int,
        gamma: float = 0.5,
        use_spectral_norm: bool = True,
        max_steps: int = 25,
        gradient_method: str = "equilibrium",
    ) -> None:
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.gamma = gamma
        self.use_spectral_norm = use_spectral_norm

        super().__init__(
            input_dim=0,
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )
        self.input_format = "spatial"

        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def _build_layers(self):
        self.embed = spectral_conv2d(
            self.input_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        self.W1 = spectral_conv2d(
            self.hidden_channels,
            self.hidden_channels * 2,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )
        self.W2 = spectral_conv2d(
            self.hidden_channels * 2,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            use_sn=self.use_spectral_norm,
        )

        self.norm = nn.GroupNorm(8, self.hidden_channels)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, self.output_dim),
        )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        return torch.zeros(
            B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def transition_modules(self) -> list[nn.Module]:
        return [self.W1, self.W2]

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        h_norm = self.norm(h)

        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)

        h_target = ffn_out + x_transformed

        if TritonEqPropOps.is_available() and h.is_cuda:
            return TritonEqPropOps.step_linear(h, h_target, self.gamma)
        else:
            h_next = torch.lerp(h, h_target, self.gamma)
            return h_next

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

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

        if isinstance(input_dim, tuple):
            input_channels = input_dim[0]
        # Try to infer from common sizes
        elif input_dim == 784:  # 28*28
            input_channels = 1
        elif input_dim == 3072:  # 32*32*3
            input_channels = 3
        elif input_dim == 64:  # 8*8
            input_channels = 1
        else:
            input_channels = 1
        return cls(
            input_channels=input_channels,
            hidden_channels=hidden_dim,
            output_dim=output_dim,
            max_steps=kwargs.get("steps", kwargs.get("max_steps", 25)),
            gradient_method=kwargs.get("gradient_method", "equilibrium"),
            gamma=kwargs.get("gamma", 0.5),
        ).to(device)
