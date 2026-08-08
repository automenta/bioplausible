"""Scratch prototype of the unified energy-contrastive EqProp engine (v2).

Transient file: confirms the *base-class* contrastive ``EqPropModel.train_step``
(the proven GraphEqProp/vision design) learns on MNIST and fits a shallow-probe
epoch budget. Deleted once the real fundamental models are rewired onto it.
"""

from __future__ import annotations

import torch
from torch import nn

from bioplausible.core.config import ModelConfig
from bioplausible.core.registry import register_model
from bioplausible.zoo.models.base import EqPropModel


@register_model(
    "_ep_proto",
    family="eqprop",
    tags=["eqprop", "proto"],
)
class EnergyEqPropProto(EqPropModel):
    """Single-hidden recurrent EqProp using the base contrastive train_step."""

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_spectral_norm:
            self.W_rec = nn.utils.parametrizations.spectral_norm(self.W_rec)

    def transition_modules(self) -> list[nn.Module]:
        return [self.W_in, self.W_rec, self.W_out]

    def forward_step(self, h: torch.Tensor, x_transformed: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x_transformed + self.W_rec(h))

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.size(0), self.hidden_dim), device=x.device, dtype=x.dtype)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.W_in(x)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def get_hebbian_pairs(self, h, x):
        x_in = x.reshape(x.size(0), -1) if x.dim() > 2 else x
        return [(self.W_in, x_in, h), (self.W_rec, h, h)]

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="proto",
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[hidden_dim],
                max_steps=int(kwargs.get("max_steps", 20)),
                learning_rate=float(kwargs.get("learning_rate", 1e-3)),
                beta=float(kwargs.get("beta", 0.3)),
                use_spectral_norm=bool(kwargs.get("use_spectral_norm", True)),
            )
        ).to(device)
