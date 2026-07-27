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
from .standard_eqprop import StandardEqProp



@register_model(
    "finite_nudge_ep",
    family="eqprop",
    tags=["eqprop", "finite-nudge"],
)
class FiniteNudgeEP(StandardEqProp):
    """
    Finite-Nudge EqProp.
    Operates with large beta values (e.g. beta=1.0) where the infinitesimal
    approximation of the gradient is replaced by a finite difference
    that optimizes a global energy bound.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        elif self.config and self.config.extra and "beta" in self.config.extra:
            self.beta = self.config.extra["beta"]

        if self.beta < 0.5:
            self.beta = 1.0

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        metrics = super().train_step(x, y)

        return metrics

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
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            extra=kwargs,
        )

        if "equilibrium_steps" in kwargs:
            config.equilibrium_steps = kwargs["equilibrium_steps"]
            config.max_steps = kwargs["equilibrium_steps"]
        if "beta" in kwargs:
            config.beta = kwargs["beta"]

        return cls(config=config).to(device)


