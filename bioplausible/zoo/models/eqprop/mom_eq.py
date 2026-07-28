"""Equilibrium Propagation model variants."""

import torch
from torch import nn

from ...base import BioModel, ModelConfig, register_model


@register_model(
    "momentum_equilibrium",
    family="eqprop",
    tags=["eqprop", "momentum"],
)
class MomentumEquilibrium(BioModel):
    """EqProp with momentum in settling dynamics."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
                if hasattr(self, "hidden_dim")
                else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.momentum = 0.5

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        activations = [x]
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        h = self.layers[-1](h)
        activations.append(h)

        velocities = [torch.zeros_like(a) for a in activations]

        for _ in range(self.config.equilibrium_steps):
            new_acts = [activations[0]]
            h = activations[0]

            for i, layer in enumerate(self.layers[:-1]):
                target = self.activation(layer(h))
                delta = target - activations[i + 1]
                velocities[i + 1] = self.momentum * velocities[i + 1] + 0.5 * delta
                h = activations[i + 1] + velocities[i + 1]
                new_acts.append(h)

            h = self.layers[-1](h)
            new_acts.append(h)
            activations = new_acts

        return activations[-1]

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
        return cls(config=config).to(device)
