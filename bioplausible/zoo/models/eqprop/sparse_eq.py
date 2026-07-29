"""Equilibrium Propagation model variants."""

import torch
from torch import nn

from ...base import (
    BioModel,
    ModelConfig,
    _build_model_config,
    register_model,
    resolve_hidden_dims,
)


@register_model(
    "sparse_equilibrium",
    family="eqprop",
    tags=["eqprop", "sparse"],
)
class SparseEquilibrium(BioModel):
    """EqProp with sparse (Top-K) updates."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.sparsity = 0.5
        self.criterion = nn.CrossEntropyLoss()

    def sparse_activation(self, x: torch.Tensor) -> torch.Tensor:
        k = int(x.size(1) * self.sparsity)
        top_vals, _ = torch.topk(torch.abs(x), k, dim=1)
        threshold = top_vals[:, -1].unsqueeze(1)
        mask = (torch.abs(x) >= threshold).float()
        return x * mask

    def forward(self, x: torch.Tensor, steps: int = 20, **kwargs) -> torch.Tensor:
        activations = [x]
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        h = self.layers[-1](h)
        activations.append(h)

        for _ in range(steps):
            new_acts = [activations[0]]
            h = activations[0]

            for i, layer in enumerate(self.layers[:-1]):
                pre_activ = layer(h)
                h = self.activation(pre_activ)
                h = self.sparse_activation(h)
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
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)
