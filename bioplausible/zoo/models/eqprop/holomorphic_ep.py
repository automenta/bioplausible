"""Equilibrium Propagation model variants."""

import torch
from torch import nn

from ....zoo._settling import settle_activations_list
from ...base import (
    BioModel,
    ModelConfig,
    _build_model_config,
    register_model,
    resolve_hidden_dims,
)
from ._contrastive import _contrastive_step


@register_model(
    "holomorphic_ep",
    family="eqprop",
    tags=["eqprop", "holomorphic"],
)
class HolomorphicEP(BioModel):
    """
    Holomorphic EqProp with complex-valued weights and states.
    Uses complex tanh activation which is holomorphic.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        self.layers = nn.ModuleList()
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            layer.weight = nn.Parameter(layer.weight.to(torch.complex64))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.to(torch.complex64))
            self.layers.append(layer)

        self.to(kwargs.get("device", "cpu"))

        opt_cls = kwargs.pop("optimizer_class", torch.optim.SGD)
        opt_kw = kwargs.pop("optimizer_kwargs", {"lr": self.lr, "momentum": 0.9})
        self.optimizer = opt_cls(self.parameters(), **opt_kw)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def forward_dynamics(
        self,
        activations: list[torch.Tensor],
        beta: float = 0.0,
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        new_activations = [activations[0]]

        num_layers = len(self.layers)

        for i in range(num_layers):
            layer = self.layers[i]
            h_prev = activations[i]

            a_bu = layer(h_prev)

            a_td = 0.0 + 0.0j
            if i < num_layers - 1:
                next_layer = self.layers[i + 1]
                h_next = activations[i + 2]
                if hasattr(next_layer, "weight"):
                    w = next_layer.weight
                    w_backward = w.conj().T
                    a_td = torch.matmul(h_next, w_backward.T)

            total_input = a_bu + a_td

            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                h_new = total_input

            if i == num_layers - 1 and beta > 0 and target is not None:
                if not target.is_complex():
                    target = target.to(h_new.dtype)

                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.complex64)

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        num_steps = steps if steps is not None else self.eq_steps

        activations, _, _ = settle_activations_list(
            activations_0=activations,
            forward_dynamics=self.forward_dynamics,
            steps=num_steps,
            beta=beta,
            target=target,
            return_trajectory=False,
            return_dynamics=False,
        )

        self._last_activations = activations

        return activations[-1].real

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        return _contrastive_step(
            self,
            x,
            y,
            layer_list=self.layers,
            beta=self.beta,
            use_conj=True,
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
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)
