"""Equilibrium Propagation model variants."""

import torch
from torch import nn

from ....acceleration import compile_settling_loop
from ....zoo._settling import settle_activations_list
from bioplausible.core.config import ModelConfig, resolve_hidden_dims
from bioplausible.core.model import BioModel
from bioplausible.core.registry import register_model
from ._contrastive import _contrastive_step

__all__ = [
    "StandardEqProp",
]


@register_model(
    "eqprop",
    family="eqprop",
    tags=["eqprop"],
)
class StandardEqProp(BioModel):
    """
    Standard EqProp with free/nudged phases and bidirectional relaxation.

    Implements the dynamics:
    h_i = sigma(W_i h_{i-1} + W_{i+1}^T h_{i+1} + b_i)
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        self.layers = nn.ModuleList()
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        n_layers = len(dims) - 1
        for i in range(n_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            role = "output" if i == n_layers - 1 else "hidden"
            layer = self.apply_spectral_norm(layer, layer_role=role)
            self.layers.append(layer)

        self.to(kwargs.get("device", "cpu"))

        opt_cls = kwargs.pop("optimizer_class", torch.optim.SGD)
        opt_kw = kwargs.pop("optimizer_kwargs", {"lr": self.lr, "momentum": 0.9})
        self.optimizer = opt_cls(self.parameters(), **opt_kw)

    def _get_spectral_normalized_weight(self, layer: nn.Module) -> torch.Tensor:
        if not self.training and hasattr(layer, "_cached_sn_weight"):
            return layer._cached_sn_weight

        weight = layer.weight

        if not self.training:
            layer._cached_sn_weight = weight.detach()

        return weight

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.modules():
                if hasattr(module, "_cached_sn_weight"):
                    delattr(module, "_cached_sn_weight")
        return self

    @compile_settling_loop
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

            if not self.training:
                w = self._get_spectral_normalized_weight(layer)
                b = layer.bias
                a_bu = torch.nn.functional.linear(h_prev, w, b)
            else:
                a_bu = layer(h_prev)

            a_td = 0.0
            if i < num_layers - 1:
                next_layer = self.layers[i + 1]
                h_next = activations[i + 2]
                if hasattr(next_layer, "weight"):
                    if not self.training:
                        w = self._get_spectral_normalized_weight(next_layer)
                    else:
                        w = next_layer.weight
                    a_td = torch.matmul(h_next, w)

            total_input = a_bu + a_td

            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                h_new = total_input

            if i == num_layers - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        eq_steps = steps if steps is not None else self.eq_steps

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        activations, trajectory, dynamics = settle_activations_list(
            activations_0=activations,
            forward_dynamics=self.forward_dynamics,
            steps=eq_steps,
            beta=beta,
            target=target,
            return_trajectory=return_trajectory,
            return_dynamics=return_dynamics,
            convergence_norm=2,
            convergence_threshold=1e-3,
            convergence_start=5,
        )

        self._last_activations = activations
        out = activations[-1]

        if return_dynamics:
            return out, dynamics
        if return_trajectory:
            return out, trajectory
        return out

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
        )
