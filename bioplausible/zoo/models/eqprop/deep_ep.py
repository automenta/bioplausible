"""Equilibrium Propagation model variants."""

import torch
from torch import nn

from bioplausible.core.config import (
    ModelConfig,
    _build_model_config,
    resolve_hidden_dims,
)
from bioplausible.core.model import BioModel
from bioplausible.core.registry import register_model

from ....zoo._settling import settle_activations_list
from ._contrastive import _contrastive_step

__all__ = [
    "DirectedEP",
]


@register_model(
    "directed_ep",
    family="eqprop",
    tags=["eqprop", "directed"],
)
class DirectedEP(BioModel):
    """
    Directed EqProp (DEEP) with separate forward and feedback weights.
    Both sets of weights are updated to minimize the energy/loss.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.beta = self.config.beta
        self.eq_steps = self.config.equilibrium_steps
        self.lr = self.config.learning_rate

        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        self.forward_layers = nn.ModuleList()
        self.feedback_layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            fwd = nn.Linear(dims[i], dims[i + 1])
            self.forward_layers.append(fwd)

            bwd = nn.Linear(dims[i + 1], dims[i], bias=False)
            self.feedback_layers.append(bwd)

        self.to(kwargs.get("device", "cpu"))
        opt_cls = kwargs.pop("optimizer_class", torch.optim.SGD)
        opt_kw = kwargs.pop("optimizer_kwargs", {"lr": self.lr, "momentum": 0.9})
        self.optimizer = opt_cls(self.parameters(), **opt_kw)

    def forward_dynamics(
        self,
        activations: list[torch.Tensor],
        beta: float = 0.0,
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:

        updated_activations = [activations[0]]

        for k in range(len(self.forward_layers)):
            h_prev = activations[k]

            a_bu = self.forward_layers[k](h_prev)

            a_td = 0.0
            if k < len(self.forward_layers) - 1:
                h_next = activations[k + 2]
                a_td = self.feedback_layers[k + 1](h_next)

            total = a_bu + a_td

            if k < len(self.forward_layers) - 1:
                h_new = self.activation(total)
            else:
                h_new = total

            if k == len(self.forward_layers) - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            updated_activations.append(h_new)

        return updated_activations

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
        for i, layer in enumerate(self.forward_layers):
            h = layer(h)
            if i < len(self.forward_layers) - 1:
                h = self.activation(h)
            activations.append(h)

        # No early convergence for DirectedEP — always run full steps
        activations, trajectory, dynamics = settle_activations_list(
            activations_0=activations,
            forward_dynamics=self.forward_dynamics,
            steps=eq_steps,
            beta=beta,
            target=target,
            return_trajectory=return_trajectory,
            return_dynamics=return_dynamics,
            convergence_norm=float("inf"),
            convergence_threshold=1e-3,
            convergence_start=eq_steps,  # never triggers early stop
        )

        self._last_activations = activations
        out = activations[-1]

        if return_dynamics:
            # Rebuild full dynamics dict with trajectory
            if trajectory is not None:
                dynamics["trajectory"] = trajectory  # type: ignore[typeddict-item]
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
            layer_list=self.forward_layers,
            beta=self.beta,
            feedback_layer_list=self.feedback_layers,
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
