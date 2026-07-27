"""Equilibrium Propagation model variants."""


import torch
from torch import nn

from ...base import BioModel, ModelConfig, register_model


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
            layer.weight = nn.Parameter(layer.weight.to(torch.complex64))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.to(torch.complex64))
            self.layers.append(layer)

        self.to(kwargs.get("device", "cpu"))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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

        for _ in range(num_steps):
            activations = self.forward_dynamics(activations, beta, target)

        self._last_activations = activations

        return activations[-1].real

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        target = target.to(torch.complex64)

        with torch.no_grad():
            self.forward(x, beta=0.0)
            free_activations = self._last_activations
            output_free = free_activations[-1]

        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged_activations = self._last_activations

        self.optimizer.zero_grad()

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                h_prev_free = free_activations[i]
                h_post_free = free_activations[i + 1]

                h_prev_nudged = nudged_activations[i]
                h_post_nudged = nudged_activations[i + 1]

                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged.conj())
                prod_free = torch.matmul(h_post_free.T, h_prev_free.conj())

                dW = (prod_nudged - prod_free) / self.beta
                dW = dW / x.size(0)

                if layer.weight.grad is None:
                    layer.weight.grad = -dW
                else:
                    layer.weight.grad += -dW

                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).sum(0) / self.beta
                    db = db / x.size(0)
                    if layer.bias.grad is None:
                        layer.bias.grad = -db
                    else:
                        layer.bias.grad += -db

        self.optimizer.step()

        pred = output_free.real.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        loss = nn.functional.cross_entropy(output_free.real, y).item()

        return {
            "loss": loss,
            "accuracy": acc,
        }

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
