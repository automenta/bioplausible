"""
Combined Predictive Coding Models
==================================

Aggregates all predictive coding models into a single module for the model zoo.
"""

from __future__ import annotations

import torch
from torch import nn

from bioplausible.config.unified import (
    ModelConfig,
    resolve_hidden_dims,
)
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import LocalityLevel, register_model
from bioplausible.core.training_mixin import supervised_step
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer

# ============================================================================
# fabricpc_graph_pcn.py - FabricPCGraphPCN
# ============================================================================


__all__ = [
    "FabricPCGraphPCN",
    "PredictiveCodingHybrid",
]


@register_model(
    "fabricpc_graph_pcn",
    family="predictive_coding",
    locality_level=LocalityLevel.LOCAL,
    tags=["predictive-coding", "fabricpc", status_tag("stable")],
)
class FabricPCGraphPCN(BioModel):
    """Predictive Coding model using FabricPC graph topology."""

    algorithm_name = "FabricPC Graph PCN"

    def transition_modules(self) -> list[nn.Module]:
        """FabricPC uses graph-internal parameters, not nn.Module chains.

        The transition dynamics are handled by ``self.structure`` (a graph
        topology), not by an ordered list of nn.Module submodules. Parameters
        live in ``self._params: dict[str, dict[str, Tensor]]``, not as
        ``nn.Module.parameters()``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses FabricPC graph topology, not "
            f"nn.Module transitions. Use a per-graph propagator (e.g. "
            f"'predictive_coding_hybrid') instead of EqProp/CHL/MEP."
        )

    def __init__(
        self,
        config: ModelConfig | None = None,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = ModelConfig(
                name=self.algorithm_name,
                input_dim=input_dim if input_dim is not None else 784,
                output_dim=output_dim if output_dim is not None else 10,
                hidden_dims=[hidden_dim] if hidden_dim is not None else [256],
                extra=kwargs,
            )
        super().__init__(config)

        from bioplausible.graph.initialization import initialize_params
        from bioplausible.graph.nodes import Linear, ReLU
        from bioplausible.graph.topology import Edge, TaskMap, graph

        hidden_dims = self.config.hidden_dims or [self.hidden_dim]
        dims = [self.input_dim] + list(hidden_dims) + [self.output_dim]

        nodes = []
        edges = []

        input_node = Linear(shape=(dims[0], dims[0]), name="input")
        nodes.append(input_node)

        prev_node = input_node
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            linear = Linear(shape=(d_in, d_out), name=f"linear_{i}")
            nodes.append(linear)
            edges.append(Edge(source=prev_node, target=linear.slot("input")))

            if i < len(dims) - 2:
                act = ReLU(name=f"relu_{i}")
                nodes.append(act)
                edges.append(Edge(source=linear, target=act.slot("input")))
                prev_node = act
            else:
                prev_node = linear

        output_node = prev_node

        from bioplausible.graph.inference import InferenceSGD

        extra = self.config.extra or {}
        infer_steps = extra.get("infer_steps", 20)
        eta_infer = extra.get("eta_infer", 0.05)

        self.structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        )

        self._params: dict[str, dict[str, torch.Tensor]] = initialize_params(
            self.structure, rng_key=0
        )

        # Register graph parameters as nn.Parameter so standard optimizers work
        for node_name, node_params in self._params.items():
            for param_name, tensor in node_params.items():
                safe_name = f"_graph_param_{node_name}_{param_name}".replace(".", "_")
                self.register_parameter(safe_name, nn.Parameter(tensor))

        self._mode = extra.get("mode", "pcn")
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> FabricPCGraphPCN:
        super().to(device)
        self._device = device
        for node_name in self._params:
            for param_name in self._params[node_name]:
                self._params[node_name][param_name] = self._params[node_name][
                    param_name
                ].to(device)
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        from bioplausible.graph.training import _feedforward

        activities = _feedforward(self.structure, self._params, x)
        return activities[self.structure.task_map.y.name]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=x.shape[0], shuffle=False)

        if self._mode == "backprop":
            from bioplausible.graph.training import train_backprop as trainer

            extra = self.config.extra or {}
            results = trainer(
                self.structure,
                self._params,
                loader,
                epochs=1,
                lr=self.config.learning_rate,
                device=self._device,
            )
        else:
            from bioplausible.graph.training import train_pcn as trainer

            extra = self.config.extra or {}
            results = trainer(
                self.structure,
                self._params,
                loader,
                epochs=1,
                lr=self.config.learning_rate,
                device=self._device,
                infer_steps=extra.get("infer_steps", 20),
                eta_infer=extra.get("eta_infer", 0.05),
            )

        return {
            "loss": results["train_loss"],
            "accuracy": results["train_acc"],
        }


# ============================================================================
# pc_hybrid.py - PredictiveCodingHybrid
# ============================================================================


@register_model(
    "predictive_coding_hybrid",
    family="predictive_coding",
    locality_level=LocalityLevel.LOCAL,
    tags=["predictive-coding", "hybrid", status_tag("experimental")],
)
class PredictiveCodingHybrid(BioModel):
    """Layers predict inputs; FA propagates prediction errors."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
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

        self.criterion = nn.CrossEntropyLoss()

        self.top_down = nn.ModuleList()
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i + 1], dims[i])
            self.top_down.append(layer)

        self.optimizer = create_optimizer(
            self,
            OptimizerConfig(name="adam", lr=self.config.learning_rate, weight_decay=0.0),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        return supervised_step(
            self,
            self.optimizer,
            x,
            y,
            loss_fn=self._composite_loss,
        )

    @staticmethod
    def _composite_loss(
        model: PredictiveCodingHybrid,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward with intermediate activations + composite PC loss."""
        activations = [x]
        h = x
        for i, layer in enumerate(model.layers):
            h = layer(h)
            if i < len(model.layers) - 1:
                h = model.activation(h)
            activations.append(h)

        output = activations[-1]
        loss_cls = model.criterion(output, y)

        pc_loss = torch.zeros((), device=output.device, dtype=output.dtype)
        for i in range(len(model.layers)):
            upper = activations[i + 1].detach()
            lower_target = activations[i].detach()
            prediction = model.top_down[i](upper)
            pc_loss = pc_loss + nn.functional.mse_loss(prediction, lower_target)

        return loss_cls + 0.1 * pc_loss, output
