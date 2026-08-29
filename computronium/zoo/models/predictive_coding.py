from __future__ import annotations

"""
Combined Predictive Coding Models
=================================

Aggregates all predictive coding models into a single module for the model zoo.

.. deprecated:: 1.0
   The legacy predictive coding models are deprecated. Use native compositions instead:
   - ``native_tile_pc`` for tile-based Predictive Coding
   - ``TileAlgorithm.from_pc()`` for generic tile PC
"""

import warnings

warnings.warn(
    "computronium.zoo.models.predictive_coding is deprecated. "
    "Use native compositions: native_tile_pc or TileAlgorithm.from_pc() instead.",
    DeprecationWarning,
    stacklevel=2,
)

import torch
from torch import nn

from computronium.config.unified import (
    ModelConfig,
    resolve_hidden_dims,
)
from computronium.core.local_learning.settling import (
    SettleConfig,
    SettleProtocol,
    SettleTelemetry,
    settle_universal,
)
from computronium.core.model import BioModel
from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model
from computronium.core.training_mixin import supervised_step
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer

__all__ = [
    "FabricPCGraphPCN",
    "PredictiveCodingHybrid",
]


# ============================================================================
# fabricpc_graph_pcn.py - FabricPCGraphPCN
# ============================================================================


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

        from computronium.graph.initialization import initialize_params
        from computronium.graph.nodes import Linear, ReLU
        from computronium.graph.topology import Edge, TaskMap, graph

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

        from computronium.graph.inference import InferenceSGD

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
        from computronium.graph.training import _feedforward

        activities = _feedforward(self.structure, self._params, x)
        return activities[self.structure.task_map.y.name]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=x.shape[0], shuffle=False)

        if self._mode == "backprop":
            from computronium.graph.training import train_backprop as trainer

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
            from computronium.graph.training import train_pcn as trainer

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
# pc_hybrid.py - PredictiveCodingHybrid with SettleProtocol
# ============================================================================


@register_model(
    "predictive_coding_hybrid",
    family="predictive_coding",
    locality_level=LocalityLevel.LOCAL,
    tags=["predictive-coding", "hybrid", status_tag("experimental")],
)
class PredictiveCodingHybrid(BioModel, SettleProtocol):
    """Layers predict inputs; FA propagates prediction errors.

    Implements SettleProtocol (Family B: activations list) for unified
    convergence instrumentation with settle_universal.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        # Extract PC-specific config before super().__init__
        if config is not None:
            extra = config.extra or {}
            infer_steps = extra.get("infer_steps", 20)
            eta_infer = extra.get("eta_infer", 0.05)
            convergence_threshold = extra.get("convergence_threshold", 1e-4)
            convergence_start = extra.get("convergence_start", 5)
        else:
            infer_steps = kwargs.get("infer_steps", 20)
            eta_infer = kwargs.get("eta_infer", 0.05)
            convergence_threshold = kwargs.get("convergence_threshold", 1e-4)
            convergence_start = kwargs.get("convergence_start", 5)

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
            OptimizerConfig(
                name="adam", lr=self.config.learning_rate, weight_decay=0.0
            ),
        )

        # SettleProtocol attributes
        self.convergence_threshold = convergence_threshold
        self.convergence_start = convergence_start
        self.max_steps = infer_steps
        self.infer_steps = infer_steps
        self.eta_infer = eta_infer

        # Transient state for settle_universal
        self._settle_beta: float = 0.0
        self._settle_target: torch.Tensor | None = None
        self._settle_activations: list[torch.Tensor] | None = None
        self._last_settle_converged: bool = False
        self._last_settle_steps: int = 0
        self._last_settle_final_delta: float = 0.0
        self._last_settle_telemetry: SettleTelemetry | None = None

    def _compute_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Compute forward activations for all layers."""
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)
        return activations

    def _compute_prediction_errors(
        self, activations: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Compute prediction errors for each layer.

        errors[i] = top_down[i](activations[i+1]) - activations[i]
        """
        errors = []
        for i in range(len(self.top_down)):
            upper = activations[i + 1].detach()
            lower_target = activations[i]
            prediction = self.top_down[i](upper)
            error = prediction - lower_target
            errors.append(error)
        return errors

    def _update_activations(
        self,
        activations: list[torch.Tensor],
        errors: list[torch.Tensor],
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Update activations based on prediction errors (one settle step).

        For hidden layers: h_i <- h_i + eta * (error_i - error_{i-1} @ W_{i+1})
        For output layer: h_out <- h_out + eta * (-error_{L-1} @ W_out + nudge)

        For the first hidden layer, error is in input space, so we backproject
        using the forward weight matrix (which is the transpose of top_down[0]).
        """
        new_activations = [activations[0]]  # Input layer fixed

        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                # Output layer: error is in hidden space (errors[-1])
                # Backproject error to output space using forward weight
                # forward weight is layers[-1].weight (hidden -> output), shape (output_dim, hidden_dim)
                # error is in hidden space (batch, hidden_dim)
                # backproject: error @ layers[-1].weight.T -> (batch, output_dim)
                w_forward = self.layers[-1].weight  # (output_dim, hidden_dim)
                error_hidden = errors[-1]  # Last error is in hidden space
                error_backproj = error_hidden @ w_forward.T  # (batch, output_dim)
                new_h = activations[i + 1] + self.eta_infer * (-error_backproj)
                # Add nudge if target provided
                if target is not None and self._settle_beta > 0:
                    new_h = new_h + self._settle_beta * (target - new_h)
                new_activations.append(new_h)
            else:
                # Hidden layers
                error_this = errors[i]
                if i > 0:
                    error_below = errors[i - 1]
                    # Back-propagate error through top-down weights
                    w_next = self.top_down[i].weight
                    error_prop = error_below @ w_next
                    new_h = activations[i + 1] + self.eta_infer * (
                        error_this - error_prop
                    )
                else:
                    # First hidden layer: error is in input space, backproject using forward weight
                    # forward weight is layers[0].weight (input -> hidden), shape (hidden, input)
                    # error_this is in input space (batch, input_dim)
                    # backproject: error_this @ layers[0].weight.T -> (batch, hidden_dim)
                    w_forward = self.layers[0].weight  # (hidden_dim, input_dim)
                    error_backproj = error_this @ w_forward.T  # (batch, hidden_dim)
                    new_h = activations[i + 1] + self.eta_infer * error_backproj
                new_activations.append(new_h)

        return new_activations

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        *,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[list[torch.Tensor]]]
        | tuple[torch.Tensor, dict[str, object]]
    ):
        """Settle the network and return output logits."""
        if return_dynamics:
            out, steps_taken, converged, telemetry = self._run_settle_universal(
                x,
                beta=beta,
                target=target,
                steps=steps,
                return_trajectory=return_trajectory,
                return_dynamics=return_dynamics,
            )
            if telemetry:
                dynamics = {
                    "deltas": telemetry.deltas,
                    "final_delta": telemetry.final_delta,
                    "steps_taken": telemetry.steps_taken,
                    "converged": telemetry.converged,
                    "settle_time_s": telemetry.settle_time_ms / 1000.0,
                }
            else:
                dynamics = {}
            return out, dynamics

        # Default path: run PC inference
        n_steps = steps if steps is not None else self.infer_steps
        activations = self._compute_activations(x)

        for _ in range(n_steps):
            errors = self._compute_prediction_errors(activations)
            activations = self._update_activations(activations, errors, target)

        self._settle_activations = activations
        out = activations[-1]

        if return_trajectory:
            return out, [activations]
        return out

    # ------------------------------------------------------------------
    # SettleProtocol Implementation (Family B: activations list)
    # ------------------------------------------------------------------

    def _initialize_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return initial activations list for settle_universal."""
        return self._compute_activations(x)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input (beta/target stored instead)."""
        return x

    def _step(
        self,
        state: list[torch.Tensor],
        x_transformed: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Single settle step for settle_universal.

        Runs one iteration of PC inference (error computation + update).
        """
        errors = self._compute_prediction_errors(state)
        return self._update_activations(state, errors, self._settle_target)

    def _check_converged(
        self,
        state_new: list[torch.Tensor],
        state_old: list[torch.Tensor],
        step: int,
    ) -> bool:
        """Custom convergence check for PC: max relative change in activations."""
        if step <= self.convergence_start:
            return False

        convergence_norm = 2
        max_rel_delta = 0.0
        for s_new, s_old in zip(state_new, state_old):
            abs_delta = torch.dist(s_new, s_old, p=convergence_norm).item()
            norm = s_old.norm(p=convergence_norm).item() + 1e-8
            rel_delta = abs_delta / norm
            max_rel_delta = max(max_rel_delta, rel_delta)

        return max_rel_delta < self.convergence_threshold

    def _on_step_end(
        self,
        step: int,
        state: list[torch.Tensor],
        delta: float,
    ) -> None:
        """Telemetry hook: called after each step."""
        # Telemetry collected by settle_universal

    def _on_converged(self, step: int, final_delta: float) -> None:
        """Telemetry hook: called when convergence is detected."""
        self._last_settle_converged = True
        self._last_settle_steps = step
        self._last_settle_final_delta = final_delta

    def _on_max_steps(self, step: int, final_delta: float) -> None:
        """Telemetry hook: called when max steps reached without convergence."""
        self._last_settle_converged = False
        self._last_settle_steps = step
        self._last_settle_final_delta = final_delta

    def _run_settle_universal(
        self,
        x: torch.Tensor,
        *,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> tuple[torch.Tensor, int, bool, SettleTelemetry | None]:
        """Run settle using the universal primitive with full telemetry."""
        self._settle_beta = beta
        self._settle_target = target

        config = SettleConfig(
            max_steps=steps if steps is not None else self.max_steps,
            convergence_threshold=self.convergence_threshold,
            convergence_start=self.convergence_start,
        )

        state, steps_taken, converged, telemetry = settle_universal(
            self,
            x,
            config=config,
            algorithm="predictive_coding",
            family="B",
            hardware=self.config.device if hasattr(self.config, "device") else "cpu",
            backend="pytorch",
            return_trajectory=return_trajectory,
        )

        self._settle_activations = state
        self._last_settle_telemetry = telemetry

        out = (
            state[-1]
            if state
            else torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
            )
        )

        return out, steps_taken, converged, telemetry

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """PC training step with settle_universal telemetry."""
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
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

        return (
            loss_cls + 0.1 * pc_loss,
            output,
            {
                "cls_loss": float(loss_cls.item()),
                "pc_loss": float(pc_loss.item()),
            },
        )

    def get_settle_telemetry(self) -> SettleTelemetry | None:
        """Return the last settle telemetry for external consumers."""
        return self._last_settle_telemetry
