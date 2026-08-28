"""Layer 2: Geometry — Topology & Routing."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from computronium.core.tile.topology import TileGraph

if TYPE_CHECKING:
    from computronium.ontology.substrate import Substrate


# ============================================================
# Geometry Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Configuration for network geometry/topology.

    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        num_layers: Number of layers (alternative to hidden_dims)
        topology_type: "feedforward", "recurrent", "tile_mesh",
            "neuromorphic", "spatial_lattice"
        connectivity: Optional adjacency specification
        recurrent_weight: Optional recurrent weight matrix (for recurrent topology)
        init_scale: Weight initialization scale for recurrent weights
    """

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    num_layers: int
    topology_type: str
    connectivity: dict | None
    recurrent_weight: list[list[float]] | None
    init_scale: float = 0.1

    @classmethod
    def feedforward(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
        )

    @classmethod
    def recurrent(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            topology_type="recurrent",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
        )

    @classmethod
    def tile_mesh(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        neurons_per_tile: int,
        tiles_per_layer: int,
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(),
            num_layers=num_layers,
            topology_type="tile_mesh",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
        )


# ============================================================
# Geometry Protocol
# ============================================================


def _set_param_name(tensor: Tensor, name: str) -> None:
    """Tag a parameter tensor with its substrate keying name."""
    setattr(tensor, "_param_name", name)


def _layer_stack(geometry: Geometry) -> nn.ModuleList | None:
    """Return the geometry's ordered module stack if it is layer-based."""
    layers = getattr(geometry, "_layers", None)
    return layers if isinstance(layers, nn.ModuleList) else None


def _recurrent_weight(geometry: Geometry) -> Tensor | None:
    """Return the geometry's recurrent weight matrix if present."""
    weight = getattr(geometry, "_recurrent_weight", None)
    return weight if isinstance(weight, Tensor) else None


@runtime_checkable
class Geometry(Protocol):
    """Spatial arrangement and message-passing topology.

    Defines the graph structure: nodes (computational units), edges (connections),
    and the routing protocol for forward/backward passes. The Geometry owns
    the parameters (weights) and exposes them to the substrate's operators.

    Key responsibility: route activations through the topology using the
    substrate's forward operator.
    """

    config: GeometryConfig

    @property
    @abstractmethod
    def params(self) -> dict[str, Tensor]:
        """Return all learnable parameters as a name -> tensor mapping."""
        ...

    @abstractmethod
    def forward(self, x: Tensor, substrate: Substrate) -> Tensor:
        """Route input through the topology using substrate's forward operator.

        Args:
            x: Input tensor
            substrate: The substrate providing the forward operator

        Returns:
            Output tensor after routing through the geometry
        """
        ...

    @abstractmethod
    def route(self, activations: Tensor) -> Tensor:
        """Route activations through the topology (single step).

        Used by StateDynamics for iterative settling.
        """
        ...

    @abstractmethod
    def update_params(self, new_params: dict[str, Tensor]) -> None:
        """Update geometry parameters in-place from ParameterUpdate output."""
        ...

    @abstractmethod
    def transition_modules(self) -> list[nn.Module]:
        """Return modules in forward order for TransitionGraph protocol."""
        ...

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate
    ) -> list[Tensor]:
        """Return intermediate activations for each layer (optional).

        Default implementation returns only the final output.
        Override for geometries that support layer-wise inspection
        (e.g., FeedforwardGeometry for predictive coding).

        Args:
            x: Input tensor
            substrate: The substrate providing the forward operator

        Returns:
            List of activations [input, layer1_out, layer2_out, ..., output]
        """
        out = self.forward(x, substrate)
        return [x, out]


# ============================================================
# Default/Reference Geometry Implementations
# ============================================================


def _learnable_weight_names(params: dict[str, Tensor]) -> list[str]:
    """Parameter names that receive pseudo-gradients (2-D weight matrices).

    Credits emit exactly one pseudo-gradient per learnable weight, in this
    order. Biases and other auxiliary parameters never receive gradients
    from the local learning rules.
    """
    return [n for n, p in params.items() if "weight" in n and p.ndim == 2]


class FeedforwardGeometry(nn.Module):
    """Standard feedforward DAG topology (MLP, CNN)."""

    _layers: nn.ModuleList

    def __init__(
        self,
        config: GeometryConfig,
        layers: nn.ModuleList | list[nn.Module] | None = None,
    ):
        super().__init__()
        self.config = config
        self._layers = nn.ModuleList(layers) if layers else nn.ModuleList()
        if not self._layers:
            self._build_layers()

    def _build_layers(self) -> None:
        dims = (
            self.config.input_dim,
            *self.config.hidden_dims,
            self.config.output_dim,
        )
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._layers = nn.ModuleList(layers)
        self._set_param_names()

    def _set_param_names(self) -> None:
        """Set _param_name attribute on weight tensors for substrate keying."""
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                _set_param_name(layer.weight, f"layer_{i}_weight")
                if layer.bias is not None:
                    _set_param_name(layer.bias, f"layer_{i}_bias")

    @property
    def params(self) -> dict[str, Tensor]:
        return dict(self._layers.named_parameters())  # type: ignore[return-value]

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    # Out-of-place add: in-place adds on grad-tracking tensors
                    # pin the whole downstream settle graph (CUDA leak)
                    h = h + layer.bias
            else:
                h = layer(h)
        return h

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer.

        Returns:
            List of activations [input, layer1_out, layer2_out, ..., output]
            where layer outputs are after activation functions (ReLU).
        """
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        acts = [h]  # Include input
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    # Out-of-place add: in-place adds on grad-tracking tensors
                    # pin the whole downstream settle graph (CUDA leak)
                    h = h + layer.bias
            else:
                h = layer(h)
                # Add after activation functions (ReLU, etc.)
                acts.append(h)
        # Add final output if last layer was Linear (no trailing activation)
        if self._layers and isinstance(self._layers[-1], nn.Linear):
            acts.append(h)
        return acts

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the geometry (for settling dynamics)."""
        h = activations
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = h @ layer.weight.T  # ruff: ignore[non-augmented-assignment]
                if layer.bias is not None:
                    # Out-of-place add: in-place adds break autograd
                    h = h + layer.bias
            else:
                h = layer(h)
        return h

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0].isdigit():
                layer_idx = int(parts[0])
                param_name = ".".join(parts[1:])
                if layer_idx < len(self._layers) and hasattr(
                    self._layers[layer_idx], param_name
                ):
                    getattr(self._layers[layer_idx], param_name).data.copy_(param)
            else:
                for layer in self._layers:
                    if hasattr(layer, name):
                        getattr(layer, name).data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [m for m in self._layers if isinstance(m, nn.Linear)]


class RecurrentGeometry(nn.Module):
    """Recurrent attractor topology (Hopfield, EqProp MLPs).

    The hidden state is recurrently connected: h_{t+1} = f(h_t, x).
    """

    _layers: nn.ModuleList
    _recurrent_weight: nn.Parameter | None

    def __init__(
        self,
        config: GeometryConfig,
        layers: nn.ModuleList | None = None,
        hidden_dim: int | None = None,
        recurrent_weight: Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self._layers = layers or nn.ModuleList()
        self._recurrent_weight = None
        if not self._layers and config.hidden_dims:
            self._build_layers()
        if recurrent_weight is not None:
            self._recurrent_weight = nn.Parameter(recurrent_weight)
        elif hidden_dim is not None and self._recurrent_weight is None:
            # For EqProp, initialize recurrent weight to small random values
            # so the nudge can propagate backwards through the network
            # (zero init prevents gradient flow to hidden layers)
            self._recurrent_weight = nn.Parameter(
                torch.randn(hidden_dim, hidden_dim) * config.init_scale * 0.1
            )

    def _build_layers(self) -> None:
        # For recurrent: input -> hidden (with recurrent), hidden -> output
        dims = (
            self.config.input_dim,
            *self.config.hidden_dims,
            self.config.output_dim,
        )
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._layers = nn.ModuleList(layers)
        # Add recurrent weight for the last hidden layer
        if len(self.config.hidden_dims) > 0 and self._recurrent_weight is None:
            hidden_dim = self.config.hidden_dims[-1]
            if self.config.recurrent_weight is not None:
                self._recurrent_weight = nn.Parameter(
                    torch.tensor(self.config.recurrent_weight)
                )
            else:
                # Small random initialization for EqProp
                self._recurrent_weight = nn.Parameter(
                    torch.randn(hidden_dim, hidden_dim) * self.config.init_scale * 0.1
                )
        self._set_param_names()

    def _set_param_names(self) -> None:
        """Set _param_name attribute on weight tensors for substrate keying."""
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                _set_param_name(layer.weight, f"layer_{i}_weight")
                if layer.bias is not None:
                    _set_param_name(layer.bias, f"layer_{i}_bias")
        if self._recurrent_weight is not None:
            _set_param_name(self._recurrent_weight, "recurrent_weight")

    @property
    def params(self) -> dict[str, Tensor]:
        params = dict(self._layers.named_parameters())
        if self._recurrent_weight is not None:
            params["recurrent_weight"] = self._recurrent_weight
        return params  # type: ignore[return-value]

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        """Full forward pass with recurrence (single step)."""
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    # Out-of-place: in-place adds break autograd
                    h = h + layer.bias
            else:
                h = layer(h)
            # Apply recurrent connection after each hidden layer (except output)
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                # Out-of-place: in-place adds break autograd
                h = h + op(h, self._recurrent_weight)
        return h

    def route(self, activations: Tensor) -> Tensor:
        """Single recurrent step: h_{t+1} = f(h_t) = activation(W_rec @ h_t).

        Expects activations to be the hidden state (batch_size x hidden_dim).
        """
        h = activations
        if self._recurrent_weight is not None:
            # Hidden state should match recurrent weight dimensions
            if h.shape[-1] == self._recurrent_weight.shape[0]:
                # Out-of-place: in-place matmul breaks autograd
                h = h @ self._recurrent_weight.T
            else:
                # Activations are output dim; we can't apply recurrent weight
                # This happens when route is called on output instead of hidden state
                pass
        return h

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            if name == "recurrent_weight" and self._recurrent_weight is not None:
                self._recurrent_weight.data.copy_(param)
            else:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0].isdigit():
                    layer_idx = int(parts[0])
                    param_name = ".".join(parts[1:])
                    if layer_idx < len(self._layers) and hasattr(
                        self._layers[layer_idx], param_name
                    ):
                        getattr(self._layers[layer_idx], param_name).data.copy_(param)
                else:
                    for layer in self._layers:
                        if hasattr(layer, name):
                            getattr(layer, name).data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [m for m in self._layers if isinstance(m, nn.Linear)]

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer."""
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        acts = [h]
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h = h + layer.bias
            else:
                h = layer(h)
                # Add after activation functions
                acts.append(h)
            # Apply recurrent connection after each hidden layer (except output)
            # Out-of-place: in-place adds pin the downstream settle graph
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                h = h + op(h, self._recurrent_weight)
        # Add final output if last layer was Linear (no trailing activation)
        if isinstance(self._layers[-1], nn.Linear):
            acts.append(h)
        return acts


class TileGeometry(nn.Module):
    """TileNet mesh topology: modular independent tiles with local boundaries and asynchronous routing.

    Implements the Geometry protocol for the TileNet architecture. The topology consists of
    a layered tile graph where each tile is a computational unit with its own neurons,
    and tiles within a layer route activations in parallel. Inter-layer connections
    follow a dense or configurable adjacency pattern.

    Key properties:
    - Tile-local computation (each tile has its own weight matrix)
    - Layer-wise parallel routing
    - Configurable intra/inter-layer connectivity
    - Support for skip connections
    - Asynchronous boundary conditions (MoE-style gating)
    """

    _tile_weights: nn.ParameterDict
    _tile_biases: nn.ParameterDict
    _graph: TileGraph
    _input_projection: nn.Linear
    _output_projection: nn.Linear

    def __init__(
        self,
        config: GeometryConfig,
        tile_graph: TileGraph | None = None,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.config = config
        self._tile_weights = nn.ParameterDict()
        self._tile_biases = nn.ParameterDict()

        if tile_graph is not None:
            self._graph = tile_graph
        else:
            self._graph = TileGraph()
            self._graph.build_layered(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                neurons_per_tile=neurons_per_tile,
                num_hidden_layers=max(config.num_layers - 2, 1),
                tiles_per_layer=tiles_per_layer,
                use_skip_connections=use_skip_connections,
            )

        self._build_projections()
        self._build_tile_params()
        self._set_projection_param_names()

    def _build_projections(self) -> None:
        """Build input/output projections between raw IO and tile-state space."""
        input_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.input_tile_ids
        )
        output_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.output_tile_ids
        )
        self._input_projection = nn.Linear(
            self.config.input_dim, input_neurons, bias=True
        )
        self._output_projection = nn.Linear(
            output_neurons, self.config.output_dim, bias=True
        )

    def _build_tile_params(self) -> None:
        """Per-edge incoming weights and per-tile biases."""
        import math

        for tid, tile in self._graph.tiles.items():
            if tile.is_input:
                continue
            bias = nn.Parameter(torch.zeros(tile.neurons))
            _set_param_name(bias, f"tile_bias_{tid}")
            self._tile_biases[str(tid)] = bias
            for src_id in tile.bwd_neighbors:
                src = self._graph.tiles[src_id]
                bound = 1.0 / math.sqrt(src.neurons) if src.neurons > 0 else 0.0
                w = torch.empty(tile.neurons, src.neurons).uniform_(-bound, bound)
                param = nn.Parameter(w)
                _set_param_name(param, f"tile_weight_{src_id}_{tid}")
                self._tile_weights[f"{src_id}_{tid}"] = param

    def _set_projection_param_names(self) -> None:
        """Set _param_name on input/output projection weights."""
        if self._input_projection is not None:
            _set_param_name(self._input_projection.weight, "input_proj_weight")
            if self._input_projection.bias is not None:
                _set_param_name(self._input_projection.bias, "input_proj_bias")
        if self._output_projection is not None:
            _set_param_name(self._output_projection.weight, "output_proj_weight")
            if self._output_projection.bias is not None:
                _set_param_name(self._output_projection.bias, "output_proj_bias")

    @staticmethod
    def _weight_key(src_id: int, dst_id: int) -> str:
        return f"{src_id}_{dst_id}"

    @property
    def params(self) -> dict[str, Tensor]:
        params = {}
        if self._input_projection is not None:
            params.update({
                f"input_proj.{k}": v
                for k, v in self._input_projection.named_parameters()
            })
        if self._output_projection is not None:
            params.update({
                f"output_proj.{k}": v
                for k, v in self._output_projection.named_parameters()
            })
        params.update({f"tile_bias.{k}": v for k, v in self._tile_biases.items()})
        params.update({f"tile_weight.{k}": v for k, v in self._tile_weights.items()})
        return params

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        """Route input through the tile mesh using substrate's forward operator."""
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Project input to tile space
        h = self._input_projection(x)

        # Set input tile activities
        offset = 0
        for tid in self._graph.input_tile_ids:
            n = self._graph.tiles[tid].neurons
            self._graph.tiles[tid].activity = h[:, offset : offset + n]
            offset += n

        # Forward propagate through layers (skip input layer)
        for layer_tiles in self._graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self._graph.tiles[tid]
                # Compute weighted sum of incoming activities + bias
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = self._graph.tiles[src_id].activity
                    if src_act is None:
                        continue
                    w = self._tile_weights[self._weight_key(src_id, tid)]
                    contrib = op(src_act, w)
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += (
                        self
                        ._tile_biases[str(tid)]
                        .unsqueeze(0)
                        .expand(acc.shape[0], -1)
                    )
                    tile.activity = acc
                    tile.prediction = acc

        # Collect output tile activities
        out_acts: list[Tensor] = []
        for tid in self._graph.output_tile_ids:
            act = self._graph.tiles[tid].activity
            if act is not None:
                out_acts.append(act)

        if not out_acts:
            return torch.empty(x.shape[0], self.config.output_dim, device=x.device)

        h = torch.cat(out_acts, dim=1)
        return self._output_projection(h)

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the tile mesh (for settling dynamics).

        This is used by StateDynamics for iterative settling. It expects
        activations to already be in tile space (i.e., after input projection).
        """
        # For settling, we assume activations represent the current tile activities
        # We need to distribute them to the appropriate tiles and route one step
        # This is a simplified version - in practice, the StateDynamics would
        # maintain the tile activities directly
        return self._route_flat(activations)

    def _route_flat(self, flat_activations: Tensor) -> Tensor:
        """Route flat concatenated tile activities through one step."""
        # Distribute flat activations to tiles
        self._set_tile_activities_from_flat(flat_activations)

        # One propagation step through all non-input tiles
        for layer_tiles in self._graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self._graph.tiles[tid]
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = self._graph.tiles[src_id].activity
                    if src_act is None:
                        continue
                    w = self._tile_weights[self._weight_key(src_id, tid)]
                    contrib = src_act @ w.T
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += (
                        self
                        ._tile_biases[str(tid)]
                        .unsqueeze(0)
                        .expand(acc.shape[0], -1)
                    )
                    tile.activity = acc
                    tile.prediction = acc

        # Collect and flatten
        acts: list[Tensor] = []
        for layer_tiles in self._graph.layer_ids:
            for tid in layer_tiles:
                act = self._graph.tiles[tid].activity
                if act is not None:
                    acts.append(act)
        return (
            torch.cat(acts, dim=1)
            if acts
            else torch.empty(
                flat_activations.shape[0], 0, device=flat_activations.device
            )
        )

    def _set_tile_activities_from_flat(self, flat_activations: Tensor) -> None:
        """Distribute flat concatenated activations to individual tiles."""
        offset = 0
        for layer_tiles in self._graph.layer_ids:
            for tid in layer_tiles:
                n = self._graph.tiles[tid].neurons
                if offset + n <= flat_activations.shape[1]:
                    self._graph.tiles[tid].activity = flat_activations[
                        :, offset : offset + n
                    ]
                    offset += n

    def _get_flat_activities(self) -> Tensor:
        """Collect all tile activities as a flat concatenated tensor."""
        acts: list[Tensor] = []
        for layer_tiles in self._graph.layer_ids:
            for tid in layer_tiles:
                act = self._graph.tiles[tid].activity
                if act is not None:
                    acts.append(act)
        return torch.cat(acts, dim=1) if acts else torch.empty(1, 0)

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        """Update geometry parameters in-place from ParameterUpdate output."""
        for name, param in new_params.items():
            if name.startswith("input_proj.") and self._input_projection is not None:
                pname = name.replace("input_proj.", "")
                if hasattr(self._input_projection, pname):
                    getattr(self._input_projection, pname).data.copy_(param)
            elif (
                name.startswith("output_proj.") and self._output_projection is not None
            ):
                pname = name.replace("output_proj.", "")
                if hasattr(self._output_projection, pname):
                    getattr(self._output_projection, pname).data.copy_(param)
            elif name.startswith("tile_bias."):
                key = name.replace("tile_bias.", "")
                if key in self._tile_biases:
                    self._tile_biases[key].data.copy_(param)
            elif name.startswith("tile_weight."):
                key = name.replace("tile_weight.", "")
                if key in self._tile_weights:
                    self._tile_weights[key].data.copy_(param)
            # Try direct match for backward compatibility
            elif name in self._tile_weights:
                self._tile_weights[name].data.copy_(param)
            elif name in self._tile_biases:
                self._tile_biases[name].data.copy_(param)

    def _validate_shapes(self) -> None:
        """Validate that projection dimensions match tile graph structure.

        Checks that input/output projection dimensions match the sum of
        input/output tile neurons. Raises ValueError if mismatch detected.
        """
        # Validate input projection
        input_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.input_tile_ids
        )
        in_feats = self._input_projection.in_features
        out_feats = self._input_projection.out_features
        if in_feats != self.config.input_dim:
            msg = f"Input projection in_features ({in_feats}) != config.input_dim ({self.config.input_dim})"
            raise ValueError(msg)
        if out_feats != input_neurons:
            msg = f"Input projection out_features ({out_feats}) != sum of input tile neurons ({input_neurons})"
            raise ValueError(msg)

        # Validate output projection
        output_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.output_tile_ids
        )
        out_in_feats = self._output_projection.in_features
        out_out_feats = self._output_projection.out_features
        if out_in_feats != output_neurons:
            msg = f"Output projection in_features ({out_in_feats}) != sum of output tile neurons ({output_neurons})"
            raise ValueError(msg)
        if out_out_feats != self.config.output_dim:
            msg = f"Output projection out_features ({out_out_feats}) != config.output_dim ({self.config.output_dim})"
            raise ValueError(msg)

    def transition_modules(self) -> list[nn.Module]:
        """Return modules in forward order for TransitionGraph protocol."""
        modules = []
        if self._input_projection is not None:
            modules.append(self._input_projection)
        # Tile weights are Parameters, not Modules
        if self._output_projection is not None:
            modules.append(self._output_projection)
        return modules

    def get_boundary_tiles(self, device_map: dict[int, int]) -> dict[int, list[int]]:
        """Identify boundary tiles that connect to different devices.

        For P2P/distributed training, this identifies tiles that need
        cross-device communication.
        """
        return self._graph.get_boundary_tiles(device_map)

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer."""
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Project input to tile space
        h = self._input_projection(x)
        acts = [h]  # Input projection output

        # Set input tile activities
        offset = 0
        for tid in self._graph.input_tile_ids:
            n = self._graph.tiles[tid].neurons
            self._graph.tiles[tid].activity = h[:, offset : offset + n]
            offset += n

        # Forward propagate through layers (skip input layer)
        for layer_tiles in self._graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self._graph.tiles[tid]
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = self._graph.tiles[src_id].activity
                    if src_act is None:
                        continue
                    w = self._tile_weights[self._weight_key(src_id, tid)]
                    contrib = op(src_act, w)
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += (
                        self
                        ._tile_biases[str(tid)]
                        .unsqueeze(0)
                        .expand(acc.shape[0], -1)
                    )
                    tile.activity = acc
                    tile.prediction = acc

        # Collect output tile activities
        out_acts: list[Tensor] = []
        for tid in self._graph.output_tile_ids:
            act = self._graph.tiles[tid].activity
            if act is not None:
                out_acts.append(act)

        if out_acts:
            h = torch.cat(out_acts, dim=1)
            out = self._output_projection(h)
            acts.append(out)

        return acts