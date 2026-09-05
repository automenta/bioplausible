"""Layer 2: Geometry — Topology & Routing."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from computronium.core.tile.topology import TileGraph
from computronium.ontology._tile_blocks import (
    TileBlockView,
    assemble_transition_blocks,
    build_block_view,
    tile_hopfield_energy,
    tile_layered_params,
)
from computronium.ontology._tile_blocks import (
    scatter_block_grads as _scatter_block_grads,
)
from computronium.ontology._tile_blocks import (
    tile_settle_block_acts as settle_block_acts,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.ontology.depth import DepthMetric
    from computronium.ontology.substrate import Substrate

_DEFAULT_INIT_SCALE = 0.1

type InitScheme = Literal["default", "mupc"]


def _linear_stack(
    dims: tuple[int, ...],
    init_scale: float,
    init_scheme: InitScheme = "default",
) -> list[nn.Module]:
    """MLP layers with weights rescaled from PyTorch's fan-in-adaptive init.

    ``init_scale`` multiplies the default init; the config default
    ``_DEFAULT_INIT_SCALE`` is the ×1.0 identity, so legacy builds are
    bit-identical.

    ``init_scheme="mupc"`` replaces the fan-in init with μPC depth-scaled
    init (Ernoult et al., arXiv:2505.13124): weights ~ N(0, 1), hidden
    layers scaled 1/√(N·L), output layer scaled 1/N (N = uniform width,
    L = hidden-layer count). ``init_scale`` is not applied in this scheme.
    """
    layers: list[nn.Module] = []
    width = dims[1] if len(dims) > 1 else dims[0]
    num_hidden = len(dims) - 2
    for i in range(len(dims) - 1):
        layer = nn.Linear(dims[i], dims[i + 1])
        match init_scheme:
            case "mupc" if num_hidden > 0:
                nn.init.normal_(layer.weight)
                scale = (
                    1.0 / width
                    if i == len(dims) - 2
                    else (1.0 / (width * num_hidden) ** 0.5)
                )
                layer.weight.data.mul_(scale)
            case _:
                if init_scale != _DEFAULT_INIT_SCALE:
                    layer.weight.data.mul_(init_scale / _DEFAULT_INIT_SCALE)
        layers.append(layer)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return layers


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
            "neuromorphic", "spatial_lattice", "attention", "conv", "graph"
        connectivity: Optional adjacency specification
        recurrent_weight: Optional recurrent weight matrix (for recurrent topology)
        init_scale: Multiplicative scale on weight initialization (weights and
            recurrent matrices; recurrent matrices additionally carry a 0.1
            sub-scale for EqProp's small-recurrent convention)
        init_scheme: "default" (fan-in × init_scale) or "mupc" (μPC
            depth-scaled init, arXiv:2505.13124 — N(0,1) weights, hidden
            1/√(N·L), output 1/N; supersedes init_scale for the
            feedforward stack only — recurrent weight matrices keep the
            EqProp small-recurrent convention (init_scale × 0.1) under
            both schemes, whose stability depends on the 0.1 sub-scale)
        residual: Skip connections between equal-width hidden layers
            (a_ℓ = a_{ℓ−1} + φ(W_ℓ a_{ℓ−1} + b_ℓ)); the paper regime for
            μPC init (Table 1 is specified and tested on residual nets,
            arXiv:2505.13124). Applied only where a Linear is square
            (hidden→hidden); the input and output projections stay
            unscaled single paths
        conv_channels: Output channels per convolution layer (conv topology)
        kernel_size: Convolution kernel edge length (conv topology)
        in_channels: Input channel count (conv topology)
        input_hw: Input spatial extent (height, width) (conv topology)
        pool_hw: Adaptive average-pool grid before the classifier head
            (conv topology)
        # Attention topology fields
        num_heads: int = 8
        head_dim: int | None = None
        attention_dropout: float = 0.0
        # SpatialLattice3D topology fields
        lattice_dims: tuple[int, int, int] = (4, 4, 4)
        connectivity_radius: int = 1
    """

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    num_layers: int
    topology_type: str
    connectivity: dict[str, object] | None
    recurrent_weight: list[list[float]] | None
    init_scale: float = _DEFAULT_INIT_SCALE
    init_scheme: InitScheme = "default"
    residual: bool = False
    conv_channels: tuple[int, ...] = ()
    kernel_size: int = 3
    in_channels: int = 1
    input_hw: tuple[int, int] = (28, 28)
    pool_hw: tuple[int, int] = (4, 4)
    # Attention topology
    num_heads: int = 8
    head_dim: int | None = None
    attention_dropout: float = 0.0
    # SpatialLattice3D topology
    lattice_dims: tuple[int, int, int] = (4, 4, 4)
    connectivity_radius: int = 1

    @classmethod
    def feedforward(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        init_scale: float = 0.1,
        init_scheme: InitScheme = "default",
        residual: bool = False,
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
            init_scheme=init_scheme,
            residual=residual,
        )

    @classmethod
    def recurrent(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        init_scale: float = 0.1,
        init_scheme: InitScheme = "default",
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
            init_scheme=init_scheme,
        )

    @classmethod
    def tile_mesh(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        neurons_per_tile: int,  # ruff: ignore[unused-class-method-argument]
        tiles_per_layer: int,  # ruff: ignore[unused-class-method-argument]
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

    @classmethod
    def conv(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        conv_channels: tuple[int, ...] = (8, 16),
        kernel_size: int = 3,
        in_channels: int = 1,
        input_hw: tuple[int, int] = (28, 28),
        pool_hw: tuple[int, int] = (4, 4),
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(),
            num_layers=len(conv_channels) + 1,
            topology_type="conv",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            in_channels=in_channels,
            input_hw=input_hw,
            pool_hw=pool_hw,
        )

    @classmethod
    def graph(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        edge_index: list[list[int]],
        hidden_dims: tuple[int, ...] = (64, 64),
        init_scale: float = 0.1,
        init_scheme: InitScheme = "default",
    ) -> GeometryConfig:
        """Create a graph topology config.

        Args:
            input_dim: Node feature dimension
            output_dim: Output dimension (num classes for node classification)
            edge_index: Graph connectivity as [2, num_edges] list of lists
            hidden_dims: Hidden dimensions for each GNN layer
            init_scale: Weight initialization scale
            init_scheme: Weight initialization scheme ("default" or "mupc")
        """
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims) + 1,
            topology_type="graph",
            connectivity={"edge_index": edge_index},
            recurrent_weight=None,
            init_scale=init_scale,
            init_scheme=init_scheme,
        )

    @classmethod
    def attention(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        head_dim: int | None = None,
        attention_dropout: float = 0.0,
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        """Create an attention topology config.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Model dimension (d_model)
            num_layers: Number of attention blocks
            num_heads: Number of attention heads
            head_dim: Dimension per head (defaults to hidden_dim // num_heads)
            attention_dropout: Dropout rate for attention weights
            init_scale: Weight initialization scale
        """
        if head_dim is None:
            head_dim = hidden_dim // num_heads
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim,) * num_layers,
            num_layers=num_layers,
            topology_type="attention",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
        )

    @classmethod
    def spatial_lattice(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        lattice_dims: tuple[int, int, int] = (4, 4, 4),
        hidden_dims: tuple[int, ...] = (32, 32),
        connectivity_radius: int = 1,
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        """Create a 3D spatial lattice topology config.

        Args:
            input_dim: Total input dimension
            output_dim: Output dimension
            lattice_dims: 3D lattice dimensions (depth, height, width)
            hidden_dims: Hidden dimensions for each layer (per site)
            connectivity_radius: Neighborhood radius for local connections
            init_scale: Weight initialization scale
        """
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims) + 1,
            topology_type="spatial_lattice",
            connectivity={"lattice_dims": lattice_dims, "radius": connectivity_radius},
            recurrent_weight=None,
            init_scale=init_scale,
            lattice_dims=lattice_dims,
            connectivity_radius=connectivity_radius,
        )


# ============================================================
# Geometry Protocol
# ============================================================


def _set_param_name(tensor: Tensor, name: str) -> None:
    """Tag a parameter tensor with its substrate keying name."""
    setattr(tensor, "_param_name", name)


def layer_stack(geometry: Geometry) -> nn.ModuleList | None:
    """Return the geometry's ordered module stack if it is layer-based."""
    layers = getattr(geometry, "_layers", None)
    return layers if isinstance(layers, nn.ModuleList) else None


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
        self.residual = config.residual
        self._layers = nn.ModuleList(layers) if layers else nn.ModuleList()
        if not self._layers:
            self._build_layers()

    def _build_layers(self) -> None:
        dims = (
            self.config.input_dim,
            *self.config.hidden_dims,
            self.config.output_dim,
        )
        self._layers = nn.ModuleList(
            _linear_stack(dims, self.config.init_scale, self.config.init_scheme)
        )
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

    def _apply_stack(
        self,
        x: Tensor,
        substrate: Substrate | None,
        intermediates: list[Tensor] | None = None,
    ) -> Tensor:
        """Route through the layer stack with optional residual skips.

        Skip arithmetic (when ``self.residual``): a hidden Linear whose
        in/out features match adds its input activity to the activated
        branch output — ``a_ℓ = a_{ℓ−1} + φ(W_ℓ a_{ℓ−1} + b_ℓ)``. The
        input projection and the output readout are never skipped.
        """
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x.flatten(1) if x.dim() > 2 else x
        if intermediates is not None:
            intermediates.append(h)
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h_in = h
                h = op(h, layer.weight)
                if layer.bias is not None:
                    # Out-of-place add: in-place adds on grad-tracking tensors
                    # pin the whole downstream settle graph (CUDA leak)
                    h = h + layer.bias  # ruff: ignore[non-augmented-assignment]
            else:
                h = layer(h)
                if self.residual and h.shape == h_in.shape:
                    h = h + h_in
                if intermediates is not None:
                    # Add after activation function and skip (post-skip
                    # activities align with settle-kernel acts)
                    intermediates.append(h)
        return h

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        return self._apply_stack(x, substrate)

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer.

        Returns:
            List of activations [input, layer1_out, layer2_out, ..., output]
            where layer outputs are after activation functions and any
            residual skip.
        """
        acts: list[Tensor] = []
        h = self._apply_stack(x, substrate, acts)
        # Add final output if last layer was Linear (no trailing activation)
        if self._layers and isinstance(self._layers[-1], nn.Linear):
            acts.append(h)
        return acts

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the geometry (for settling dynamics)."""
        h = activations
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h_in = h
                h = h @ layer.weight.T  # ruff: ignore[non-augmented-assignment]
                if layer.bias is not None:
                    # Out-of-place adds on autograd break
                    h = h + layer.bias  # ruff: ignore[non-augmented-assignment]
            else:
                h = layer(h)
                if self.residual and h.shape == h_in.shape:
                    h = h + h_in  # ruff: ignore[non-augmented-assignment]
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
        self._layers = nn.ModuleList(
            _linear_stack(dims, self.config.init_scale, self.config.init_scheme)
        )
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
        h = x.flatten(1) if x.dim() > 2 else x
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    # Out-of-place: in-place adds break autograd
                    h = h + layer.bias  # ruff: ignore[non-augmented-assignment]
            else:
                h = layer(h)
            # Apply recurrent connection after each hidden layer (except output)
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                # Out-of-place: in-place adds break autograd
                h = h + op(h, self._recurrent_weight)  # ruff: ignore[non-augmented-assignment]
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
                h = h @ self._recurrent_weight.T  # ruff: ignore[non-augmented-assignment]
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
        h = x.flatten(1) if x.dim() > 2 else x
        acts = [h]
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h = h + layer.bias  # ruff: ignore[non-augmented-assignment]
            else:
                h = layer(h)
                # Add after activation functions
                acts.append(h)
            # Apply recurrent connection after each hidden layer (except output)
            # Out-of-place: in-place adds pin the downstream settle graph
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                h = h + op(h, self._recurrent_weight)  # ruff: ignore[non-augmented-assignment]
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

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:  # ruff: ignore[complex-structure]
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

    def update_params(self, new_params: dict[str, Tensor]) -> None:  # ruff: ignore[complex-structure]
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

    def forward_with_intermediates(  # ruff: ignore[complex-structure]
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

    # --- Tile block view (R11.1.4 settle-kernel contract) ---

    @property
    def _block_view(self) -> TileBlockView:
        view = getattr(self, "_block_view_cache", None)
        if view is None:
            view = build_block_view(self._graph)
            self._block_view_cache = view
        return view

    @property
    def block_act_count(self) -> int:
        """Length of the settled block-act layout [x, z_0..z_{L-1}, output]."""
        return self._block_view.block_act_count

    def layered_params(self):
        """Per-transition block weights for the substrate settle kernel."""
        return tile_layered_params(self._block_view, self.params)

    def settle_blocks(self, x: Tensor, substrate: Substrate) -> list[Tensor]:
        """Initial block activities for settling: [x, z_0..z_{L-1}, output]."""
        return settle_block_acts(self._block_view, self, x, substrate)

    def assemble_blocks(self, named: dict[str, Tensor]) -> tuple[Tensor, ...]:
        """Per-transition block matrices from a name->tensor mapping."""
        return assemble_transition_blocks(self._block_view, named)

    def scatter_block_grads(self, block_grads: list[Tensor]) -> list[Tensor]:
        """Scatter per-transition block pseudo-gradients to per-edge parameters."""
        from computronium.ontology.utils import _learnable_weight_names

        return _scatter_block_grads(
            self._block_view, block_grads, _learnable_weight_names(self.params)
        )

    def hopfield_energy(self, acts: list[Tensor]) -> Tensor:
        """Hopfield energy over the block layout [x, z_0..z_{L-1}, output]."""
        return tile_hopfield_energy(self._block_view, acts, self.params)


class ConvGeometry(nn.Module):
    """Convolutional topology: shared kernels routed through the substrate operator.

    Each conv layer is a flat kernel matrix applied to im2col patches
    (``unfold → op(patches, W_i) → fold``), so substrate physics stay in
    the loop exactly as for dense routing. An adaptive average pool fixes
    the classifier head's input extent.
    """

    _conv_weights: nn.ParameterDict
    _conv_biases: nn.ParameterDict
    _head: nn.Linear

    def __init__(self, config: GeometryConfig):
        if config.kernel_size % 2 == 0:
            raise ValueError(
                f"ConvGeometry requires an odd kernel_size, got {config.kernel_size}"
            )
        super().__init__()
        self.config = config
        self._conv_weights = nn.ParameterDict()
        self._conv_biases = nn.ParameterDict()
        c_in = config.in_channels
        for i in range(len(config.conv_channels)):
            c_out = config.conv_channels[i]
            fan_in = c_in * config.kernel_size**2
            weight = nn.Parameter(torch.randn(c_out, fan_in) * (1.0 / fan_in) ** 0.5)
            if config.init_scale != _DEFAULT_INIT_SCALE:
                weight.data.mul_(config.init_scale / _DEFAULT_INIT_SCALE)
            self._conv_weights[f"layer_{i}"] = weight
            self._conv_biases[f"layer_{i}"] = nn.Parameter(torch.zeros(c_out))
            c_in = c_out
        self._head = nn.Linear(
            c_in * config.pool_hw[0] * config.pool_hw[1], config.output_dim
        )
        if config.init_scale != _DEFAULT_INIT_SCALE:
            self._head.weight.data.mul_(config.init_scale / _DEFAULT_INIT_SCALE)
        self._set_param_names()

    def _set_param_names(self) -> None:
        for key, weight in self._conv_weights.items():
            _set_param_name(weight, f"{key}_weight")
            _set_param_name(self._conv_biases[key], f"{key}_bias")
        _set_param_name(self._head.weight, "head_weight")
        if self._head.bias is not None:
            _set_param_name(self._head.bias, "head_bias")

    @property
    def params(self) -> dict[str, Tensor]:
        params: dict[str, Tensor] = {}
        for key, weight in self._conv_weights.items():
            params[f"{key}_weight"] = weight
            params[f"{key}_bias"] = self._conv_biases[key]
        params["head_weight"] = self._head.weight
        params["head_bias"] = self._head.bias
        return params

    def _im2col(self, x: Tensor) -> tuple[Tensor, int]:
        """Reshape (B, C, H, W) input into flat patches for the operator."""
        k = self.config.kernel_size
        p = k // 2
        patches = nn.functional.unfold(x, k, stride=1, padding=p)
        b, _, n = patches.shape
        return patches.transpose(1, 2).reshape(b * n, -1), n

    def _stack(self, x: Tensor, op: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        c = self.config.in_channels
        h, w = self.config.input_hw
        if x.dim() == 2:
            x = x.view(-1, c, h, w)
        side = h
        for i, _ in enumerate(self.config.conv_channels):
            patches, n = self._im2col(x)
            out = op(patches, self._conv_weights[f"layer_{i}"])
            # Out-of-place: in-place adds pin the downstream settle graph
            out = out + self._conv_biases[f"layer_{i}"]  # ruff: ignore[non-augmented-assignment]
            x = torch.relu(out.view(x.shape[0], n, -1).transpose(1, 2))
            x = x.reshape(x.shape[0], -1, side, side)
        return nn.functional.adaptive_avg_pool2d(x, self.config.pool_hw).flatten(1)

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        features = self._stack(x, op)
        return op(features, self._head.weight) + self._head.bias

    def route(self, activations: Tensor) -> Tensor:
        features = self._stack(activations, lambda a, w: a @ w.T)
        return features @ self._head.weight.T + self._head.bias

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        c = self.config.in_channels
        h, w = self.config.input_hw
        acts = [x.flatten(1) if x.dim() > 2 else x]
        if x.dim() == 2:
            x = x.view(-1, c, h, w)
        side = h
        for i, _ in enumerate(self.config.conv_channels):
            patches, n = self._im2col(x)
            out = op(patches, self._conv_weights[f"layer_{i}"])
            # Out-of-place: in-place adds pin the downstream settle graph
            out = out + self._conv_biases[f"layer_{i}"]  # ruff: ignore[non-augmented-assignment]
            x = torch.relu(out.view(x.shape[0], n, -1).transpose(1, 2))
            x = x.reshape(x.shape[0], -1, side, side)
            acts.append(x.flatten(1))
        pooled = nn.functional.adaptive_avg_pool2d(x, self.config.pool_hw).flatten(1)
        acts.append(op(pooled, self._head.weight) + self._head.bias)
        return acts

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            if name.endswith("_weight") and name.startswith("layer_"):
                self._conv_weights[name[: -len("_weight")]].data.copy_(param)
            elif name.endswith("_bias") and name.startswith("layer_"):
                self._conv_biases[name[: -len("_bias")]].data.copy_(param)
            elif name == "head_weight":
                self._head.weight.data.copy_(param)
            elif name == "head_bias":
                self._head.bias.data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [self._head]


class GraphGeometry(nn.Module):
    """Graph topology: message passing over an edge index.

    Implements a multi-layer Graph Neural Network where each layer performs:
        1. Feature transformation via the substrate's forward operator
        2. Neighborhood aggregation (mean pooling over neighbors)

    The edge_index is provided via config (serialized as nested lists for JSON
    compatibility). Multiple layers are stacked with ReLU activations between.
    """

    _layer_weights: nn.ParameterDict
    _layer_biases: nn.ParameterDict
    _head: nn.Linear

    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config
        self._layer_weights = nn.ParameterDict()
        self._layer_biases = nn.ParameterDict()

        # Parse edge_index from config (stored as list[list[int]])
        if config.connectivity is None or "edge_index" not in config.connectivity:
            raise ValueError(
                "GraphGeometry requires config.connectivity with 'edge_index' key"
            )
        edge_index = config.connectivity["edge_index"]
        self._edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Build layers
        c_in = config.input_dim
        hidden_dims = config.hidden_dims if config.hidden_dims else (64,)
        width = hidden_dims[0]
        num_hidden = len(hidden_dims)
        for i, c_out in enumerate(hidden_dims):
            weight = nn.Parameter(torch.randn(c_out, c_in))
            if config.init_scheme == "mupc":
                nn.init.normal_(weight)
                weight.data.mul_(1.0 / (width * num_hidden) ** 0.5)
            else:
                weight.data.mul_((1.0 / c_in) ** 0.5)
                if config.init_scale != _DEFAULT_INIT_SCALE:
                    weight.data.mul_(config.init_scale / _DEFAULT_INIT_SCALE)
            self._layer_weights[f"layer_{i}"] = weight
            self._layer_biases[f"layer_{i}"] = nn.Parameter(torch.zeros(c_out))
            c_in = c_out

        # Classifier head
        self._head = nn.Linear(c_in, config.output_dim)
        if config.init_scheme == "mupc":
            nn.init.normal_(self._head.weight)
            self._head.weight.data.mul_(1.0 / width)
        elif config.init_scale != _DEFAULT_INIT_SCALE:
            self._head.weight.data.mul_(config.init_scale / _DEFAULT_INIT_SCALE)

        self._set_param_names()
        self._compute_degrees()

    def _set_param_names(self) -> None:
        for key, weight in self._layer_weights.items():
            _set_param_name(weight, f"{key}_weight")
            _set_param_name(self._layer_biases[key], f"{key}_bias")
        _set_param_name(self._head.weight, "head_weight")
        if self._head.bias is not None:
            _set_param_name(self._head.bias, "head_bias")

    def _compute_degrees(self) -> None:
        """Pre-compute neighbor counts for mean aggregation."""
        num_nodes = int(self._edge_index.max().item()) + 1
        row, _ = self._edge_index
        deg = torch.zeros(num_nodes, dtype=torch.long)
        deg.scatter_add_(0, row, torch.ones_like(row))
        self.register_buffer("_deg", deg.clamp(min=1), persistent=False)

    @property
    def num_nodes(self) -> int:
        return int(self._edge_index.max().item()) + 1

    def node_depths(self, metric: DepthMetric) -> Tensor:
        """Per-node effective depth under ``metric`` (R11.3.13).

        Depth-scaled (μPC) studies on graph topologies key off this instead
        of layer counts, which understate path length on non-layered graphs.
        """
        return metric.per_node(self.num_nodes)

    def _move_edge_index(self, device: torch.device) -> None:
        """Move edge_index and degree buffer to device."""
        if self._edge_index.device != device:
            self._edge_index = self._edge_index.to(device)
        if hasattr(self, "_deg") and self._deg.device != device:
            self._deg = self._deg.to(device)

    @property
    def params(self) -> dict[str, Tensor]:
        params: dict[str, Tensor] = {}
        for key, weight in self._layer_weights.items():
            params[f"{key}_weight"] = weight
            params[f"{key}_bias"] = self._layer_biases[key]
        params["head_weight"] = self._head.weight
        params["head_bias"] = self._head.bias
        return params

    def _aggregate(self, h: Tensor) -> Tensor:
        """Mean aggregation over neighbors: h_out[i] = mean(h[j] for j in N(i))."""
        row, col = self._edge_index
        # h[col] gives source node features for each edge
        # scatter_add aggregates to destination nodes (row)
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, h[col])
        return aggr / self._deg.unsqueeze(1).to(h.dtype)

    def _stack(self, x: Tensor, op: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        """Apply stacked graph layers with substrate operator for feature transform."""
        self._move_edge_index(x.device)
        h = x
        for i in range(len(self._layer_weights)):
            weight = self._layer_weights[f"layer_{i}"]
            bias = self._layer_biases[f"layer_{i}"]
            # Feature transformation via substrate operator
            h = op(h, weight)
            # Out-of-place: in-place adds pin the downstream settle graph
            h = h + bias  # ruff: ignore[non-augmented-assignment]
            # Non-linearity
            h = torch.relu(h)
            # Neighborhood aggregation
            h = self._aggregate(h)
        return h

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = self._stack(x, op)
        return op(h, self._head.weight) + self._head.bias

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the graph (for settling dynamics)."""
        h = self._stack(activations, lambda a, w: a @ w.T)
        return h @ self._head.weight.T + self._head.bias

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        self._move_edge_index(x.device)

        acts = [x]
        h = x
        for i in range(len(self._layer_weights)):
            weight = self._layer_weights[f"layer_{i}"]
            bias = self._layer_biases[f"layer_{i}"]
            h = op(h, weight)
            h = h + bias  # ruff: ignore[non-augmented-assignment]
            h = torch.relu(h)
            h = self._aggregate(h)
            acts.append(h)

        out = op(h, self._head.weight) + self._head.bias
        acts.append(out)
        return acts

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            if name.endswith("_weight") and name.startswith("layer_"):
                self._layer_weights[name[: -len("_weight")]].data.copy_(param)
            elif name.endswith("_bias") and name.startswith("layer_"):
                self._layer_biases[name[: -len("_bias")]].data.copy_(param)
            elif name == "head_weight":
                self._head.weight.data.copy_(param)
            elif name == "head_bias":
                self._head.bias.data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [self._head]


class _AttentionBlock(nn.Module):
    """One pre-norm transformer block: MHA + FFN, substrate-routed projections."""

    q_weight: nn.Parameter
    k_weight: nn.Parameter
    v_weight: nn.Parameter
    o_weight: nn.Parameter
    ffn1_weight: nn.Parameter
    ffn2_weight: nn.Parameter
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(self, h: int, init_scale: float):
        super().__init__()
        scale = init_scale / _DEFAULT_INIT_SCALE
        uniform = 1.0 / h**0.5
        self.q_weight = nn.Parameter(torch.randn(h, h) * uniform)
        self.k_weight = nn.Parameter(torch.randn(h, h) * uniform)
        self.v_weight = nn.Parameter(torch.randn(h, h) * uniform)
        self.o_weight = nn.Parameter(torch.randn(h, h) * uniform)
        ffn_hidden = h * 4
        self.ffn1_weight = nn.Parameter(torch.randn(ffn_hidden, h) * uniform)
        self.ffn2_weight = nn.Parameter(
            torch.randn(h, ffn_hidden) * (1.0 / ffn_hidden) ** 0.5
        )
        if init_scale != _DEFAULT_INIT_SCALE:
            for param in self.parameters():
                param.data.mul_(scale)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)


class AttentionGeometry(nn.Module):
    """Attention topology: multi-head self-attention blocks.

    Implements a stacked transformer-style architecture where each block consists of:
    1. Multi-head self-attention with substrate-routed projections
    2. Feed-forward network (MLP) with substrate-routed projections
    3. Residual connections and layer normalization

    The attention projections (Q, K, V, O) and FFN weights are routed through
    the substrate's forward operator, keeping substrate physics in the loop.
    """

    _blocks: nn.ModuleList
    _input_projection: nn.Linear
    _output_projection: nn.Linear
    _num_heads: int
    _head_dim: int
    _hidden_dim: int
    _dropout: float

    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config
        self._num_heads = config.num_heads
        self._head_dim = config.head_dim or (config.hidden_dims[0] // config.num_heads)
        self._hidden_dim = (
            config.hidden_dims[0] if config.hidden_dims else config.input_dim
        )
        self._dropout = config.attention_dropout

        # Input projection to hidden_dim
        self._input_projection = nn.Linear(config.input_dim, self._hidden_dim)
        if config.init_scale != _DEFAULT_INIT_SCALE:
            self._input_projection.weight.data.mul_(
                config.init_scale / _DEFAULT_INIT_SCALE
            )

        # Build attention blocks
        self._blocks = nn.ModuleList()
        for i in range(config.num_layers):
            block = self._build_attention_block(i, config)
            self._blocks.append(block)

        # Output projection
        self._output_projection = nn.Linear(self._hidden_dim, config.output_dim)
        if config.init_scale != _DEFAULT_INIT_SCALE:
            self._output_projection.weight.data.mul_(
                config.init_scale / _DEFAULT_INIT_SCALE
            )

        self._set_param_names()

    @property
    def _typed_blocks(self) -> list[_AttentionBlock]:
        return list(self._blocks)  # type: ignore[arg-type]

    def _build_attention_block(
        self, block_idx: int, config: GeometryConfig
    ) -> _AttentionBlock:
        """Build a single attention block with MHA + FFN."""
        block = _AttentionBlock(self._hidden_dim, config.init_scale)
        _set_param_name(block.q_weight, f"block_{block_idx}_q_weight")
        _set_param_name(block.k_weight, f"block_{block_idx}_k_weight")
        _set_param_name(block.v_weight, f"block_{block_idx}_v_weight")
        _set_param_name(block.o_weight, f"block_{block_idx}_o_weight")
        _set_param_name(block.ffn1_weight, f"block_{block_idx}_ffn1_weight")
        _set_param_name(block.ffn2_weight, f"block_{block_idx}_ffn2_weight")
        return block

    def _set_param_names(self) -> None:
        _set_param_name(self._input_projection.weight, "input_proj_weight")
        if self._input_projection.bias is not None:
            _set_param_name(self._input_projection.bias, "input_proj_bias")
        _set_param_name(self._output_projection.weight, "output_proj_weight")
        if self._output_projection.bias is not None:
            _set_param_name(self._output_projection.bias, "output_proj_bias")

    @property
    def params(self) -> dict[str, Tensor]:
        params = {}
        # Input/output projections
        for name, param in self._input_projection.named_parameters():
            params[f"input_proj.{name}"] = param
        for name, param in self._output_projection.named_parameters():
            params[f"output_proj.{name}"] = param
        # Attention blocks
        for i, block in enumerate(self._typed_blocks):
            for key in (
                "q_weight",
                "k_weight",
                "v_weight",
                "o_weight",
                "ffn1_weight",
                "ffn2_weight",
            ):
                params[f"block_{i}_{key}"] = getattr(block, key)
        return params

    def _multi_head_attention(
        self,
        x: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        op: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Compute multi-head attention with substrate operator."""
        b, n, h = x.shape
        nh = self._num_heads
        hd = self._head_dim

        # Project to Q, K, V
        q = op(x.reshape(-1, h), q_weight).view(b, n, nh, hd).transpose(1, 2)
        k = op(x.reshape(-1, h), k_weight).view(b, n, nh, hd).transpose(1, 2)
        v = op(x.reshape(-1, h), v_weight).view(b, n, nh, hd).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (hd**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self._dropout > 0 and self.training:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self._dropout)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, h)

        # Output projection
        return op(attn_out.reshape(-1, h), o_weight).view(b, n, h)

    def _ffn(
        self,
        x: Tensor,
        ffn1_weight: Tensor,
        ffn2_weight: Tensor,
        op: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Feed-forward network with substrate operator."""
        b, n, h = x.shape
        # First linear + ReLU
        hidden = torch.relu(op(x.reshape(-1, h), ffn1_weight))
        # Second linear
        out = op(hidden, ffn2_weight).view(b, n, h)
        return out

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Flatten input if needed (e.g., [B, C, H, W] -> [B, C*H*W])
        h = x.flatten(1) if x.dim() > 2 else x

        # Input projection
        h = op(h, self._input_projection.weight)
        if self._input_projection.bias is not None:
            h = h + self._input_projection.bias  # ruff: ignore[non-augmented-assignment]
        # Treat flat input as single token: [B, H] -> [B, 1, H]
        h = h.unsqueeze(1) if h.dim() == 2 else h  # [B, 1, H] for single token

        # Pass through attention blocks
        for block in self._typed_blocks:
            # Attention with residual
            residual = h
            h = block.ln1(h)
            attn_out = self._multi_head_attention(
                h,
                block.q_weight,
                block.k_weight,
                block.v_weight,
                block.o_weight,
                op,
            )
            h = residual + attn_out

            # FFN with residual
            residual = h
            h = block.ln2(h)
            ffn_out = self._ffn(h, block.ffn1_weight, block.ffn2_weight, op)
            h = residual + ffn_out

        # Output projection (take first/last token or mean pool)
        h = h.mean(dim=1) if h.dim() == 3 else h  # [B, H]
        out = op(h, self._output_projection.weight)
        if self._output_projection.bias is not None:
            out = out + self._output_projection.bias  # ruff: ignore[non-augmented-assignment]
        return out

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through attention blocks (for settling dynamics)."""
        # For route, we use direct matmul instead of substrate operator

        def op(a: Tensor, w: Tensor) -> Tensor:
            return a @ w.T

        h = activations
        if h.dim() == 2:
            h = h.unsqueeze(1)
        for block in self._typed_blocks:
            residual = h
            h = block.ln1(h)
            attn_out = self._multi_head_attention(
                h,
                block.q_weight,
                block.k_weight,
                block.v_weight,
                block.o_weight,
                op,
            )
            h = residual + attn_out

            residual = h
            h = block.ln2(h)
            ffn_out = self._ffn(h, block.ffn1_weight, block.ffn2_weight, op)
            h = residual + ffn_out

        h = h.mean(dim=1) if h.dim() == 3 else h
        out = h @ self._output_projection.weight.T
        if self._output_projection.bias is not None:
            out = out + self._output_projection.bias  # ruff: ignore[non-augmented-assignment]
        return out

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Flatten input if needed (e.g., [B, C, H, W] -> [B, C*H*W])
        h = x.flatten(1) if x.dim() > 2 else x
        h = op(h, self._input_projection.weight)
        if self._input_projection.bias is not None:
            h = h + self._input_projection.bias  # ruff: ignore[non-augmented-assignment]
        h = h.unsqueeze(1) if h.dim() == 2 else h
        acts = [h.squeeze(1) if h.shape[1] == 1 else h]

        for block in self._typed_blocks:
            residual = h
            h = block.ln1(h)
            attn_out = self._multi_head_attention(
                h,
                block.q_weight,
                block.k_weight,
                block.v_weight,
                block.o_weight,
                op,
            )
            h = residual + attn_out
            acts.append(h.mean(dim=1) if h.dim() == 3 else h)

            residual = h
            h = block.ln2(h)
            ffn_out = self._ffn(h, block.ffn1_weight, block.ffn2_weight, op)
            h = residual + ffn_out
            acts.append(h.mean(dim=1) if h.dim() == 3 else h)

        h = h.mean(dim=1) if h.dim() == 3 else h
        out = op(h, self._output_projection.weight)
        if self._output_projection.bias is not None:
            out = out + self._output_projection.bias  # ruff: ignore[non-augmented-assignment]
        acts.append(out)
        return acts

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            if name.startswith("input_proj."):
                pname = name.replace("input_proj.", "")
                if hasattr(self._input_projection, pname):
                    getattr(self._input_projection, pname).data.copy_(param)
            elif name.startswith("output_proj."):
                pname = name.replace("output_proj.", "")
                if hasattr(self._output_projection, pname):
                    getattr(self._output_projection, pname).data.copy_(param)
            elif name.startswith("block_"):
                parts = name.split("_")
                if len(parts) >= 3:
                    block_idx = int(parts[1])
                    param_name = "_".join(parts[2:])
                    if block_idx < len(self._blocks) and hasattr(
                        self._blocks[block_idx], param_name
                    ):
                        getattr(self._blocks[block_idx], param_name).data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self._input_projection, self._output_projection]
        for block in self._typed_blocks:
            modules.extend([block.ln1, block.ln2])
        return modules


class SpatialLattice3DGeometry(nn.Module):
    """3D Spatial Lattice topology: local connectivity on a 3D grid.

    Implements a neural cube where each site connects to its neighbors within
    a radius. Uses message passing similar to GraphGeometry but with
    structured 3D lattice connectivity. The substrate operator is applied
    for feature transformations at each site.
    """

    _site_weights: nn.ParameterDict
    _site_biases: nn.ParameterDict
    _input_projection: nn.Linear
    _output_projection: nn.Linear
    _lattice_dims: tuple[int, int, int]
    _num_sites: int
    _radius: int
    _neighbors: dict[int, list[int]]  # precomputed neighbor lists

    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config
        self._lattice_dims = config.lattice_dims
        self._radius = config.connectivity_radius
        d, h, w = self._lattice_dims
        self._num_sites = d * h * w

        # Input projection: flatten lattice sites into feature vectors
        first_hidden = (
            config.hidden_dims[0] if config.hidden_dims else config.output_dim
        )
        self._input_projection = nn.Linear(
            config.input_dim, self._num_sites * first_hidden
        )
        if config.init_scale != _DEFAULT_INIT_SCALE:
            self._input_projection.weight.data.mul_(
                config.init_scale / _DEFAULT_INIT_SCALE
            )

        # Build site-specific weights for each layer
        self._site_weights = nn.ParameterDict()
        self._site_biases = nn.ParameterDict()

        # Number of hidden layers = len(hidden_dims)
        # Each layer transforms per-site features
        num_hidden_layers = len(config.hidden_dims)
        for layer_idx in range(num_hidden_layers):
            c_in = first_hidden if layer_idx == 0 else config.hidden_dims[layer_idx - 1]
            c_out = config.hidden_dims[layer_idx]

            for site in range(self._num_sites):
                weight = nn.Parameter(torch.randn(c_out, c_in) * (1.0 / c_in) ** 0.5)
                if config.init_scale != _DEFAULT_INIT_SCALE:
                    weight.data.mul_(config.init_scale / _DEFAULT_INIT_SCALE)
                self._site_weights[f"layer_{layer_idx}_site_{site}"] = weight
                bias = nn.Parameter(torch.zeros(c_out))
                self._site_biases[f"layer_{layer_idx}_site_{site}"] = bias

        # Output projection: maps flattened site features to output_dim
        # Always needed when we have multiple sites and want single output vector
        final_features = self._num_sites * config.hidden_dims[-1]
        self._output_projection = nn.Linear(final_features, config.output_dim)
        if config.init_scale != _DEFAULT_INIT_SCALE:
            self._output_projection.weight.data.mul_(
                config.init_scale / _DEFAULT_INIT_SCALE
            )

        # Precompute neighbor lists for each site
        self._compute_neighbors()

        self._set_param_names()

    def _compute_neighbors(self) -> None:
        """Precompute neighbor indices for each lattice site."""
        d, h, w = self._lattice_dims
        self._neighbors = {}

        for dz in range(d):  # ruff: ignore[too-many-nested-blocks]
            for dy in range(h):
                for dx in range(w):
                    site = (dz * h + dy) * w + dx
                    neighbors = []
                    for ndz in range(
                        max(0, dz - self._radius), min(d, dz + self._radius + 1)
                    ):
                        for ndy in range(
                            max(0, dy - self._radius), min(h, dy + self._radius + 1)
                        ):
                            for ndx in range(
                                max(0, dx - self._radius), min(w, dx + self._radius + 1)
                            ):
                                if ndz == dz and ndy == dy and ndx == dx:
                                    continue
                                nbr = (ndz * h + ndy) * w + ndx
                                neighbors.append(nbr)
                    self._neighbors[site] = neighbors

    def _set_param_names(self) -> None:
        if self._input_projection is not None:
            _set_param_name(self._input_projection.weight, "input_proj_weight")
            if self._input_projection.bias is not None:
                _set_param_name(self._input_projection.bias, "input_proj_bias")
        if self._output_projection is not None:
            _set_param_name(self._output_projection.weight, "output_proj_weight")
            if self._output_projection.bias is not None:
                _set_param_name(self._output_projection.bias, "output_proj_bias")
        for key, weight in self._site_weights.items():
            _set_param_name(weight, f"{key}_weight")
            _set_param_name(self._site_biases[key], f"{key}_bias")

    @property
    def params(self) -> dict[str, Tensor]:
        params = {}
        if self._input_projection is not None:
            for name, param in self._input_projection.named_parameters():
                params[f"input_proj.{name}"] = param
        if self._output_projection is not None:
            for name, param in self._output_projection.named_parameters():
                params[f"output_proj.{name}"] = param
        params.update({f"{k}_weight": v for k, v in self._site_weights.items()})
        params.update({f"{k}_bias": v for k, v in self._site_biases.items()})
        return params

    def _propagate_layer(
        self,
        x: Tensor,  # [B, num_sites, c_in]
        layer_idx: int,
        op: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Apply one layer of site-wise transformations with neighbor aggregation."""
        b, n, c_in = x.shape
        c_out = self._site_weights[f"layer_{layer_idx}_site_0"].shape[0]
        out = torch.zeros(b, n, c_out, device=x.device, dtype=x.dtype)

        for site in range(n):
            # Aggregate neighbors + self
            nbrs = self._neighbors[site]
            if nbrs:
                nbr_acts = x[:, nbrs, :]  # [B, num_nbrs, c_in]
                # Mean aggregation over neighbors
                agg = nbr_acts.mean(dim=1)
            else:
                agg = torch.zeros(b, c_in, device=x.device, dtype=x.dtype)
            # Include self
            agg = agg + x[:, site, :]  # ruff: ignore[non-augmented-assignment]

            # Transform via substrate operator
            weight = self._site_weights[f"layer_{layer_idx}_site_{site}"]
            bias = self._site_biases[f"layer_{layer_idx}_site_{site}"]
            out[:, site, :] = op(agg, weight) + bias
            out[:, site, :] = torch.relu(out[:, site, :])

        return out

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Flatten input if needed (e.g., [B, C, H, W] -> [B, C*H*W])
        h = x.flatten(1) if x.dim() > 2 else x

        # Input projection and reshape to [B, num_sites, hidden_dim]
        h = self._input_projection(h)
        h = h.view(h.shape[0], self._num_sites, -1)

        # Pass through lattice layers
        for layer_idx in range(len(self.config.hidden_dims)):
            h = self._propagate_layer(h, layer_idx, op)

        # Flatten and project to output
        h = h.flatten(1)
        h = op(h, self._output_projection.weight)
        if self._output_projection.bias is not None:
            h = h + self._output_projection.bias  # ruff: ignore[non-augmented-assignment]
        return h

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through lattice (for settling dynamics)."""
        b, n, c_in = activations.shape
        h = activations

        for layer_idx in range(len(self.config.hidden_dims)):
            c_out = self._site_weights[f"layer_{layer_idx}_site_0"].shape[0]
            out = torch.zeros(b, n, c_out, device=h.device, dtype=h.dtype)

            for site in range(n):
                nbrs = self._neighbors[site]
                if nbrs:
                    nbr_acts = h[:, nbrs, :]
                    agg = nbr_acts.mean(dim=1)
                else:
                    agg = torch.zeros(b, c_in, device=h.device, dtype=h.dtype)
                agg = agg + h[:, site, :]  # ruff: ignore[non-augmented-assignment]

                weight = self._site_weights[f"layer_{layer_idx}_site_{site}"]
                bias = self._site_biases[f"layer_{layer_idx}_site_{site}"]
                out[:, site, :] = agg @ weight.T + bias
                out[:, site, :] = torch.relu(out[:, site, :])

            h = out
            c_in = c_out

        # Flatten and project to output
        h = h.flatten(1)
        h = h @ self._output_projection.weight.T  # ruff: ignore[non-augmented-assignment]
        if self._output_projection.bias is not None:
            h = h + self._output_projection.bias  # ruff: ignore[non-augmented-assignment]
        return h

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        if substrate is None:
            from computronium.ontology.substrate import DigitalSubstrate

            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Flatten input if needed (e.g., [B, C, H, W] -> [B, C*H*W])
        h = x.flatten(1) if x.dim() > 2 else x

        h = self._input_projection(h)
        h = h.view(h.shape[0], self._num_sites, -1)
        acts = [h.flatten(1)]

        for layer_idx in range(len(self.config.hidden_dims)):
            h = self._propagate_layer(h, layer_idx, op)
            acts.append(h.flatten(1))

        h = h.flatten(1)
        h = op(h, self._output_projection.weight)
        if self._output_projection.bias is not None:
            h = h + self._output_projection.bias  # ruff: ignore[non-augmented-assignment]
        acts.append(h)
        return acts

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            if name.startswith("input_proj."):
                pname = name.replace("input_proj.", "")
                if hasattr(self._input_projection, pname):
                    getattr(self._input_projection, pname).data.copy_(param)
            elif name.startswith("output_proj."):
                pname = name.replace("output_proj.", "")
                if hasattr(self._output_projection, pname):
                    getattr(self._output_projection, pname).data.copy_(param)
            elif name.endswith("_weight") and name.startswith("layer_"):
                key = name[: -len("_weight")]
                if key in self._site_weights:
                    self._site_weights[key].data.copy_(param)
            elif name.endswith("_bias") and name.startswith("layer_"):
                key = name[: -len("_bias")]
                if key in self._site_biases:
                    self._site_biases[key].data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        modules = []
        if self._input_projection is not None:
            modules.append(self._input_projection)
        if self._output_projection is not None:
            modules.append(self._output_projection)
        return modules


# ============================================================
# Geometry Dispatch
# ============================================================


def geometry_from_config(config: GeometryConfig) -> Geometry:  # ruff: ignore[too-many-return-statements]
    """Instantiate the geometry implementation named by ``config.topology_type``."""
    topology_type = config.topology_type.lower()
    if topology_type in ("recurrent", "recurrent_attractor"):  # ruff: ignore[literal-membership]
        hidden_dim = config.hidden_dims[-1] if config.hidden_dims else None
        recurrent_weight = None
        if config.recurrent_weight is not None:
            recurrent_weight = torch.tensor(config.recurrent_weight)
        return RecurrentGeometry(
            config, hidden_dim=hidden_dim, recurrent_weight=recurrent_weight
        )
    if topology_type in ("tile_mesh", "tile"):  # ruff: ignore[literal-membership]
        return TileGeometry(config, neurons_per_tile=8, tiles_per_layer=2)
    if topology_type == "conv":
        return ConvGeometry(config)
    if topology_type == "graph":
        return GraphGeometry(config)
    if topology_type == "attention":
        return AttentionGeometry(config)
    if topology_type == "spatial_lattice":
        return SpatialLattice3DGeometry(config)
    if topology_type == "feedforward":
        return FeedforwardGeometry(config)
    raise ValueError(f"Unknown topology_type: {topology_type!r}")
