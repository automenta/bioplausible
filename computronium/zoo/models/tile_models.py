"""Tile substrate model classes: PC, TargetProp, SNN, and GNN.

Thin model classes that configure the generic :class:`~computronium.core.local_learning.
TileAlgorithm` substrate for each algorithm family. They inject algorithm-specific
dynamics (inverse-network feedback for TP, spiking reset for SNN, etc.) while
reusing the topology, optimizers, and contrastive loop from the substrate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from computronium.core.local_learning import (
    TaskHandler,
    TileAlgorithm,
    TileAlgorithmConfig,
    WeightLookup,
)
from computronium.core.local_learning.weight_update import contrastive_weight_update
from computronium.core.local_learning.activity import ep_activity_update, spiking_activity_update
from computronium.core.local_learning.feedback import symmetric_feedback
from computronium.core.tile.kernels import compute_activity_update
from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model

if TYPE_CHECKING:
    from computronium.core.tile import TileGraph, TileState

__all__ = [
    "TileGNN",
    "TilePC",
    "TileSNN",
    "TileTargetProp",
]


# ──────────────────────────────────────────────────────────────────────────────
# TilePC — Predictive Coding on tiles
# ──────────────────────────────────────────────────────────────────────────────


@register_model(
    "tile_pc",
    family="tile",
    locality_level=LocalityLevel.LOCAL,
    tags=["predictive-coding", "tile", status_tag("experimental")],
)
class TilePC(TileAlgorithm):
    """Predictive Coding on the tile substrate.

    Uses symmetric feedback and EP-style activity settling (prediction-error
    relaxation). The output head is nudged toward targets during the nudged
    phase, driving prediction-error minimization via contrastive updates.
    """

    _algorithm = "pc"

    @classmethod
    def from_pc(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        beta: float = 0.1,
        **kwargs,
    ) -> TilePC:
        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm="pc",
            mode="ep",
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            beta=beta,
            extra=kwargs,
        )
        return cls(config)

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: str = "cpu",
        task_type: str = "vision",
        **kwargs,
    ) -> TilePC:
        return cls.from_pc(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            importance_lr=float(kwargs.get("importance_lr", 0.01)),
        ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# TileTargetProp — Target Propagation on tiles
# ──────────────────────────────────────────────────────────────────────────────


@register_model(
    "tile_target_prop",
    family="tile",
    locality_level=LocalityLevel.LOCAL,
    tags=["target-prop", "tile", status_tag("experimental")],
)
class TileTargetProp(TileAlgorithm):
    """Target Propagation on the tile substrate.

    Injects an inverse-network :meth:`FeedbackFn` that projects output
    targets backward through learned inverse maps (not weight transposes),
    which is the defining property of target propagation.
    """

    _algorithm = "tp"

    def __init__(self, config: TileAlgorithmConfig, **kwargs) -> None:
        handler = kwargs.pop("task_handler", None) or TaskHandler(
            task_type="classification", output_dim=config.output_dim
        )
        super().__init__(
            config,
            task_handler=handler,
            feedback_fn=self._tp_feedback_bound,
            activity_fn=_ep_activity_update,
            weight_fn=_contrastive_weight_update,
            **kwargs,
        )
        self._inverse_weights: nn.ParameterDict = nn.ParameterDict()
        self._init_inverse_weights()

    def _init_inverse_weights(self) -> None:
        for src_id, dst_id in self.graph.edges:
            src = self.graph.tiles[src_id]
            dst = self.graph.tiles[dst_id]
            inv = torch.randn(src.neurons, dst.neurons) * 0.1
            self._inverse_weights[self._weight_key(src_id, dst_id)] = nn.Parameter(
                inv, requires_grad=True
            )

    def _tp_feedback_bound(
        self, tile: TileState, graph: TileGraph, lookup: WeightLookup
    ) -> list[Tensor]:
        """Project downstream errors through learned inverse networks."""
        if tile.activity is None:
            return []
        feedback: list[Tensor] = []
        for dst_id in tile.fwd_neighbors:
            dst = graph.tiles[dst_id]
            if dst.error is None:
                continue
            inv = self._inverse_weights[self._weight_key(tile.id, dst_id)]
            feedback.append(dst.error @ inv)
        return feedback

    @classmethod
    def from_tp(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        beta: float = 0.1,
        **kwargs,
    ) -> TileTargetProp:
        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm="tp",
            mode="ep",
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            beta=beta,
            extra=kwargs,
        )
        return cls(config)

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: str = "cpu",
        task_type: str = "vision",
        **kwargs,
    ) -> TileTargetProp:
        return cls.from_tp(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            importance_lr=float(kwargs.get("importance_lr", 0.01)),
        ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# TileSNN — Spiking Neural Network on tiles
# ──────────────────────────────────────────────────────────────────────────────


@register_model(
    "tile_snn",
    family="tile",
    locality_level=LocalityLevel.LOCAL,
    tags=["spiking", "tile", status_tag("experimental")],
)
class TileSNN(TileAlgorithm):
    """Spiking Neural Network on the tile substrate.

    Uses a spike-and-reset activity update: neurons integrate prediction error
    and feedback, fire when exceeding a threshold, then reset to zero.
    """

    _algorithm = "snn"

    def __init__(self, config: TileAlgorithmConfig, **kwargs) -> None:
        handler = TaskHandler(task_type="classification", output_dim=config.output_dim)
        super().__init__(
            config,
            task_handler=handler,
            feedback_fn=_symmetric_feedback,
            activity_fn=_spiking_activity_update,
            weight_fn=_contrastive_weight_update,
            **kwargs,
        )
        self.spike_threshold: float = float(
            kwargs.get("spike_threshold", config.clamp_max)
        )

    @classmethod
    def from_snn(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        beta: float = 0.1,
        **kwargs,
    ) -> TileSNN:
        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm="snn",
            mode="ep",
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            beta=beta,
            extra=kwargs,
        )
        return cls(config)

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: str = "cpu",
        task_type: str = "vision",
        **kwargs,
    ) -> TileSNN:
        return cls.from_snn(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            importance_lr=float(kwargs.get("importance_lr", 0.01)),
        ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# TileGNN — Graph Neural Network on tiles
# ──────────────────────────────────────────────────────────────────────────────


@register_model(
    "tile_gnn",
    family="tile",
    locality_level=LocalityLevel.LOCAL,
    tags=["graph", "gnn", "tile", status_tag("experimental")],
)
class TileGNN(TileAlgorithm):
    """Graph Neural Network on the tile substrate.

    Uses message-passing: each tile aggregates transformed activities from
    its backward neighbors before applying the activity update, implementing
    a learned graph convolution over the tile topology.
    """

    _algorithm = "gnn"

    def __init__(self, config: TileAlgorithmConfig, **kwargs) -> None:
        handler = TaskHandler(task_type="classification", output_dim=config.output_dim)
        super().__init__(
            config,
            task_handler=handler,
            feedback_fn=_symmetric_feedback,
            activity_fn=self._gnn_activity_update,
            weight_fn=_contrastive_weight_update,
            **kwargs,
        )
        self._msg_weights: nn.ParameterDict = nn.ParameterDict()
        self._build_message_weights()
        self._gnn_gate: nn.ModuleDict = nn.ModuleDict()
        self._build_gnn_gates()
        self._optim_io.add_param_group({
            "params": list(self._msg_weights.values())
            + list(self._gnn_gate.parameters())
        })

    def _build_message_weights(self) -> None:
        for src_id, dst_id in self.graph.edges:
            src = self.graph.tiles[src_id]
            dst = self.graph.tiles[dst_id]
            w = torch.randn(dst.neurons, src.neurons) * (1.0 / src.neurons**0.5)
            self._msg_weights[self._weight_key(src_id, dst_id)] = nn.Parameter(w)

    def _build_gnn_gates(self) -> None:
        """Per-tile gate over (activity, aggregated-message) -> activity delta.

        Gate input width is ``2 * tile.neurons`` (``[activity, agg]`` concat);
        output width is ``tile.neurons`` (gated delta added back to activity).
        """
        for tid, tile in self.graph.tiles.items():
            if tile.is_input:
                continue
            self._gnn_gate[str(tid)] = nn.Linear(tile.neurons * 2, tile.neurons)

    def _gnn_activity_update(  # ruff: ignore[too-many-arguments]
        self,
        tile: TileState,
        *,
        feedback: list[Tensor],
        importance: float,
        step_size: float,
        lambda_error: float,
        clamp_min: float,
        clamp_max: float,
        clamp: bool,
    ) -> Tensor:
        """Message-passing activity update: aggregate neighbor signals then relax."""
        if tile.activity is None or tile.error is None:
            raise ValueError("_gnn_activity_update requires settled activity and error")
        messages = [f for f in feedback if f is not None]
        agg = torch.zeros_like(tile.activity)
        for msg in messages:
            agg = agg + msg
        pred = self._predict_tile(tile.id)
        if pred is not None:
            agg = agg + pred
        merged = torch.cat([tile.activity, agg], dim=-1)
        gate = torch.sigmoid(self._gnn_gate[str(tile.id)](merged))
        gated = tile.activity + gate * agg
        return compute_activity_update(
            activity=gated,
            error=tile.error,
            fwd_feedback=[],
            importance=importance,
            step_size=step_size,
            lambda_error=lambda_error,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            clamp=clamp,
        )

    @classmethod
    def from_gnn(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        beta: float = 0.1,
        **kwargs,
    ) -> TileGNN:
        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm="gnn",
            mode="ep",
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            beta=beta,
            extra=kwargs,
        )
        return cls(config)

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: str = "cpu",
        task_type: str = "vision",
        **kwargs,
    ) -> TileGNN:
        return cls.from_gnn(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            importance_lr=float(kwargs.get("importance_lr", 0.01)),
        ).to(device)
