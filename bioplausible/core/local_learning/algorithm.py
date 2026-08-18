"""Generic tile algorithm builder (REFACTOR.md §8).

A single configurable substrate that subsumes tile-based local learning:

- TileEP  (Equilibrium Propagation / EquiTile): free/nudged contrastive
- TileFA  (Feedback Alignment): fixed random backward paths
- TileTP  (Target Propagation): target-driven feedback
- TilePC  (Predictive Coding): prediction-error activity dynamics
- TileHebbian: pure local Hebbian

The three algorithm-specific decisions are exposed as injection points:

- ``feedback``  — how downstream error reaches a tile (``FeedbackFn``)
- ``activity``  — how a tile settles its activity (``ActivityUpdateFn``)
- ``weight``    — how edge weights change from free/nudged statistics (``WeightUpdateFn``)

Each resolves from ``config.algorithm`` by default and can be overridden at
construction or by subclass. ``local_update()`` runs the canonical bio-plausible
loop (free phase -> nudged phase -> contrastive update) without autograd;
``train_step()`` provides the autograd baseline over the same parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bioplausible.core.local_learning.mixins import (
    LocalLearningConfigProtocol,
    MultiOptimizerMixin,
)
from bioplausible.core.local_learning.settling import (
    SettleConfig,
    SettleProtocol,
    SettleTelemetry,
    settle_universal,
)
from bioplausible.core.local_learning.task import TaskHandler
from bioplausible.core.losses import compute_accuracy
from bioplausible.core.tile import TileGraph, TileState
from bioplausible.core.tile.kernels import (
    compute_activity_update,
    compute_contrastive_hebbian_update,
    compute_hebbian_update,
)
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer

__all__ = [
    "ActivityUpdateFn",
    "FeedbackFn",
    "TileAlgorithm",
    "TileAlgorithmConfig",
    "WeightUpdateFn",
]

# ──────────────────────────────────────────────
# Dynamics protocols (the extensibility surface)
# ──────────────────────────────────────────────

type WeightLookup = Callable[[int, int], Tensor]


class FeedbackFn(Protocol):
    """Downstream error projected back into a tile's state space."""

    def __call__(
        self, tile: TileState, graph: TileGraph, lookup: WeightLookup
    ) -> list[Tensor]: ...


class ActivityUpdateFn(Protocol):
    """Settle a tile: current state + prediction error + feedback -> new activity."""

    def __call__(
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
    ) -> Tensor: ...


class WeightUpdateFn(Protocol):
    """Per-edge weight/bias deltas from free and nudged activity statistics."""

    def __call__(
        self,
        *,
        src_neurons: int,
        dst_neurons: int,
        src_free: Tensor | None,
        dst_free: Tensor | None,
        src_nudged: Tensor | None,
        dst_nudged: Tensor | None,
        learning_rate: float,
        beta: float,
        batch_size: int,
        importance: float,
    ) -> tuple[Tensor, Tensor]: ...


# ──────────────────────────────────────────────
# Built-in dynamics implementations
# ──────────────────────────────────────────────


def _symmetric_feedback(
    tile: TileState, graph: TileGraph, lookup: WeightLookup
) -> list[Tensor]:
    """Symmetric (transpose) backward projection; EP/PC/TB/Hebbian default."""
    feedback: list[Tensor] = []
    for dst_id in tile.fwd_neighbors:
        dst = graph.tiles[dst_id]
        if dst.error is None:
            continue
        w = lookup(tile.id, dst_id)
        feedback.append(dst.error @ w)
    return feedback


def _no_feedback(
    tile: TileState, graph: TileGraph, lookup: WeightLookup
) -> list[Tensor]:
    """No downstream coupling (pure feedforward Hebbian / single-pass settling)."""
    return []


def _spiking_activity_update(  # ruff: ignore[too-many-arguments]  # dynamics contract
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
    """Spiking activity: integrate input, fire above threshold, reset.

    Neuron model: ``activity = ReLU(activity - threshold) * (1 - fired)``
    then add the EP-style relaxation on the sub-threshold component.
    """
    if tile.activity is None or tile.error is None:
        raise ValueError("_spiking_activity_update requires settled activity and error")
    relaxed = compute_activity_update(
        activity=tile.activity,
        error=tile.error,
        fwd_feedback=feedback,
        importance=importance,
        step_size=step_size,
        lambda_error=lambda_error,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        clamp=clamp,
    )
    spike_thresh = clamp_max if clamp else 1.0
    fired = relaxed > spike_thresh
    return torch.where(fired, torch.zeros_like(relaxed), relaxed)


def _ep_activity_update(  # ruff: ignore[too-many-arguments]  # dynamics contract
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
    """Equilibrium activity settling: activity -= step * (error + lambda*act + feedback)."""
    if tile.activity is None or tile.error is None:
        raise ValueError("_ep_activity_update requires settled activity and error")
    return compute_activity_update(
        activity=tile.activity,
        error=tile.error,
        fwd_feedback=feedback,
        importance=importance,
        step_size=step_size,
        lambda_error=lambda_error,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        clamp=clamp,
    )


def _hebbian_activity_update(  # ruff: ignore[too-many-arguments]  # dynamics contract signature
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
    """Single-pass activity: settle straight to the prediction (no relaxation)."""
    if tile.prediction is None:
        raise ValueError("_hebbian_activity_update requires a precomputed prediction")
    return tile.prediction


def _contrastive_weight_update(  # ruff: ignore[too-many-arguments]  # dynamics contract signature
    *,
    src_neurons: int,
    dst_neurons: int,
    src_free: Tensor | None,
    dst_free: Tensor | None,
    src_nudged: Tensor | None,
    dst_nudged: Tensor | None,
    learning_rate: float,
    beta: float,
    batch_size: int,
    importance: float,
) -> tuple[Tensor, Tensor]:
    """Contrastive Hebbian: (free stats - nudged stats) / beta (Equilibrium Prop)."""
    if src_free is None or dst_free is None or src_nudged is None or dst_nudged is None:
        return torch.zeros(dst_neurons, src_neurons), torch.zeros(dst_neurons)
    w_up, b_up = compute_contrastive_hebbian_update(
        src_free=src_free,
        dst_free=dst_free,
        src_nudged=src_nudged,
        dst_nudged=dst_nudged,
        learning_rate=learning_rate,
        beta=beta,
        batch_size=batch_size,
    )
    # Kernel returns (src_neurons, dst_neurons); weights are (dst_neurons, src_neurons)
    return importance * w_up.T, importance * b_up


def _hebbian_weight_update(  # ruff: ignore[too-many-arguments]  # dynamics contract signature
    *,
    src_neurons: int,
    dst_neurons: int,
    src_free: Tensor | None,
    dst_free: Tensor | None,
    src_nudged: Tensor | None,
    dst_nudged: Tensor | None,
    learning_rate: float,
    beta: float,
    batch_size: int,
    importance: float,
) -> tuple[Tensor, Tensor]:
    """Pure local Hebbian: importance * avg(src_freex dst_free)."""
    if src_free is None or dst_free is None:
        return torch.zeros(dst_neurons, src_neurons), torch.zeros(dst_neurons)
    w_up, b_up = compute_hebbian_update(
        src_act=src_free, dst_err=dst_free, importance=importance, batch_size=batch_size
    )
    return w_up.T, b_up


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TileAlgorithmConfig(LocalLearningConfigProtocol):
    """Configuration for :class:`TileAlgorithm`.

    Satisfies :class:`~bioplausible.core.local_learning.mixins.LocalLearningConfigProtocol`
    so the model integrates with :class:`MultiOptimizerMixin` scheduler helpers.
    """

    input_dim: int
    output_dim: int

    # Topology
    neurons_per_tile: int = 48
    tiles_per_layer: int = 4
    num_hidden_layers: int = 3
    use_skip_connections: bool = False

    # Dynamics selection: "ep" | "fa" | "tp" | "pc" | "hebbian"
    algorithm: str = "ep"

    # Bio-plausible loop knobs
    beta: float = 0.1
    step_size: float = 0.1
    lambda_error: float = 1.0
    clamp_min: float = -10.0
    clamp_max: float = 10.0
    clamp: bool = True
    free_steps: int = 10
    nudged_steps: int = 10

    # MultiOptimizerMixin required fields
    learning_rate: float = 0.001
    importance_lr: float = 0.01
    mode: str = "ep"  # "ep" | "backprop" | "fa"

    # Algorithm-specific extras (forwarded to dynamics)
    extra: dict[str, object] = field(default_factory=dict)


# ──────────────────────────────────────────────
# TileAlgorithm
# ──────────────────────────────────────────────


class TileAlgorithm(nn.Module, MultiOptimizerMixin, SettleProtocol):
    """Generic tile-based local learning model.

    Builds a layered :class:`~bioplausible.core.tile.TileGraph` with per-edge
    weights and biases, and drives them through either the canonical
    bio-plausible loop (``local_update``: free phase -> nudged phase ->
    contrastive update, no autograd) or autograd BPTT (``train_step``).

    Dynamics resolve from ``config.algorithm`` and are injectable at construction:

    .. code-block:: python

        TileAlgorithm(config, feedback_fn=_symmetric_feedback,
                      activity_fn=_ep_activity_update,
                      weight_fn=_contrastive_weight_update)
    """

    W_in: nn.Linear
    W_out: nn.Linear
    tile_importance: nn.Parameter
    edge_importance: nn.Parameter
    equitile_config: LocalLearningConfigProtocol

    def __init__(
        self,
        config: TileAlgorithmConfig,
        *,
        feedback_fn: FeedbackFn | None = None,
        activity_fn: ActivityUpdateFn | None = None,
        weight_fn: WeightUpdateFn | None = None,
        task_handler: TaskHandler | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.equitile_config = config
        self._step_count = 0
        self._task_handler = task_handler

        self.graph = TileGraph()
        self.graph.build_layered(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            neurons_per_tile=config.neurons_per_tile,
            num_hidden_layers=config.num_hidden_layers,
            tiles_per_layer=config.tiles_per_layer,
            use_skip_connections=config.use_skip_connections,
        )

        self._build_io_projections()
        self._build_tile_weights()
        self._build_importance_params()

        # Resolve dynamics
        self._feedback = feedback_fn or self._resolve_feedback()
        self._activity = activity_fn or self._resolve_activity()
        self._weight = weight_fn or self._resolve_weight()

        self._setup_optimizers()

        # SettleProtocol attributes
        self.convergence_threshold: float = getattr(
            config, "convergence_threshold", 1e-3
        )
        self.convergence_start: int = getattr(config, "convergence_start", 5)
        self.max_steps: int = config.free_steps

        # Transient state for settle_universal
        self._settle_beta: float = 0.0
        self._settle_target: Tensor | None = None
        self._output_clamped: bool = False
        self._last_activations: list[Tensor] | None = None
        self._last_settle_converged: bool = False
        self._last_settle_steps: int = 0
        self._last_settle_final_delta: float = 0.0
        self._last_settle_telemetry: SettleTelemetry | None = None

    def _task_handler_ref(self) -> TaskHandler:
        """Lazily construct a classification task handler when none was injected.

        Exposed publicly (``task_handler``) so the substrate can act as a
        drop-in ``head`` for deployment model classes that delegate loss and
        metric computation to the handler.
        """
        if self._task_handler is None:
            self._task_handler = TaskHandler(
                task_type="classification", output_dim=self.config.output_dim
            )
        return self._task_handler

    @property
    def task_handler(self) -> TaskHandler:
        """Public task handler (loss/metric computations for head consumers)."""
        return self._task_handler_ref()

    def compute_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """Task-aware loss for a logits/target pair (head-facing API)."""
        return self.task_handler.compute_loss(logits, y)

    def compute_metrics(self, logits: Tensor, y: Tensor) -> float:
        """Task-aware metric for a logits/target pair (head-facing API)."""
        return self.task_handler.compute_metrics(logits, y)

    def get_config(self) -> TileAlgorithmConfig:
        """Return the model configuration (head-consumer contract)."""
        return self.config

    # ── Topology construction ────────────────────────────────────────────────

    def _build_io_projections(self) -> None:
        """Input/output projections between raw IO and tile-state space."""
        input_neurons = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.input_tile_ids
        )
        output_neurons = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids
        )
        self.W_in = nn.Linear(self.config.input_dim, input_neurons, bias=True)
        self.W_out = nn.Linear(output_neurons, self.config.output_dim, bias=True)

    def _build_importance_params(self) -> None:
        """Per-tile and per-edge importance (sigmoid-gated plasticity)."""
        self.tile_importance = nn.Parameter(torch.zeros(len(self.graph.tiles)))
        self.edge_importance = nn.Parameter(torch.zeros(len(self.graph.edges)))
        self._tile_idx = {tid: i for i, tid in enumerate(sorted(self.graph.tiles))}

    def _build_tile_weights(self) -> None:
        """Per-edge incoming weights and per-tile biases.

        ``_tile_weights[key(src, dst)]`` shapes ``(dst.neurons, src.neurons)``.
        """
        self._tile_weights = nn.ParameterDict()
        self._tile_biases = nn.ParameterDict()
        for tid, tile in self.graph.tiles.items():
            if tile.is_input:
                continue
            self._tile_biases[str(tid)] = nn.Parameter(torch.zeros(tile.neurons))
            for src_id in tile.bwd_neighbors:
                src = self.graph.tiles[src_id]
                bound = 1.0 / math.sqrt(src.neurons) if src.neurons > 0 else 0.0
                w = torch.empty(tile.neurons, src.neurons).uniform_(-bound, bound)
                self._tile_weights[self._weight_key(src_id, tid)] = nn.Parameter(w)

    @staticmethod
    def _weight_key(src_id: int, dst_id: int) -> str:
        return f"{src_id}_{dst_id}"

    # ── Optimizer split (MultiOptimizerMixin) ────────────────────────────────

    def _setup_optimizers(self) -> None:
        """Two parameter groups: IO+tile weights (``_optim_io``) and
        importance parameters (``_optim_importance``), per the mixin contract."""
        weight_params = (
            list(self.W_in.parameters())
            + list(self.W_out.parameters())
            + list(self._tile_weights.values())
            + list(self._tile_biases.values())
        )
        self._optim_io = create_optimizer(
            weight_params,
            OptimizerConfig(name="adam", lr=self.equitile_config.learning_rate),
        )
        self._optim_importance = create_optimizer(
            self.importance_params(),
            OptimizerConfig(name="adam", lr=self.equitile_config.importance_lr),
        )
        self._optim_full: torch.optim.Optimizer | None = None
        self._lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._lr_scheduler_type: str | None = None
        self._warmup_steps: int = 0
        self._warmup_start_lr: float = self.equitile_config.learning_rate * 0.1
        self._total_steps: int = 0

    # ── Dynamics dispatch ────────────────────────────────────────────────────

    def _resolve_feedback(self) -> FeedbackFn:
        match self.config.algorithm:
            case "hebbian" | "fa":
                return _no_feedback
            case _:
                return _symmetric_feedback

    def _resolve_activity(self) -> ActivityUpdateFn:
        match self.config.algorithm:
            case "hebbian":
                return _hebbian_activity_update
            case "snn":
                return _spiking_activity_update
            case _:
                return _ep_activity_update

    def _resolve_weight(self) -> WeightUpdateFn:
        match self.config.algorithm:
            case "hebbian":
                return _hebbian_weight_update
            case _:
                return _contrastive_weight_update

    def _weight_lookup(self, src_id: int, dst_id: int) -> Tensor:
        return self._tile_weights[self._weight_key(src_id, dst_id)]

    # ── Graph utilities ──────────────────────────────────────────────────────

    def _tile_activities_to_tensor(self, tile_ids: list[int]) -> Tensor:
        acts: list[Tensor] = [
            act
            for tid in tile_ids
            if (act := self.graph.tiles[tid].activity) is not None
        ]
        return torch.cat(acts, dim=1)

    def _set_tile_activities(self, tile_ids: list[int], x: Tensor) -> None:
        offset = 0
        for tid in tile_ids:
            n = self.graph.tiles[tid].neurons
            self.graph.tiles[tid].activity = x[:, offset : offset + n]
            offset += n

    def _predict_tile(self, tid: int) -> Tensor | None:
        """Weighted sum of incoming activities + bias."""
        tile = self.graph.tiles[tid]
        acc: Tensor | None = None
        for src_id in tile.bwd_neighbors:
            src_act = self.graph.tiles[src_id].activity
            if src_act is None:
                continue
            contrib = src_act @ self._weight_lookup(src_id, tid).T
            acc = contrib if acc is None else acc + contrib
        if acc is None:
            return None
        return acc + self._tile_biases[str(tid)].unsqueeze(0).expand(acc.shape[0], -1)

    def _clamp_input(self, x: Tensor, *, detach_input: bool = True) -> None:
        """Set input-tile activities to the projected input.

        ``detach_input=True`` cuts the autograd graph at the model boundary
        (the bio-plausible loop, whose weight updates are manual). Backprop
        consumers (deployment model classes) pass ``False`` so gradients flow
        into a preceding feature extractor.
        """
        self._set_tile_activities(
            self.graph.input_tile_ids, self.W_in(x.detach() if detach_input else x)
        )

    def _clamp_output_to_target(self, y: Tensor) -> Tensor:
        """Set output-tile activities to the (projected) one-hot target."""
        onehot = F.one_hot(y, self.config.output_dim).float()
        out_tile_neurons = sum(
            self.graph.tiles[tid].neurons for tid in self.graph.output_tile_ids
        )
        if out_tile_neurons == self.config.output_dim:
            target = onehot
        else:
            target = onehot @ self.W_out.weight.detach().T  # (batch, out_tiles)
        self._set_tile_activities(self.graph.output_tile_ids, target)
        return target

    # ── Bio-plausible loop ───────────────────────────────────────────────────

    def _topic_importance(self, tid: int) -> float:
        return torch.sigmoid(self.tile_importance[self._tile_idx[tid]]).item()

    def _edge_importance(self, edge_index: int) -> float:
        return torch.sigmoid(self.edge_importance[edge_index]).item()

    def _forward_propagation(self) -> None:
        """Initialize every non-input tile's activity to its feedforward prediction."""
        for layer_tiles in self.graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self.graph.tiles[tid]
                pred = self._predict_tile(tid)
                tile.prediction = pred
                tile.activity = pred
                tile.error = None

    def _settle(self, steps: int, nudged: bool) -> None:
        """Relax all non-input tiles for ``steps`` iterations."""
        self._forward_propagation()  # guarantees tile.activity/prediction are set
        for _ in range(steps):
            for layer_tiles in self.graph.layer_ids[1:]:
                for tid in layer_tiles:
                    tile = self.graph.tiles[tid]
                    if nudged and tid in self.graph.output_tile_ids:
                        continue  # output stays clamped
                    pred = self._predict_tile(tid)
                    assert tile.activity is not None and pred is not None
                    tile.prediction = pred
                    tile.error = tile.activity - pred
                    feedback = self._feedback(tile, self.graph, self._weight_lookup)
                    tile.activity = self._activity(
                        tile,
                        feedback=feedback,
                        importance=self._topic_importance(tid),
                        step_size=self.config.step_size,
                        lambda_error=self.config.lambda_error,
                        clamp_min=self.config.clamp_min,
                        clamp_max=self.config.clamp_max,
                        clamp=self.config.clamp,
                    )

    def free_phase(self, x: Tensor, steps: int | None = None) -> dict[int, Tensor]:
        """Unclamped settling; returns per-tile free activities."""
        self._clamp_input(x)
        self._settle(steps or self.config.free_steps, nudged=False)
        return {tid: self._clone_activity(tid) for tid in self.graph.tiles}

    def nudged_phase(
        self, x: Tensor, y: Tensor, steps: int | None = None
    ) -> dict[int, Tensor]:
        """Target-clamped settling; returns per-tile nudged activities."""
        self._clamp_input(x)
        self._settle(1, nudged=False)  # bootstrap via forward pass
        self._clamp_output_to_target(y)
        self._settle(steps or self.config.nudged_steps, nudged=True)
        return {tid: self._clone_activity(tid) for tid in self.graph.tiles}

    def _clone_activity(self, tid: int) -> Tensor:
        act = self.graph.tiles[tid].activity
        assert act is not None  # _settle guarantees activities are set
        return act.clone()

    def contrastive_update(
        self,
        free: dict[int, Tensor],
        nudged: dict[int, Tensor],
    ) -> None:
        """Apply per-edge contrastive (or Hebbian) weight deltas."""

        def _f(k: int) -> Tensor | None:
            return free.get(k)

        def _n(k: int) -> Tensor | None:
            return nudged.get(k)

        batch_size = next(iter(free.values())).shape[0]
        out_tiles = self.graph.output_tile_ids

        with torch.no_grad():
            for ei, (src_id, dst_id) in enumerate(self.graph.edges):
                src = self.graph.tiles[src_id]
                dst = self.graph.tiles[dst_id]
                w_up, b_up = self._weight(
                    src_neurons=src.neurons,
                    dst_neurons=dst.neurons,
                    src_free=_f(src_id),
                    dst_free=_f(dst_id),
                    src_nudged=_n(src_id),
                    dst_nudged=_n(dst_id),
                    learning_rate=self.config.learning_rate,
                    beta=self.config.beta,
                    batch_size=batch_size,
                    importance=self._edge_importance(ei),
                )
                self._tile_weights[self._weight_key(src_id, dst_id)].sub_(w_up)
                self._tile_biases[str(dst_id)].sub_(b_up)

            # W_out: free-phase gradient from (free - nudged) output statistics.
            # The nudged output is clamped to the target, so free - nudged
            # encodes the output error without needing ``y`` here.
            out_free_acts = free[out_tiles[0]]
            out_nudged_acts = nudged[out_tiles[0]]
            scale = self.config.learning_rate / self.config.beta
            w_out_up = (
                scale * (out_free_acts - out_nudged_acts).T @ out_free_acts / batch_size
            )
            b_out_up = (
                scale * (out_free_acts - out_nudged_acts).mean(dim=0) / batch_size
            )
            self.W_out.weight.sub_(w_out_up)
            self.W_out.bias.sub_(b_out_up)

    # ── SettleProtocol Implementation (Family B: activations list) ─────────────

    def _get_settle_state(self) -> list[Tensor]:
        """Get current tile activities as a flat list for SettleProtocol.

        Returns activities in layer order (input, hidden..., output).
        """
        state: list[Tensor] = []
        for layer_tiles in self.graph.layer_ids:
            for tid in layer_tiles:
                act = self.graph.tiles[tid].activity
                if act is not None:
                    state.append(act.clone())
        return state

    def _set_settle_state(self, state: list[Tensor]) -> None:
        """Set tile activities from a flat list."""
        idx = 0
        for layer_tiles in self.graph.layer_ids:
            for tid in layer_tiles:
                if idx < len(state):
                    self.graph.tiles[tid].activity = state[idx]
                    idx += 1

    def _initialize_state(self, x: Tensor) -> list[Tensor]:
        """Return initial tile activities for settle_universal."""
        self._clamp_input(x)
        self._forward_propagation()
        return self._get_settle_state()

    def _transform_input(self, x: Tensor) -> Tensor:
        """Transform input (stored for _step)."""
        return x

    def _step(
        self,
        state: list[Tensor],
        x_transformed: Tensor,
    ) -> list[Tensor]:
        """Single settle step for settle_universal.

        Runs one iteration over all non-input tiles.
        """
        self._set_settle_state(state)
        nudged = self._settle_beta > 0 and self._settle_target is not None

        if nudged and not hasattr(self, "_output_clamped"):
            self._clamp_output_to_target(self._settle_target)
            self._output_clamped = True

        # One settle iteration
        for layer_tiles in self.graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self.graph.tiles[tid]
                if nudged and tid in self.graph.output_tile_ids:
                    continue
                pred = self._predict_tile(tid)
                if tile.activity is None or pred is None:
                    continue
                tile.prediction = pred
                tile.error = tile.activity - pred
                feedback = self._feedback(tile, self.graph, self._weight_lookup)
                tile.activity = self._activity(
                    tile,
                    feedback=feedback,
                    importance=self._topic_importance(tid),
                    step_size=self.config.step_size,
                    lambda_error=self.config.lambda_error,
                    clamp_min=self.config.clamp_min,
                    clamp_max=self.config.clamp_max,
                    clamp=self.config.clamp,
                )

        return self._get_settle_state()

    def _check_converged(
        self,
        state_new: list[Tensor],
        state_old: list[Tensor],
        step: int,
    ) -> bool:
        """Custom convergence check for tile activities."""
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
        state: list[Tensor],
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
        x: Tensor,
        *,
        beta: float = 0.0,
        target: Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> tuple[Tensor, int, bool, SettleTelemetry | None]:
        """Run settle using the universal primitive with full telemetry."""
        self._settle_beta = beta
        self._settle_target = target
        self._output_clamped = False

        config = SettleConfig(
            max_steps=steps if steps is not None else self.max_steps,
            convergence_threshold=self.convergence_threshold,
            convergence_start=self.convergence_start,
        )

        state, steps_taken, converged, telemetry = settle_universal(
            self,
            x,
            config=config,
            algorithm="tile",
            family="B",
            hardware=self.config.device if hasattr(self.config, "device") else "cpu",
            backend="pytorch",
            return_trajectory=return_trajectory,
        )

        self._last_activations = state
        self._last_settle_telemetry = telemetry

        # Project output tile activities to logits
        out_acts = self._tile_activities_to_tensor(self.graph.output_tile_ids)
        out = self.W_out(out_acts)

        return out, steps_taken, converged, telemetry

    def get_settle_telemetry(self) -> SettleTelemetry | None:
        """Return the last settle telemetry for external consumers."""
        return self._last_settle_telemetry

    # ── Public training entry points ─────────────────────────────────────────

    def local_update(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """Bio-plausible training step: free -> nudged -> contrastive update."""
        self.train()
        free = self.free_phase(x)
        nudged = self.nudged_phase(x, y)
        self.contrastive_update(free, nudged)
        self._step_count += 1
        return self._eval_metrics(x, y)

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """Autograd BPTT baseline over the same parameters."""
        self.train()
        out = self.forward_logits(x)
        loss = F.cross_entropy(out, y)
        self._optim_io.zero_grad()
        loss.backward()
        self._optim_io.step()
        self._optim_importance.step()
        self._step_count += 1
        return {"loss": loss.item(), "accuracy": self._accuracy(out, y)}

    def forward_logits(
        self,
        x: Tensor,
        *,
        detach_input: bool = True,
        beta: float = 0.0,
        target: Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Tensor | tuple[Tensor, list[list[Tensor]]] | tuple[Tensor, dict[str, object]]:
        """Feedforward through the tile graph to logits.

        ``detach_input=False`` keeps the graph open to the input (backprop
        consumers); the bio-plausible loop always uses the detached variant.
        """
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

        self._clamp_input(x, detach_input=detach_input)
        self._settle(1, nudged=False)
        out = self._tile_activities_to_tensor(self.graph.output_tile_ids)
        return self.W_out(out)

    def _eval_metrics(self, x: Tensor, y: Tensor) -> dict[str, float]:
        with torch.no_grad():
            logits = self.forward_logits(x)
            loss = F.cross_entropy(logits, y).item()
        return {"loss": loss, "accuracy": self._accuracy(logits, y)}

    @staticmethod
    def _accuracy(logits: Tensor, y: Tensor) -> float:
        return compute_accuracy(logits, y)

    def forward(
        self,
        x: Tensor,
        *,
        beta: float = 0.0,
        target: Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Tensor | tuple[Tensor, list[list[Tensor]]] | tuple[Tensor, dict[str, object]]:
        """Feedforward through the tile graph to logits.

        Supports SettleProtocol dynamics via ``return_dynamics``.
        """
        return self.forward_logits(
            x,
            detach_input=True,
            beta=beta,
            target=target,
            steps=steps,
            return_trajectory=return_trajectory,
            return_dynamics=return_dynamics,
        )

    # ──────────────────────────────────────────────
    # Tile-growth API (for dynamic topology)
    # ──────────────────────────────────────────────

    def add_tile(
        self,
        *,
        neurons: int,
        layer_id: int,
        pos_x: float = 0.0,
        pos_y: float = 0.0,
        is_input: bool = False,
        is_output: bool = False,
    ) -> int:
        """Add a new tile to the graph.

        Parameters
        ----------
        neurons : int
            Number of neurons in the tile
        layer_id : int
            Layer ID
        pos_x : float
            X position (for visualization/topology)
        pos_y : float
            Y position
        is_input : bool
            Is input tile
        is_output : bool
            Is output tile

        Returns
        -------
        int
            New tile ID
        """
        new_id = max(self.graph.tiles.keys()) + 1 if self.graph.tiles else 0
        tile = TileState(
            id=new_id,
            neurons=neurons,
            layer_id=layer_id,
            pos_x=pos_x,
            pos_y=pos_y,
            is_input=is_input,
            is_output=is_output,
        )
        self.graph.tiles[new_id] = tile

        if is_input:
            self.graph.input_tile_ids.append(new_id)
        if is_output:
            self.graph.output_tile_ids.append(new_id)

        # Ensure layer_ids has the layer
        while len(self.graph.layer_ids) <= layer_id:
            self.graph.layer_ids.append([])
        self.graph.layer_ids[layer_id].append(new_id)

        # Update tile importance
        with torch.no_grad():
            old_importance = self.tile_importance.data
            self.tile_importance = nn.Parameter(
                torch.cat([
                    old_importance,
                    torch.ones(1, device=old_importance.device),
                ])
            )
        self._tile_idx[new_id] = len(old_importance)

        # Rebuild weight structures for the new tile if not input
        if not is_input:
            self._tile_biases[str(new_id)] = nn.Parameter(torch.zeros(neurons))
            # Note: edges to/from this tile must be added via add_edge

        self.reset_optimizers()
        return new_id

    def remove_tile(self, tile_id: int) -> None:
        """Remove a tile from the graph.

        Parameters
        ----------
        tile_id : int
            Tile ID to remove
        """
        if tile_id not in self.graph.tiles:
            return

        # Remove edges connected to this tile
        edges_to_remove = [
            (src, dst) for src, dst in self.graph.edges if tile_id in {src, dst}
        ]
        for src, dst in edges_to_remove:
            self.remove_edge(src, dst)

        # Update tile importance
        # graph.all_tiles is sorted by ID
        sorted_ids = sorted(self.graph.tiles.keys())
        try:
            idx = sorted_ids.index(tile_id)
            with torch.no_grad():
                mask = torch.ones(
                    len(self.tile_importance),
                    dtype=torch.bool,
                    device=self.tile_importance.device,
                )
                mask[idx] = False
                self.tile_importance = nn.Parameter(self.tile_importance.data[mask])
        except ValueError:
            pass  # Should not happen if tile_id is in graph.tiles

        # Remove from graph
        del self.graph.tiles[tile_id]
        if tile_id in self.graph.input_tile_ids:
            self.graph.input_tile_ids.remove(tile_id)
        if tile_id in self.graph.output_tile_ids:
            self.graph.output_tile_ids.remove(tile_id)

        # Remove from layer_ids
        for layer in self.graph.layer_ids:
            if tile_id in layer:
                layer.remove(tile_id)

        # Rebuild the id->index mapping (indices shifted by the removal)
        self._tile_idx = {tid: i for i, tid in enumerate(sorted(self.graph.tiles))}

        self.reset_optimizers()

    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> None:
        """Add an edge between two tiles.

        Parameters
        ----------
        src_id : int
            Source tile ID
        dst_id : int
            Destination tile ID
        weight : torch.Tensor, optional
            Initial weight. If None, initialized randomly.
        bias : torch.Tensor, optional
            Initial bias. If None, initialized to zeros.
        """
        if src_id not in self.graph.tiles or dst_id not in self.graph.tiles:
            return

        # Add to graph
        self.graph._add_edge(src_id, dst_id)

        # Initialize parameters
        src = self.graph.tiles[src_id]
        dst = self.graph.tiles[dst_id]
        key = self._weight_key(src_id, dst_id)

        if weight is None:
            bound = 1.0 / math.sqrt(src.neurons) if src.neurons > 0 else 0.0
            weight = torch.empty(
                dst.neurons, src.neurons, device=self.tile_importance.device
            ).uniform_(-bound, bound)

        if not isinstance(weight, nn.Parameter):
            weight = nn.Parameter(weight)

        if bias is None:
            bias = torch.zeros(dst.neurons, device=self.tile_importance.device)

        if not isinstance(bias, nn.Parameter):
            bias = nn.Parameter(bias)

        self._tile_weights[key] = weight
        self._tile_biases[str(dst_id)] = bias  # Note: biases are per-destination tile

        # Update edge importance
        with torch.no_grad():
            old_importance = self.edge_importance.data
            self.edge_importance = nn.Parameter(
                torch.cat([
                    old_importance,
                    torch.ones(1, device=old_importance.device),
                ])
            )

        self.reset_optimizers()

    def remove_edge(self, src_id: int, dst_id: int) -> None:
        """Remove an edge between two tiles.

        Parameters
        ----------
        src_id : int
            Source tile ID
        dst_id : int
            Destination tile ID
        """
        if (src_id, dst_id) not in self.graph._edge_set:
            return

        # Find index in graph.edges for importance removal
        try:
            idx = self.graph.edges.index((src_id, dst_id))
            with torch.no_grad():
                mask = torch.ones(
                    len(self.edge_importance),
                    dtype=torch.bool,
                    device=self.edge_importance.device,
                )
                mask[idx] = False
                self.edge_importance = nn.Parameter(self.edge_importance.data[mask])
        except ValueError:
            pass

        # Remove from graph
        self.graph._edge_set.remove((src_id, dst_id))
        self.graph.edges.remove((src_id, dst_id))

        # Update neighbors
        if dst_id in self.graph.tiles[src_id].fwd_neighbors:
            self.graph.tiles[src_id].fwd_neighbors.remove(dst_id)
        if src_id in self.graph.tiles[dst_id].bwd_neighbors:
            self.graph.tiles[dst_id].bwd_neighbors.remove(src_id)

        # Remove parameters
        key = self._weight_key(src_id, dst_id)
        if key in self._tile_weights:
            del self._tile_weights[key]
        # Note: biases are per-dst tile, so we don't remove them here
        # (they're shared across all incoming edges to the same dst)

        self.reset_optimizers()

    def _get_edge_params(
        self, src_id: int, dst_id: int
    ) -> tuple[Tensor | None, Tensor | None]:
        """Get weight and bias for an edge.

        Returns
        -------
        tuple
            (weight, bias) or (None, None) if edge doesn't exist
        """
        weight = self._tile_weights.get(self._weight_key(src_id, dst_id))
        bias = self._tile_biases.get(str(dst_id))
        return weight, bias

    # ──────────────────────────────────────────────
    # Static factories
    # ──────────────────────────────────────────────

    @classmethod
    def _build_config(  # ruff: ignore[too-many-arguments]  # factory param bundle
        cls,
        *,
        algorithm: str,
        mode: str,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        neurons_per_tile: int,
        tiles_per_layer: int,
        learning_rate: float,
        importance_lr: float,
        beta: float | None,
        **kwargs,
    ) -> TileAlgorithmConfig:
        extra = dict(kwargs)
        if beta is not None:
            extra["beta"] = beta
        return TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm=algorithm,
            mode=mode,
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            beta=beta or 0.1,
            extra=extra,
        )

    @classmethod
    def from_ep(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
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
    ) -> TileAlgorithm:
        """Equilibrium Propagation (EquiTile)."""
        return cls(
            cls._build_config(
                algorithm="ep",
                mode="ep",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=beta,
                **kwargs,
            )
        )

    @classmethod
    def from_fa(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        **kwargs,
    ) -> TileAlgorithm:
        """Feedback Alignment (fixed random backward feedback via subclass)."""
        return cls(
            cls._build_config(
                algorithm="fa",
                mode="fa",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=None,
                **kwargs,
            )
        )

    @classmethod
    def from_hebbian(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        **kwargs,
    ) -> TileAlgorithm:
        """Pure local Hebbian (single-pass settling, no feedback)."""
        return cls(
            cls._build_config(
                algorithm="hebbian",
                mode="ep",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=None,
                **kwargs,
            )
        )

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
    ) -> TileAlgorithm:
        """Predictive Coding."""
        return cls(
            cls._build_config(
                algorithm="pc",
                mode="ep",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=beta,
                **kwargs,
            )
        )

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
    ) -> TileAlgorithm:
        """Target Propagation."""
        return cls(
            cls._build_config(
                algorithm="tp",
                mode="ep",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=beta,
                **kwargs,
            )
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
    ) -> TileAlgorithm:
        """Spiking Neural Network (threshold-and-reset activity dynamics)."""
        return cls(
            cls._build_config(
                algorithm="snn",
                mode="ep",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=beta,
                **kwargs,
            )
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
    ) -> TileAlgorithm:
        """Graph Neural Network (message-passing across tile edges)."""
        return cls(
            cls._build_config(
                algorithm="gnn",
                mode="ep",
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                neurons_per_tile=neurons_per_tile,
                tiles_per_layer=tiles_per_layer,
                learning_rate=learning_rate,
                importance_lr=importance_lr,
                beta=beta,
                **kwargs,
            )
        )
