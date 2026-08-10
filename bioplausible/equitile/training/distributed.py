"""
EquiTile Distributed: Multi-GPU Tile Distribution
==================================================

Unified multi-GPU training for EquiTile, merging features previously
split across ``distributed.py`` and ``multigpu.py``:

- Tile distribution across devices
- Inter-GPU communication for tile boundaries (``TileCommunicator``)
- NCCL-based gradient synchronization (``NCCLCommunicator`` from ``._nccl``)
- Mixed precision support (FP16/BF16 with loss scaling)
- Asynchronous tile execution with CUDA stream overlap
- Dynamic tile growth/pruning
- Timing instrumentation

Key Components
--------------
- ``DistributedConfig``: Training configuration
- ``TileCommunicator``: Inter-GPU tile-boundary communication
- ``MixedPrecisionTrainer``: FP16/BF16 support with loss scaling
- ``AsyncTileExecutor``: CUDA stream overlap for compute/communication
- ``DistributedEquiTile``: Single multi-GPU wrapper class

Examples
--------
>>> from bioplausible.equitile import EquiTile, DistributedEquiTile
>>> model = EquiTile(
...     neurons_per_tile=64, num_layers=4, tiles_per_layer=4,
...     input_dim=784, output_dim=10,
... )
>>> dist_model = DistributedEquiTile(
...     model,
...     device_ids=[0, 1, 2, 3],
...     mixed_precision=True,
... )
>>> stats = dist_model.train_step(X, y)
"""

import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp

from bioplausible.core.logging import get_logger
from bioplausible.core.tile.kernels import (
    compute_activity_update,
    compute_hebbian_update,
    compute_tile_prediction,
)
from bioplausible.equitile.core.config import DistributedConfig, TileGrowthConfig
from bioplausible.equitile.training._nccl import NCCLCommunicator

__all__ = [
    "AsyncTileExecutor",
    "DeviceAssignment",
    "DistributedEquiTile",
    "MixedPrecisionTrainer",
    "TileCommunicator",
    "create_distributed_model",
    "logger",
    "spawn_distributed_worker",
]
if TYPE_CHECKING:
    from bioplausible.equitile.core import EquiTile

logger = get_logger()

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DeviceAssignment:
    """Assignment of tiles to devices.

    Attributes
    ----------
    device_id : int
        Device index
    device : torch.device
        Device object
    tile_ids : list of int
        Tile IDs assigned to this device
    edge_ids : list of tuple
        Edge IDs assigned to this device
    """

    device_id: int
    device: torch.device
    tile_ids: list[int]
    edge_ids: list[tuple[int, int]]


# =============================================================================
# Tile Communicator
# =============================================================================


class TileCommunicator:
    """Handles inter-GPU communication for tile boundaries.

    When tiles are distributed across GPUs, boundary tiles need to
    exchange activity and error information.  This class identifies
    those boundaries and manages the exchange.

    Parameters
    ----------
    assignments : list of DeviceAssignment
        Device assignments
    graph : TileGraph
        The tile graph topology
    """

    def __init__(
        self,
        assignments: list[DeviceAssignment],
        graph: object,
    ) -> None:
        self.assignments = assignments
        self.graph = graph
        self.n_devices = len(assignments)

        # Build communication groups
        self._boundary_tiles = self._find_boundary_tiles()
        self._comm_buffers: dict[int, dict[str, torch.Tensor]] = {}

    def _find_boundary_tiles(self) -> dict[int, list[tuple[int, int]]]:
        """Find tiles that need cross-device communication.

        Returns
        -------
        dict
            Dict mapping device_id to list of (local_tile, remote_tile) pairs
        """
        tile_to_device: dict[int, int] = {}
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile_to_device[tile_id] = assignment.device_id

        boundary_map = self.graph.get_boundary_tiles(tile_to_device)

        boundary: dict[int, list[tuple[int, int]]] = {
            i: [] for i in range(self.n_devices)
        }

        for local_tile, remote_tiles in boundary_map.items():
            local_dev = tile_to_device.get(local_tile)
            if local_dev is not None:
                for remote_tile in remote_tiles:
                    boundary[local_dev].append((local_tile, remote_tile))

        return boundary

    def exchange_activities(
        self,
        activities: dict[int, torch.Tensor],
        device_id: int,
    ) -> dict[int, torch.Tensor]:
        """Exchange tile activities across device boundaries.

        Parameters
        ----------
        activities : dict
            Local tile activities
        device_id : int
            This device's ID

        Returns
        -------
        dict
            Activities from remote boundary tiles
        """
        if self.n_devices == 1:
            return {}

        received: dict[int, torch.Tensor] = {}

        for local_tile, remote_tile in self._boundary_tiles.get(device_id, []):
            if remote_tile in activities:
                received[local_tile] = activities[remote_tile].clone()

        return received

    def sync_gradients(
        self,
        gradients: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sync gradients across devices (all_reduce).

        Parameters
        ----------
        gradients : dict
            Gradient tensors by name

        Returns
        -------
        dict
            Synced gradients
        """
        if self.n_devices == 1:
            return gradients

        for name, grad in gradients.items():
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG)

        return gradients


# =============================================================================
# Mixed Precision Trainer
# =============================================================================


class MixedPrecisionTrainer:
    """Mixed precision training for EquiTile.

    Supports FP16 and BF16 with loss scaling.

    Parameters
    ----------
    model : EquiTile
        The model
    dtype : str
        Precision type: ``'float16'`` or ``'bfloat16'``
    initial_scale : float
        Initial loss scale
    scale_window : int
        Steps before increasing scale
    """

    def __init__(
        self,
        model: EquiTile,
        dtype: str = "float16",
        initial_scale: float = 65536.0,
        scale_window: int = 1000,
    ) -> None:
        self.model = model
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.enabled = self.dtype != torch.float32

        self.scale = initial_scale if self.enabled else 1.0
        self.scale_window = scale_window
        self.steps_without_overflow = 0

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def cast_model(self) -> None:
        """Cast model weights to mixed precision."""
        if not self.enabled:
            return

        for weight in self.model.edge_weights.values():
            weight.data = weight.data.to(self.dtype)
        for bias in self.model.edge_biases.values():
            bias.data = bias.data.to(self.dtype)

    def autocast(self):
        """Context manager for autocast.

        Returns
        -------
        torch.amp.autocast
            Autocast context manager
        """
        return torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient scaling.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor

        Returns
        -------
        torch.Tensor
            Scaled loss
        """
        if self.enabled:
            return loss * self.scale
        return loss

    def unscale_and_clip_grads(
        self,
        gradients: dict[str, torch.Tensor],
        max_norm: float = 1.0,
    ) -> float:
        """Unscale gradients and clip.

        Parameters
        ----------
        gradients : dict
            Gradient tensors
        max_norm : float
            Maximum gradient norm

        Returns
        -------
        float
            Total gradient norm
        """
        total_norm = 0.0

        for name, grad in gradients.items():
            if grad is None:
                continue

            grad.data.mul_(1.0 / self.scale)

            param_norm = grad.data.norm(2)
            total_norm = max(total_norm, param_norm.item())

            if total_norm > max_norm:
                clip_coef = max_norm / (total_norm + 1e-6)
                grad.data.mul_(clip_coef)

        return total_norm

    def update_scale(self, found_inf: bool) -> None:
        """Update loss scale based on overflow.

        Parameters
        ----------
        found_inf : bool
            Whether inf/nan was found
        """
        if found_inf:
            self.scale = max(self.scale / 2.0, 1.0)
            self.steps_without_overflow = 0
        else:
            self.steps_without_overflow += 1
            if self.steps_without_overflow >= self.scale_window:
                self.scale = min(self.scale * 2.0, 65536.0)
                self.steps_without_overflow = 0


# =============================================================================
# Async Tile Executor
# =============================================================================


class AsyncTileExecutor:
    """Executes tile operations asynchronously with NCCL.

    Overlaps communication and computation for maximum throughput.

    Parameters
    ----------
    communicator : NCCLCommunicator
        NCCL communicator
    """

    def __init__(self, communicator: NCCLCommunicator) -> None:
        self.communicator = communicator
        self._compute_stream: torch.cuda.Stream | None = None
        self._comm_stream: torch.cuda.Stream | None = None
        self._running = False
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start async executor."""
        if torch.cuda.is_available():
            self._compute_stream = torch.cuda.Stream()
            self._comm_stream = torch.cuda.Stream()
        self._running = True

    def stop(self) -> None:
        """Stop async executor."""
        self._running = False
        self.synchronize()

    def submit_compute(
        self,
        op: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Submit compute operation.

        Parameters
        ----------
        op : callable
            Operation to execute
        *args
            Positional arguments
        **kwargs
            Keyword arguments
        """
        if self._compute_stream is not None:
            with torch.cuda.stream(self._compute_stream):
                op(*args, **kwargs)
        else:
            op(*args, **kwargs)

    def submit_comm(
        self,
        op: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Submit communication operation.

        Parameters
        ----------
        op : callable
            Operation to execute
        *args
            Positional arguments
        **kwargs
            Keyword arguments
        """
        if self._comm_stream is not None:
            with torch.cuda.stream(self._comm_stream):
                op(*args, **kwargs)
        else:
            op(*args, **kwargs)

    def synchronize(self) -> None:
        """Synchronize compute and communication streams."""
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        if self._comm_stream is not None:
            self._comm_stream.synchronize()

    @contextmanager
    def compute_stream_context(self):
        """Context manager for compute stream."""
        if self._compute_stream is not None:
            with torch.cuda.stream(self._compute_stream):
                yield
        else:
            yield

    @contextmanager
    def comm_stream_context(self):
        """Context manager for communication stream."""
        if self._comm_stream is not None:
            with torch.cuda.stream(self._comm_stream):
                yield
        else:
            yield


# =============================================================================
# Distributed EquiTile (unified single class)
# =============================================================================


class DistributedEquiTile:
    """Multi-GPU distributed EquiTile.

    Distributes tiles across multiple GPUs for parallel training with
    optional mixed precision and async CUDA stream execution.

    Parameters
    ----------
    model : EquiTile
        Base EquiTile model
    config : DistributedConfig, optional
        Distributed training configuration.  If *None*, auto-detects GPUs.
    device_ids : list of int, optional
        Override device IDs (takes precedence over config).

    Examples
    --------
    >>> model = EquiTile(
    ...     neurons_per_tile=64, num_layers=4, tiles_per_layer=4,
    ...     input_dim=784, output_dim=10,
    ... )
    >>> dist_model = DistributedEquiTile(model, device_ids=[0, 1, 2, 3])
    >>> stats = dist_model.train_step(X, y)
    """

    def __init__(
        self,
        model: EquiTile,
        config: DistributedConfig | None = None,
        *,
        device_ids: list[int] | None = None,
        mixed_precision: bool | None = None,
        async_execution: bool | None = None,
        tile_balance: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or DistributedConfig()

        # Allow kwargs to override config fields (convenience)
        if device_ids is not None:
            self.config.device_ids = device_ids
        if mixed_precision is not None:
            self.config.mixed_precision = mixed_precision
        if async_execution is not None:
            self.config.overlap_communication = async_execution
        if tile_balance is not None:
            self.config.tile_balance = tile_balance  # type: ignore[assignment]

        # Set up devices
        if not self.config.device_ids:
            if torch.cuda.is_available():
                self.config.device_ids = list(range(torch.cuda.device_count()))
                self.devices = [
                    torch.device(f"cuda:{i}") for i in self.config.device_ids
                ]
            else:
                self.config.device_ids = [0]
                self.devices = [torch.device("cpu")]
        else:
            self.devices = [torch.device(f"cuda:{i}") for i in self.config.device_ids]
        self.n_devices = len(self.devices)

        # Assign tiles to devices
        self.assignments = self._assign_tiles()

        # Tile communicator (boundary management)
        self.communicator = TileCommunicator(
            self.assignments,
            self.model.graph,
        )

        # NCCL communicator (gradient sync across devices)
        self.nccl_communicator = NCCLCommunicator()

        # Mixed precision
        self.mp_trainer: MixedPrecisionTrainer | None = None
        if self.config.mixed_precision and self.devices[0].type == "cpu":
            self.config.mixed_precision = False

        if self.config.mixed_precision:
            self.mp_trainer = MixedPrecisionTrainer(
                model, dtype=self.config.mixed_precision_dtype
            )

        # Async executor
        self.executor: AsyncTileExecutor | None = None
        if self.config.overlap_communication and self.n_devices > 1:
            self.executor = AsyncTileExecutor(self.nccl_communicator)
            self.executor.start()

        # Tile growth/pruning
        self.growth_config = TileGrowthConfig()
        self._steps_since_modify = 0

        # Move tiles to assigned devices
        self._distribute_tiles()

        # Gradient accumulation
        self._accumulated_gradients: dict[str, torch.Tensor] = {}
        self._accumulation_step = 0

        # Timing
        self._comm_time = 0.0
        self._compute_time = 0.0

    # ------------------------------------------------------------------
    # Tile assignment & distribution
    # ------------------------------------------------------------------

    def _assign_tiles(self) -> list[DeviceAssignment]:
        """Assign tiles to devices.

        Returns
        -------
        list of DeviceAssignment
            Device assignments
        """
        n_tiles = len(self.model.graph.tiles)
        tile_ids = list(self.model.graph.tiles.keys())
        balance = self.config.tile_balance

        assignments: list[DeviceAssignment] = []

        for i, device_id in enumerate(self.config.device_ids):
            if balance == "round_robin":
                assigned = tile_ids[i :: self.n_devices]
            elif balance == "layered":
                layer_size = n_tiles // self.n_devices
                start = i * layer_size
                end = start + layer_size if i < self.n_devices - 1 else n_tiles
                assigned = tile_ids[start:end]
            elif balance == "balanced":
                # Balance by neuron count
                remaining = list(tile_ids)
                assigned = []
                for _ in range(self.n_devices):
                    device_tiles = []
                    device_neurons = 0
                    for tid in list(remaining):
                        tile = self.model.graph.tiles[tid]
                        if device_neurons + tile.neurons <= (
                            n_tiles // self.n_devices
                        ) * max(
                            (t.neurons for t in self.model.graph.tiles.values()),
                            default=1,
                        ):
                            device_tiles.append(tid)
                            device_neurons += tile.neurons
                            remaining.remove(tid)
                    if i == len(assignments):
                        assigned = device_tiles
                        break
                if not assigned:
                    assigned = tile_ids
            else:
                assigned = tile_ids

            assignments.append(
                DeviceAssignment(
                    device_id=i,
                    device=self.devices[i],
                    tile_ids=assigned,
                    edge_ids=[],
                )
            )

        return assignments

    def _distribute_tiles(self) -> None:
        """Move tiles to assigned devices."""
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]

                if tile.activity is not None:
                    tile.activity = tile.activity.to(assignment.device)
                if tile.prediction is not None:
                    tile.prediction = tile.prediction.to(assignment.device)
                if tile.error is not None:
                    tile.error = tile.error.to(assignment.device)

        for assignment in self.assignments:
            for edge_key in assignment.edge_ids:
                weight, bias = self.model._get_edge_params(*edge_key)
                if weight is not None:
                    weight.data = weight.data.to(assignment.device)
                if bias is not None:
                    bias.data = bias.data.to(assignment.device)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Training step with distributed execution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        if self.n_devices == 1:
            if self.mp_trainer:
                return self._train_step_mixed_precision(x, y)
            return self.model.train_step(x, y)

        start_time = time.perf_counter()

        batch_size = x.shape[0]
        device = self.devices[0]

        x = x.to(device)
        y = y.to(device)

        input_proj = self.model.W_in(x)

        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]

                if tile.is_input:
                    idx = self.model.graph.input_tile_ids.index(tile.id)
                    start = idx * self.model.config.neurons_per_tile
                    tile.activity = input_proj[:, start : start + tile.neurons].clone()
                else:
                    tile.activity = torch.zeros(
                        batch_size, tile.neurons, device=assignment.device
                    )
                tile.prediction = None
                tile.error = None

        for _ in range(self.model.config.inference_steps):
            self._unified_relax_step(batch_size)

        stats = self._unified_learning(y)

        elapsed = time.perf_counter() - start_time
        stats["total_time"] = elapsed
        stats["comm_time"] = self._comm_time
        stats["compute_time"] = self._compute_time
        stats["n_devices"] = self.n_devices

        return stats

    def _train_step_mixed_precision(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Training step with mixed precision (single-GPU path).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        if self.mp_trainer is None:
            return self.model.train_step(x, y)

        with self.mp_trainer.autocast():
            stats = self.model.train_step(x, y)

        return stats

    # ------------------------------------------------------------------
    # Relaxation loop
    # ------------------------------------------------------------------

    def _unified_relax_step(self, batch_size: int) -> None:
        """One relaxation step with optional async communication.

        Parameters
        ----------
        batch_size : int
            Batch size
        """
        compute_start = time.perf_counter()

        for assignment in self.assignments:
            if self.executor:
                with self.executor.compute_stream_context():
                    self._compute_predictions_device(batch_size, assignment)
            else:
                self._compute_predictions_device(batch_size, assignment)

        if self.executor:
            self.executor.synchronize()

        self._compute_time = time.perf_counter() - compute_start

        # Compute errors locally
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]
                if tile.activity is None:
                    continue

                if tile.prediction is None:
                    tile.error = tile.activity.clone()
                else:
                    tile.error = tile.activity - tile.prediction

        # Exchange boundary activities
        comm_start = time.perf_counter()

        for assignment in self.assignments:
            activities = {
                tile_id: self.model.graph.tiles[tile_id].activity
                for tile_id in assignment.tile_ids
                if self.model.graph.tiles[tile_id].activity is not None
            }

            received = self.communicator.exchange_activities(
                activities, assignment.device_id
            )

            for local_tile, remote_activity in received.items():
                tile = self.model.graph.tiles[local_tile]
                if tile.prediction is not None:
                    pass  # Would need edge weights for full implementation

        if self.executor:
            self.executor.synchronize()

        self._comm_time = time.perf_counter() - comm_start

        # Update activities
        for assignment in self.assignments:
            for tile_id in assignment.tile_ids:
                tile = self.model.graph.tiles[tile_id]
                if tile.is_input or tile.error is None:
                    continue

                tile_idx = list(self.model.graph.tiles.keys()).index(tile.id)
                imp = torch.sigmoid(self.model.tile_importance[tile_idx]).item()

                fwd_feedback = []
                for dst_id in tile.fwd_neighbors:
                    dst = self.model.graph.tiles[dst_id]
                    weight, _ = self.model._get_edge_params(tile.id, dst_id)
                    if weight is not None and dst.error is not None:
                        fwd_feedback.append(dst.error @ weight.T)

                tile.activity = compute_activity_update(
                    activity=tile.activity,
                    error=tile.error,
                    fwd_feedback=fwd_feedback,
                    importance=imp,
                    step_size=self.model.config.step_size,
                    lambda_error=self.model.config.lambda_error,
                    clamp_min=self.model.config.activity_clamp_min,
                    clamp_max=self.model.config.activity_clamp_max,
                    clamp=self.model.config.clamp_activities,
                )

    def _compute_predictions_device(
        self,
        batch_size: int,
        assignment: DeviceAssignment,
    ) -> None:
        """Compute predictions for tiles on a device.

        Parameters
        ----------
        batch_size : int
            Batch size
        assignment : DeviceAssignment
            Device assignment for this device
        """
        device = assignment.device

        for tile_id in assignment.tile_ids:
            tile = self.model.graph.tiles[tile_id]
            if tile.is_input:
                continue

            inputs = []
            total_bias = None

            for src_id in tile.bwd_neighbors:
                src = self.model.graph.tiles[src_id]
                weight, bias = self.model._get_edge_params(src_id, tile.id)

                if weight is None:
                    continue

                src_activity = (
                    src.activity
                    if src.activity is not None
                    else torch.zeros(batch_size, src.neurons, device=device)
                )
                inputs.append(self.model._apply_activation(src_activity) @ weight)

                if bias is not None:
                    total_bias = bias if total_bias is None else total_bias + bias

            tile.prediction = compute_tile_prediction(
                inputs,
                total_bias,
                output_shape=(batch_size, tile.neurons),
                device=device,
            )

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _unified_learning(self, y: torch.Tensor) -> dict[str, float]:
        """Learning step: gather outputs, compute loss, update weights.

        Parameters
        ----------
        y : torch.Tensor
            Target tensor

        Returns
        -------
        dict
            Training statistics
        """
        w_out_device = self.model.W_out.weight.device
        out_activities_list = []

        for tid in self.model.graph.output_tile_ids:
            act = self.model.graph.tiles[tid].activity
            if act is not None:
                out_activities_list.append(act.to(w_out_device))

        if not out_activities_list:
            batch_size = y.shape[0] if y.dim() > 0 else 1
            out_dim = self.model.W_out.in_features
            out_activities = torch.zeros(batch_size, out_dim, device=w_out_device)
        else:
            out_activities = torch.cat(out_activities_list, dim=-1)

        logits = self.model.W_out(out_activities)
        loss = self.model.task_handler.compute_loss(logits, y)

        # Backprop for I/O projections with optional gradient sync
        self.model._ensure_local_optimizers()
        self.model._optim_io.zero_grad()
        loss.backward()

        if self.n_devices > 1:
            for param in self.model.W_in.parameters():
                if param.grad is not None:
                    self.nccl_communicator.all_reduce(param.grad)
            for param in self.model.W_out.parameters():
                if param.grad is not None:
                    self.nccl_communicator.all_reduce(param.grad)

        self.model._optim_io.step()

        # Local Hebbian updates per device
        for assignment in self.assignments:
            for edge_key in assignment.edge_ids:
                weight, bias = self.model._get_edge_params(*edge_key)
                if weight is None:
                    continue

                src = self.model.graph.tiles[edge_key[0]]
                dst = self.model.graph.tiles[edge_key[1]]

                if src.activity is None or dst.error is None:
                    continue

                edge_idx = self.model.graph.edges.index(edge_key)
                imp = torch.sigmoid(self.model.edge_importance[edge_idx]).item()

                src_act = self.model._apply_activation(src.activity)
                dst_err = dst.error

                batch_size = src_act.shape[0]
                weight_update, bias_update = compute_hebbian_update(
                    src_act, dst_err, imp, batch_size
                )

                if weight is not None:
                    weight.data = weight.data - self.model.config.learning_rate * (
                        weight_update + self.model.config.weight_decay * weight.data
                    )
                if bias is not None:
                    bias.data = (
                        bias.data - self.model.config.learning_rate * bias_update
                    )

        # Also handle edges not explicitly in assignments (e.g. in single-device mode
        # or when edge_ids are empty — iterate all edges)
        if not any(a.edge_ids for a in self.assignments):
            for edge_key in self.model.graph.edges:
                if edge_key in self._processed_edges(y):
                    continue
                weight, bias = self.model._get_edge_params(*edge_key)
                if weight is None:
                    continue

                src = self.model.graph.tiles[edge_key[0]]
                dst = self.model.graph.tiles[edge_key[1]]

                if src.activity is None or dst.error is None:
                    continue

                edge_idx = self.model.graph.edges.index(edge_key)
                imp = torch.sigmoid(self.model.edge_importance[edge_idx]).item()

                src_act = self.model._apply_activation(src.activity)
                dst_err = dst.error

                batch_size = src_act.shape[0]
                weight_update, bias_update = compute_hebbian_update(
                    src_act, dst_err, imp, batch_size
                )

                if weight is not None:
                    weight.data = weight.data - self.model.config.learning_rate * (
                        weight_update + self.model.config.weight_decay * weight.data
                    )
                if bias is not None:
                    bias.data = (
                        bias.data - self.model.config.learning_rate * bias_update
                    )

        accuracy = self.model.task_handler.compute_metrics(logits, y)

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "mode": self.model.equitile_config.mode,
            "distributed": True,
            "n_devices": self.n_devices,
        }

    def _processed_edges(self, y: torch.Tensor) -> set[tuple[int, int]]:
        """Return edges already processed by assignment-based loop."""
        return set()

    # ------------------------------------------------------------------
    # Tile growth & pruning
    # ------------------------------------------------------------------

    def grow_tile(self, parent_tile_id: int) -> int:
        """Add a new tile as a child of an existing tile.

        Parameters
        ----------
        parent_tile_id : int
            Parent tile ID

        Returns
        -------
        int
            New tile ID (-1 if failed)
        """
        if not self.growth_config.growth_enabled:
            return -1

        parent = self.model.graph.tiles[parent_tile_id]

        new_id = self.model.add_tile(
            neurons=parent.neurons,
            layer_id=parent.layer_id + 1,
            pos_x=parent.pos_x,
            pos_y=parent.pos_y,
            is_input=False,
            is_output=False,
        )

        self.model.add_edge(parent_tile_id, new_id)

        parent_device = self.devices[0]
        for assignment in self.assignments:
            if parent_tile_id in assignment.tile_ids:
                parent_device = assignment.device
                assignment.tile_ids.append(new_id)
                assignment.edge_ids.append((parent_tile_id, new_id))
                break

        tile = self.model.graph.tiles[new_id]
        if tile.activity is not None:
            tile.activity = tile.activity.to(parent_device)
        if tile.prediction is not None:
            tile.prediction = tile.prediction.to(parent_device)
        if tile.error is not None:
            tile.error = tile.error.to(parent_device)

        weight, bias = self.model._get_edge_params(parent_tile_id, new_id)
        if weight is not None:
            weight.data = weight.data.to(parent_device)
        if bias is not None:
            bias.data = bias.data.to(parent_device)

        self._steps_since_modify = 0
        return new_id

    def prune_tile(self, tile_id: int) -> bool:
        """Remove a tile and its connections.

        Parameters
        ----------
        tile_id : int
            Tile ID to remove

        Returns
        -------
        bool
            Whether tile was pruned
        """
        if not self.growth_config.prune_enabled:
            return False

        tile = self.model.graph.tiles.get(tile_id)
        if tile is None or tile.is_input or tile.is_output:
            return False

        edges_to_remove = [
            (src, dst) for (src, dst) in self.model.graph.edges if tile_id in (src, dst)
        ]

        self.model.remove_tile(tile_id)

        for assignment in self.assignments:
            if tile_id in assignment.tile_ids:
                assignment.tile_ids.remove(tile_id)
            assignment.edge_ids = [
                e for e in assignment.edge_ids if e not in edges_to_remove
            ]

        self._steps_since_modify = 0
        return True

    def maybe_modify_tiles(self, errors: dict[int, float]) -> dict[str, int]:
        """Check if tiles should be grown or pruned.

        Parameters
        ----------
        errors : dict
            Error values per tile

        Returns
        -------
        dict
            Modification counts
        """
        stats: dict[str, int] = {"grown": 0, "pruned": 0}

        if not self.growth_config.growth_enabled:
            return stats

        self._steps_since_modify += 1
        if self._steps_since_modify < self.growth_config.growth_cooldown:
            return stats

        for tile_id, error in errors.items():
            if error > self.growth_config.growth_threshold:
                if len(self.model.graph.tiles) < self.growth_config.max_tiles:
                    new_id = self.grow_tile(tile_id)
                    if new_id >= 0:
                        stats["grown"] += 1
                        break

        for tile_id, error in errors.items():
            if error < self.growth_config.prune_threshold:
                if len(self.model.graph.tiles) > self.growth_config.min_tiles:
                    if self.prune_tile(tile_id):
                        stats["pruned"] += 1
                        break

        return stats

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.n_devices > 1

    @property
    def is_mixed_precision(self) -> bool:
        """Check if mixed precision is enabled."""
        return self.mp_trainer is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Clean up resources (async executor, NCCL)."""
        if self.executor:
            self.executor.stop()
        self.nccl_communicator.destroy()

    def __del__(self) -> None:
        self.destroy()


# =============================================================================
# Multi-Process Spawn Helper
# =============================================================================


def spawn_distributed_worker(
    worker_fn: Callable[[int, int], None],
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "29500",
) -> None:
    """Spawn multi-GPU worker processes.

    Parameters
    ----------
    worker_fn : callable
        Worker function with signature ``(rank, world_size)``
    world_size : int
        Number of processes to spawn
    master_addr : str
        Master node address
    master_port : str
        Master node port

    Examples
    --------
    >>> def worker(rank, world_size):
    ...     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    ...     model = EquiTile(...)
    ...     dist_model = DistributedEquiTile(model)
    ...     ...
    >>> spawn_distributed_worker(worker, world_size=4)
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    mp.spawn(worker_fn, args=(world_size,), nprocs=world_size, join=True)


# =============================================================================
# Factory Functions
# =============================================================================


def create_distributed_model(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int = 784,
    output_dim: int = 10,
    device_ids: list[int] | None = None,
    mixed_precision: bool = True,
    tile_balance: str = "round_robin",
    **kwargs,
) -> tuple[EquiTile, DistributedEquiTile]:
    """Create a distributed EquiTile model.

    Parameters
    ----------
    neurons_per_tile : int
        Neurons per tile
    num_layers : int
        Number of layers
    tiles_per_layer : int
        Tiles per layer
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    device_ids : list of int, optional
        GPU device IDs
    mixed_precision : bool
        Enable mixed precision
    tile_balance : str
        Tile balancing strategy
    **kwargs
        Additional arguments for EquiTile

    Returns
    -------
    tuple of (EquiTile, DistributedEquiTile)
        Base model and distributed wrapper
    """
    from bioplausible.equitile.core import EquiTile

    model = EquiTile(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs,
    )

    dist_model = DistributedEquiTile(
        model,
        config=DistributedConfig(
            device_ids=device_ids or [],
            mixed_precision=mixed_precision,
            tile_balance=tile_balance,
        ),
    )

    return model, dist_model
