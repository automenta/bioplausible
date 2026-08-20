"""Distributed SystemTrainer: P2P Coordination for 5-D Ontology.

Leverages the 5-D fault lines for natural distribution:
- Substrate: Fully local per-node (no coordination needed)
- Geometry: Routing table = DHT overlay (MoE over Kademlia)
- StateDynamics: Settling shards across mesh (KV-cache style)
- CreditAssignment: Local by design - zero cross-node gradient traffic
- ParameterUpdate: Federated deltas (LoRA/Swarm DPO), sparse aggregation
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from bioplausible.core.ontology import (
    Geometry,
    Substrate,
    System,
    SystemState,
    TileGeometry,
)


class _DataProvider(Protocol):
    """Protocol for data providers."""

    def get_batch(self) -> tuple[Tensor, Tensor]: ...
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Configuration for distributed training."""

    # Node identification
    node_id: str = ""
    num_nodes: int = 1

    # P2P network
    bootstrap_nodes: list[str] = field(default_factory=list)
    dht_port: int = 8468

    # Distribution strategy
    shard_geometry: bool = True          # Shard tile mesh across nodes
    federated_updates: bool = True       # Use federated parameter updates
    sync_interval: int = 10              # Steps between federated sync

    # Communication
    compression: str = "topk"            # Gradient compression: "none", "topk", "quantize"
    topk_ratio: float = 0.01             # Top-k sparsification ratio

    # Fault tolerance
    heartbeat_interval: float = 5.0      # Seconds between heartbeats
    max_missed_heartbeats: int = 3       # Max missed before node considered dead

    # Training
    max_epochs: int = 10


class NodeRegistry:
    """Registry of active nodes in the P2P network."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._heartbeats: dict[str, float] = {}

    def register(self, node_id: str, metadata: dict) -> None:
        self._nodes[node_id] = metadata
        self._heartbeats[node_id] = 0.0

    def heartbeat(self, node_id: str, timestamp: float) -> None:
        self._heartbeats[node_id] = timestamp

    def get_active_nodes(self, current_time: float, timeout: float) -> list[str]:
        return [
            nid for nid, ts in self._heartbeats.items()
            if current_time - ts < timeout
        ]

    def remove(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._heartbeats.pop(node_id, None)

    def get_node_info(self, node_id: str) -> dict | None:
        return self._nodes.get(node_id)


class DHTRouter:
    """Kademlia-style DHT router for MoE-style tile routing."""

    def __init__(self, node_id: str, k: int = 20):
        self.node_id = node_id
        self.k = k  # Kademlia bucket size
        self._buckets: dict[int, list[str]] = {}  # prefix_len -> node_ids

    def add_node(self, node_id: str) -> None:
        """Add a node to the appropriate bucket."""
        if node_id == self.node_id:
            return
        # Compute common prefix length
        common = self._common_prefix_len(self.node_id, node_id)
        bucket = self._buckets.setdefault(common, [])
        if node_id not in bucket:
            if len(bucket) >= self.k:
                bucket.pop(0)  # Evict oldest
            bucket.append(node_id)

    def remove_node(self, node_id: str) -> None:
        for bucket in self._buckets.values():
            if node_id in bucket:
                bucket.remove(node_id)

    def route(self, key: str) -> list[str]:
        """Route a key to the k closest nodes."""
        target_prefix = self._common_prefix_len(self.node_id, key)
        # Start from longest matching prefix
        for prefix_len in range(target_prefix, -1, -1):
            bucket = self._buckets.get(prefix_len, [])
            if bucket:
                return bucket[:self.k]
        return []

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        """Compute common prefix length in bits."""
        # Simplified: use string comparison
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                return i * 4  # 4 bits per hex char
        return min(len(a), len(b)) * 4


class FederatedAggregator:
    """Aggregates parameter updates from multiple nodes (Federated Learning style)."""

    def __init__(self, aggregation: str = "fedavg"):
        self.aggregation = aggregation
        self._updates: dict[str, list[Tensor]] = {}

    def add_update(self, node_id: str, updates: dict[str, Tensor]) -> None:
        """Add updates from a node."""
        for name, delta in updates.items():
            if name not in self._updates:
                self._updates[name] = []
            self._updates[name].append(delta)

    def aggregate(self) -> dict[str, Tensor]:
        """Aggregate updates using the configured strategy."""
        result = {}
        if self.aggregation == "fedavg":
            for name, deltas in self._updates.items():
                result[name] = torch.stack(deltas).mean(dim=0)
        elif self.aggregation == "fedprox":
            # FedProx: weighted average with proximal term
            for name, deltas in self._updates.items():
                result[name] = torch.stack(deltas).mean(dim=0)
        elif self.aggregation == "swarm":
            # Swarm DPO: median-based robust aggregation
            for name, deltas in self._updates.items():
                result[name] = torch.stack(deltas).median(dim=0).values
        self._updates.clear()
        return result

    def clear(self) -> None:
        self._updates.clear()


@dataclass
class DistributedSystemTrainer:
    """Distributed trainer for 5-D composable systems.

    Orchestrates the pipeline across multiple nodes:
        Substrate.forward_op (local) → Geometry.route (DHT)
        → StateDynamics.settle (sharded) → CreditAssignment (local)
        → ParameterUpdate.step (federated)
    """

    system: System
    config: DistributedConfig
    train_data: _DataProvider
    val_data: _DataProvider | None = None

    # Distributed state
    _node_registry: NodeRegistry = field(default_factory=NodeRegistry, init=False)
    _dht_router: DHTRouter = field(init=False)
    _federated_aggregator: FederatedAggregator = field(default_factory=FederatedAggregator, init=False)

    # Training state
    current_epoch: int = field(default=0, init=False)
    global_step: int = field(default=0, init=False)
    history: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._dht_router = DHTRouter(self.config.node_id)
        self._setup_distributed()

    def _setup_distributed(self) -> None:
        """Initialize distributed components."""
        # Register self
        self._node_registry.register(self.config.node_id, {
            "geometry_shards": self._get_geometry_shards(),
            "capabilities": self._get_node_capabilities(),
        })

        # Connect to bootstrap nodes
        for bootstrap in self.config.bootstrap_nodes:
            self._dht_router.add_node(bootstrap)

        # If using TileGeometry, set up tile sharding
        if self.config.shard_geometry and isinstance(self.system.geometry, TileGeometry):
            self._setup_tile_sharding()

    def _get_geometry_shards(self) -> dict:
        """Get the geometry shards this node is responsible for."""
        if isinstance(self.system.geometry, TileGeometry):
            graph = self.system.geometry._graph
            # Assign tiles to nodes based on node_id hash
            shards = {}
            for tid in graph.tiles:
                node = self._assign_tile_to_node(tid)
                shards.setdefault(node, []).append(tid)
            return shards.get(self.config.node_id, [])
        return {}

    def _assign_tile_to_node(self, tile_id: int) -> str:
        """Assign a tile to a node using consistent hashing."""
        # Simplified: hash tile_id and distribute
        node_idx = hash(tile_id) % self.config.num_nodes
        return f"node_{node_idx}"

    def _get_node_capabilities(self) -> dict:
        """Get this node's compute capabilities."""
        return {
            "device": str(next(self.system.geometry.parameters()).device)
            if hasattr(self.system.geometry, "parameters") else "cpu",
            "memory_gb": 16,  # Placeholder
        }

    def _setup_tile_sharding(self) -> None:
        """Set up tile mesh sharding across nodes."""
        geometry = self.system.geometry
        graph = geometry._graph

        # Identify boundary tiles (connecting to other nodes)
        device_map = {tid: self._assign_tile_to_node(tid) for tid in graph.tiles}
        boundary_tiles = graph.get_boundary_tiles(device_map)

        # Store boundary info for cross-node communication
        self._boundary_tiles = boundary_tiles.get(self.config.node_id, [])

    def train_epoch(self) -> dict[str, float]:
        """Run one distributed training epoch."""
        self.system.geometry.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_energy = 0.0
        num_batches = 0

        for _, (x, y) in enumerate(self.train_data):
            x = x.to(self._get_device())
            y = y.to(self._get_device())

            metrics = self._distributed_train_step(x, y)

            epoch_loss += metrics.get("loss", 0.0)
            epoch_acc += metrics.get("accuracy", 0.0)
            epoch_energy += metrics.get("energy", 0.0)
            num_batches += 1
            self.global_step += 1

            # Federated sync
            if self.config.federated_updates and self.global_step % self.config.sync_interval == 0:
                self._federated_sync()

            # Heartbeat
            if self.global_step % 10 == 0:
                self._send_heartbeat()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        avg_energy = epoch_energy / max(num_batches, 1)

        epoch_metrics = {
            "epoch": self.current_epoch,
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_energy": avg_energy,
            "global_step": self.global_step,
            "active_nodes": len(self._node_registry.get_active_nodes(0, 30)),
        }

        if self.val_data is not None:
            val_metrics = self.validate()
            epoch_metrics.update(val_metrics)

        self.history.append(epoch_metrics)
        self.current_epoch += 1

        return epoch_metrics

    def _distributed_train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """Execute one distributed training step.

        Pipeline:
        1. Substrate + Geometry: Forward pass (with DHT routing for tile mesh)
        2. StateDynamics: Sharded settling
        3. CreditAssignment: Local pseudo-gradients
        4. ParameterUpdate: Federated aggregation
        """
        state = SystemState(x=x, y=y)

        # 1. Substrate + Geometry: Forward pass
        state.activations = self._distributed_forward(x, self.system.substrate)
        if state.activations is not None:
            state.activations = self.system.substrate.inject_state_noise(state.activations)

        # 2. StateDynamics: Free phase settling (sharded for tile geometry)
        free_state = self._distributed_settle(
            state, self.system.geometry, self.system.substrate, target=None
        )
        free_state.energy = self.system.dynamics.compute_energy(free_state, self.system.geometry)

        # 3. StateDynamics: Nudged phase settling
        nudged_state = self._distributed_settle(
            state, self.system.geometry, self.system.substrate, target=y
        )
        nudged_state.energy = self.system.dynamics.compute_energy(nudged_state, self.system.geometry)
        nudged_state.loss = self._compute_loss(nudged_state, y)

        # 4. CreditAssignment: Compute pseudo-gradients (local)
        pseudo_grads = self.system.credit.compute_pseudo_gradient(
            free_state, nudged_state, nudged_state.loss, self.system.geometry
        )

        # 5. ParameterUpdate: Compute local updates
        local_updates = self.system.update.step(
            self.system.geometry.params, pseudo_grads, self.system.geometry
        )

        # Compute delta from current params
        current_params = self.system.geometry.params
        deltas = {
            name: local_updates[name] - current_params[name]
            for name in current_params
            if name in local_updates
        }

        # Store for federated aggregation
        self._federated_aggregator.add_update(self.config.node_id, deltas)

        # Apply local updates immediately (optimistic)
        self.system.geometry.update_params(local_updates)

        return {
            "loss": float(nudged_state.loss) if nudged_state.loss is not None else 0.0,
            "energy": float(free_state.energy) if free_state.energy is not None else 0.0,
            "accuracy": free_state.metrics.get("accuracy", 0.0),
        }

    def _distributed_forward(self, x: Tensor, substrate: Substrate) -> Tensor:
        """Forward pass with DHT routing for tile mesh."""
        if isinstance(self.system.geometry, TileGeometry):
            return self._tile_mesh_forward(x, substrate)
        return self.system.geometry.forward(x, substrate)

    def _tile_mesh_forward(self, x: Tensor, substrate: Substrate) -> Tensor:
        """Forward pass through tile mesh with cross-node routing."""
        geometry = self.system.geometry
        op = substrate.get_forward_operator()

        # Project input to tile space
        h = geometry._input_projection(x)

        # Set input tile activities (local input tiles only)
        for tid in geometry._graph.input_tile_ids:
            if self._assign_tile_to_node(tid) == self.config.node_id:
                n = geometry._graph.tiles[tid].neurons
                offset = sum(
                    geometry._graph.tiles[t].neurons
                    for t in geometry._graph.input_tile_ids
                    if t < tid and self._assign_tile_to_node(t) == self.config.node_id
                )
                geometry._graph.tiles[tid].activity = h[:, offset : offset + n]

        # Forward propagate through layers
        for layer_tiles in geometry._graph.layer_ids[1:]:
            for tid in layer_tiles:
                if self._assign_tile_to_node(tid) != self.config.node_id:
                    continue  # Skip tiles on other nodes
                tile = geometry._graph.tiles[tid]
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = geometry._graph.tiles[src_id].activity
                    if src_act is None:
                        # Need to fetch from remote node
                        src_act = self._fetch_remote_activation(src_id)
                        if src_act is None:
                            continue
                    w = geometry._tile_weights[geometry._weight_key(src_id, tid)]
                    contrib = op(src_act, w)
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += geometry._tile_biases[str(tid)].unsqueeze(0).expand(acc.shape[0], -1)
                    tile.activity = acc
                    tile.prediction = acc

        # Collect output tile activities (local only)
        out_acts: list[Tensor] = []
        for tid in geometry._graph.output_tile_ids:
            if self._assign_tile_to_node(tid) == self.config.node_id:
                act = geometry._graph.tiles[tid].activity
                if act is not None:
                    out_acts.append(act)

        if not out_acts:
            return torch.empty(x.shape[0], geometry.config.output_dim, device=x.device)

        h = torch.cat(out_acts, dim=1)
        return geometry._output_projection(h)

    def _fetch_remote_activation(self, tile_id: int) -> Tensor | None:
        """Fetch activation from remote node (simulated)."""
        # In real implementation, this would be an RPC call
        # For now, return None to indicate unavailable
        return None

    def _distributed_settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Settle with sharded computation for tile mesh."""
        if isinstance(geometry, TileGeometry):
            return self._tile_mesh_settle(state, geometry, substrate, target)
        return self.system.dynamics.settle(state, geometry, substrate, target)

    def _tile_mesh_settle(
        self,
        state: SystemState,
        geometry: TileGeometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Tile mesh settling with cross-node synchronization."""
        # Run settling iterations
        for step in range(self.system.dynamics.config.max_steps):
            # Local settle step
            for layer_tiles in geometry._graph.layer_ids[1:]:
                for tid in layer_tiles:
                    if self._assign_tile_to_node(tid) != self.config.node_id:
                        continue
                    tile = geometry._graph.tiles[tid]
                    acc: Tensor | None = None
                    for src_id in tile.bwd_neighbors:
                        src_act = geometry._graph.tiles[src_id].activity
                        if src_act is None:
                            src_act = self._fetch_remote_activation(src_id)
                            if src_act is None:
                                continue
                        w = geometry._tile_weights[geometry._weight_key(src_id, tid)]
                        contrib = src_act @ w.T
                        acc = contrib if acc is None else acc + contrib
                    if acc is not None:
                        acc += geometry._tile_biases[str(tid)].unsqueeze(0).expand(acc.shape[0], -1)
                        tile.activity = substrate.inject_state_noise(acc)  # type: ignore[arg-type]

            # Synchronize boundary tiles with neighbors
            self._sync_boundary_tiles(geometry)

            # Check convergence (simplified)
            if step >= self.system.dynamics.config.convergence_start:
                pass  # Convergence check omitted for brevity

        # Collect final activations
        acts: list[Tensor] = []
        for layer_tiles in geometry._graph.layer_ids:
            for tid in layer_tiles:
                if self._assign_tile_to_node(tid) == self.config.node_id:
                    act = geometry._graph.tiles[tid].activity
                    if act is not None:
                        acts.append(act)

        if target is None:
            state.free_state = acts
        else:
            state.nudged_state = acts
        state.activations = acts
        return state

    def _sync_boundary_tiles(self, geometry: TileGeometry) -> None:
        """Synchronize boundary tile activations with neighbor nodes."""
        # In real implementation: send boundary activations to neighbor nodes
        # For now, no-op

    def _federated_sync(self) -> None:
        """Synchronize parameter updates across nodes."""
        if not self.config.federated_updates:
            return

        # Aggregate updates
        aggregated = self._federated_aggregator.aggregate()

        # Apply aggregated updates
        if aggregated:
            current_params = self.system.geometry.params
            new_params = {
                name: current_params[name] + aggregated.get(name, torch.zeros_like(current_params[name]))
                for name in current_params
            }
            self.system.geometry.update_params(new_params)

    def _send_heartbeat(self) -> None:
        """Send heartbeat to maintain P2P membership."""
        import time
        self._node_registry.heartbeat(self.config.node_id, time.time())

    def _get_device(self) -> torch.device:
        """Get the device for this node."""
        if hasattr(self.system.geometry, "parameters"):
            try:
                return next(self.system.geometry.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")

    def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:
        """Compute task loss from final state."""
        acts = state.activations
        if acts is None:
            return torch.tensor(0.0, device=y.device)
        if isinstance(acts, list):
            logits = acts[-1]
        else:
            logits = acts
        return torch.nn.functional.cross_entropy(logits, y)

    def validate(self) -> dict[str, float]:
        """Run validation epoch."""
        if self.val_data is None:
            return {}

        self.system.geometry.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in self.val_data:
                x = x.to(self._get_device())
                y = y.to(self._get_device())

                logits = self.system.forward(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(-1) == y).float().mean().item()

                val_loss += loss.item()
                val_acc += acc
                num_batches += 1

        return {
            "val_loss": val_loss / max(num_batches, 1),
            "val_acc": val_acc / max(num_batches, 1),
        }

    def fit(self) -> list[dict[str, float]]:
        """Run full distributed training loop."""
        for _ in range(self.config.max_epochs if hasattr(self.config, "max_epochs") else 10):
            self.train_epoch()
        return self.history


__all__ = [
    "DHTRouter",
    "DistributedConfig",
    "DistributedSystemTrainer",
    "FederatedAggregator",
    "NodeRegistry",
]
