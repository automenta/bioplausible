"""gRPC service implementation for Tile Mesh P2P communication."""

from __future__ import annotations

import asyncio
import pickle
import threading
import time
from concurrent import futures
from typing import TYPE_CHECKING, Protocol

import grpc
import torch

from bioplausible.core.logging import get_logger

if TYPE_CHECKING:
    from bioplausible.core.ontology import TileGeometry
    from bioplausible.p2p.proto import tile_mesh_pb2, tile_mesh_pb2_grpc

    class _TensorProto(Protocol):
        data: bytes
        shape: list[int]
        dtype: str

    class _TileActivationRequest(Protocol):
        tile_id: int
        request_id: int

    class _TileActivationResponse(Protocol):
        tile_id: int
        request_id: int
        activation: _TensorProto
        success: bool
        error: str

    class _BoundarySyncRequest(Protocol):
        source_node_id: str
        boundary_activations: list[_TensorProto]
        boundary_tile_ids: list[int]
        step: int

    class _BoundarySyncResponse(Protocol):
        success: bool
        error: str

    class _HeartbeatRequest(Protocol):
        node_id: str
        timestamp: int
        metadata: dict[str, str]

    class _HeartbeatResponse(Protocol):
        success: bool
        active_nodes: list[str]

    class _ParameterUpdateRequest(Protocol):
        node_id: str
        step: int
        updates: dict[str, _TensorProto]

    class _ParameterUpdateResponse(Protocol):
        success: bool
        aggregated_updates: dict[str, _TensorProto]
        error: str

else:
    # Runtime imports
    try:
        from bioplausible.p2p.proto import tile_mesh_pb2, tile_mesh_pb2_grpc
    except ImportError:
        tile_mesh_pb2 = None  # type: ignore[assignment]
        tile_mesh_pb2_grpc = None  # type: ignore[assignment]

logger = get_logger("TileMeshGRPC")


def _tensor_to_proto(tensor: torch.Tensor) -> tile_mesh_pb2.TensorProto:
    """Convert torch.Tensor to TensorProto."""
    return tile_mesh_pb2.TensorProto(  # type: ignore[union-attr]
        data=pickle.dumps(tensor.detach().cpu()),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
    )


def _proto_to_tensor(proto: tile_mesh_pb2.TensorProto) -> torch.Tensor:
    """Convert TensorProto to torch.Tensor."""
    tensor = pickle.loads(proto.data)  # type: ignore[union-attr]
    return tensor.to(dtype=getattr(torch, proto.dtype))  # type: ignore[union-attr]


# Type alias for the servicer base class
if tile_mesh_pb2_grpc is not None:
    _ServicerBase = tile_mesh_pb2_grpc.TileMeshServiceServicer
else:
    _ServicerBase = object


class TileMeshServicer(_ServicerBase):
    """gRPC servicer for TileMeshService."""

    def __init__(
        self,
        geometry: TileGeometry,
        node_id: str,
        device: torch.device,
    ):
        self.geometry = geometry
        self.node_id = node_id
        self.device = device
        self._remote_activations: dict[int, torch.Tensor] = {}
        self._boundary_cache: dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()

    def FetchTileActivation(  # ruff: ignore[invalid-function-name] - gRPC method name must match proto
        self,
        request: tile_mesh_pb2.TileActivationRequest,
        context: grpc.ServicerContext,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
    ) -> tile_mesh_pb2.TileActivationResponse:
        """Serve tile activation to remote node."""
        try:
            tile_id = request.tile_id
            tile = self.geometry._graph.tiles.get(tile_id)

            if tile is None or tile.activity is None:
                return tile_mesh_pb2.TileActivationResponse(  # type: ignore[union-attr]
                    tile_id=tile_id,
                    request_id=request.request_id,
                    success=False,
                    error=f"Tile {tile_id} not found or no activity",
                )

            # Ensure tensor is on CPU for serialization
            activation = tile.activity.detach().cpu()

            return tile_mesh_pb2.TileActivationResponse(  # type: ignore[union-attr]
                tile_id=tile_id,
                request_id=request.request_id,
                activation=_tensor_to_proto(activation),
                success=True,
            )
        except Exception as e:
            logger.exception("FetchTileActivation error for tile %s", request.tile_id)
            return tile_mesh_pb2.TileActivationResponse(  # type: ignore[union-attr]
                tile_id=request.tile_id,
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    def SyncBoundaryTiles(  # ruff: ignore[invalid-function-name] - gRPC method name must match proto
        self,
        request: tile_mesh_pb2.BoundarySyncRequest,
        context: grpc.ServicerContext,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
    ) -> tile_mesh_pb2.BoundarySyncResponse:
        """Receive boundary tile activations from neighbor."""
        try:
            with self._lock:
                for tile_id, activation_proto in zip(
                    request.boundary_tile_ids, request.boundary_activations
                ):
                    tensor = _proto_to_tensor(activation_proto).to(self.device)
                    self._boundary_cache[tile_id] = tensor

                    # Also update the tile's activity if it exists locally
                    tile = self.geometry._graph.tiles.get(tile_id)
                    if tile is not None:
                        tile.activity = tensor

            return tile_mesh_pb2.BoundarySyncResponse(success=True)  # type: ignore[union-attr]
        except Exception as e:
            logger.exception("SyncBoundaryTiles error")
            return tile_mesh_pb2.BoundarySyncResponse(  # type: ignore[union-attr]
                success=False, error=str(e)
            )

    def Heartbeat(  # ruff: ignore[invalid-function-name] - gRPC method name must match proto
        self,
        request: tile_mesh_pb2.HeartbeatRequest,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
        context: grpc.ServicerContext,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
    ) -> tile_mesh_pb2.HeartbeatResponse:
        """Handle heartbeat from peer."""
        # In a full implementation, this would update a node registry
        # For now, just acknowledge
        return tile_mesh_pb2.HeartbeatResponse(  # type: ignore[union-attr]
            success=True,
            active_nodes=[self.node_id],  # Simplified
        )

    def PushParameterUpdate(  # ruff: ignore[invalid-function-name, no-self-use] - gRPC method name must match proto
        self,
        request: tile_mesh_pb2.ParameterUpdateRequest,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
        context: grpc.ServicerContext,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
    ) -> tile_mesh_pb2.ParameterUpdateResponse:
        """Receive parameter updates from peer for federated aggregation."""
        # This would integrate with FederatedAggregator
        # For now, just acknowledge
        return tile_mesh_pb2.ParameterUpdateResponse(  # type: ignore[union-attr]
            success=True,
            aggregated_updates={},
        )

    def ExecuteStep(  # ruff: ignore[invalid-function-name] - gRPC method name must match proto
        self,
        request: tile_mesh_pb2.ExecuteStepRequest,
        context: grpc.ServicerContext,  # ruff: ignore[unused-method-argument] - Required by gRPC interface
    ) -> tile_mesh_pb2.ExecuteStepResponse:
        """Execute a single distributed training step on this worker.

        This is a simplified implementation for testing. In a full
        implementation, this would:
        1. Deserialize the SystemState (x, y)
        2. Run the distributed training step locally
        3. Return pseudo-gradients and parameter updates
        """
        try:
            # For testing purposes, we'll return a simple success response
            # with dummy data. The actual implementation would require
            # a full System and DistributedSystemTrainer on each worker.
            return tile_mesh_pb2.ExecuteStepResponse(  # type: ignore[union-attr]
                pseudo_grads_data=b"",
                updates_data=b"",
                loss=0.0,
                energy=0.0,
                success=True,
                error="",
            )
        except Exception as e:
            logger.exception("ExecuteStep error")
            return tile_mesh_pb2.ExecuteStepResponse(  # type: ignore[union-attr]
                pseudo_grads_data=b"",
                updates_data=b"",
                loss=0.0,
                energy=0.0,
                success=False,
                error=str(e),
            )

    def get_boundary_activation(self, tile_id: int) -> torch.Tensor | None:
        """Get cached boundary activation (thread-safe)."""
        with self._lock:
            return self._boundary_cache.get(tile_id)


class GRPCServer:
    """gRPC server for Tile Mesh P2P communication."""

    def __init__(
        self,
        geometry: TileGeometry,
        node_id: str,
        port: int = 50051,
        device: torch.device = torch.device("cpu"),
    ):
        self.geometry = geometry
        self.node_id = node_id
        self.port = port
        self.device = device
        self._server: grpc.aio.Server | None = None
        self._servicer: TileMeshServicer | None = None
        self._running = False

    async def start(self) -> None:
        """Start the gRPC server."""
        if tile_mesh_pb2_grpc is None:
            logger.error("gRPC proto not available, server not started")
            return

        self._servicer = TileMeshServicer(self.geometry, self.node_id, self.device)

        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        tile_mesh_pb2_grpc.add_TileMeshServiceServicer_to_server(
            self._servicer, self._server
        )

        listen_addr = f"[::]:{self.port}"
        bound_port = self._server.add_insecure_port(listen_addr)
        await self._server.start()

        # Update port with actual bound port (if port was 0)
        if self.port == 0:
            self.port = bound_port

        self._running = True
        logger.info("gRPC server started on [::]:%d for node %s", self.port, self.node_id)

    async def stop(self) -> None:
        """Stop the gRPC server."""
        if self._server:
            await self._server.stop(grace=5)
            self._running = False
            logger.info("gRPC server stopped")

    @property
    def servicer(self) -> TileMeshServicer | None:
        return self._servicer


class GRPCClient:
    """gRPC client for Tile Mesh P2P communication."""

    def __init__(
        self,
        target: str,
        node_id: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.target = target
        self.node_id = node_id
        self.device = device
        self._channel: grpc.aio.Channel | None = None
        self._stub: tile_mesh_pb2_grpc.TileMeshServiceStub | None = None

    async def connect(self) -> None:
        """Connect to the gRPC server."""
        if tile_mesh_pb2_grpc is None:
            logger.error("gRPC proto not available, client not connected")
            return

        self._channel = grpc.aio.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        self._stub = tile_mesh_pb2_grpc.TileMeshServiceStub(self._channel)
        logger.info("gRPC client connected to %s", self.target)

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()

    async def fetch_tile_activation(
        self, tile_id: int, request_id: int = 0, timeout: float = 10.0
    ) -> torch.Tensor | None:
        """Fetch activation for a remote tile."""
        if not self._stub:
            return None

        try:
            request = tile_mesh_pb2.TileActivationRequest(  # type: ignore[union-attr]
                tile_id=tile_id, request_id=request_id
            )
            response = await asyncio.wait_for(
                self._stub.FetchTileActivation(request), timeout=timeout
            )

            if response.success and response.activation:
                return _proto_to_tensor(response.activation).to(self.device)

            logger.debug("Fetch failed for tile %s: %s", tile_id, response.error)
            return None
        except TimeoutError:
            logger.warning("Fetch timeout for tile %s from %s", tile_id, self.target)
            return None
        except Exception as e:
            logger.debug("Fetch error for tile %s from %s: %s", tile_id, self.target, e)
            return None

    async def sync_boundary_tiles(
        self,
        boundary_tile_ids: list[int],
        boundary_activations: list[torch.Tensor],
        step: int,
        timeout: float = 5.0,
    ) -> bool:
        """Send boundary tile activations to neighbor."""
        if not self._stub:
            return False

        try:
            request = tile_mesh_pb2.BoundarySyncRequest(  # type: ignore[union-attr]
                source_node_id=self.node_id,
                boundary_activations=[
                    _tensor_to_proto(act.detach().cpu()) for act in boundary_activations
                ],
                boundary_tile_ids=boundary_tile_ids,
                step=step,
            )
            response = await asyncio.wait_for(
                self._stub.SyncBoundaryTiles(request), timeout=timeout
            )
            return response.success
        except Exception as e:
            logger.debug("Sync boundary error to %s: %s", self.target, e)
            return False

    async def heartbeat(self, metadata: dict[str, str] | None = None) -> list[str]:
        """Send heartbeat and get active nodes."""
        if not self._stub:
            return []

        try:
            request = tile_mesh_pb2.HeartbeatRequest(  # type: ignore[union-attr]
                node_id=self.node_id,
                timestamp=int(time.time()),
                metadata=metadata or {},
            )
            response = await asyncio.wait_for(
                self._stub.Heartbeat(request), timeout=5.0
            )
            return list(response.active_nodes)
        except Exception as e:
            logger.debug("Heartbeat error to %s: %s", self.target, e)
            return []

    async def push_parameter_update(
        self, updates: dict[str, torch.Tensor], step: int, timeout: float = 10.0
    ) -> dict[str, torch.Tensor] | None:
        """Push parameter updates for federated aggregation."""
        if not self._stub:
            return None

        try:
            request = tile_mesh_pb2.ParameterUpdateRequest(  # type: ignore[union-attr]
                node_id=self.node_id,
                step=step,
                updates={
                    name: _tensor_to_proto(tensor.detach().cpu())
                    for name, tensor in updates.items()
                },
            )
            response = await asyncio.wait_for(
                self._stub.PushParameterUpdate(request), timeout=timeout
            )

            if response.success and response.aggregated_updates:
                return {
                    name: _proto_to_tensor(proto).to(self.device)
                    for name, proto in response.aggregated_updates.items()
                }
            return None
        except Exception as e:
            logger.debug("Parameter update error to %s: %s", self.target, e)
            return None

    async def execute_step(
        self, state_data: bytes, step: int, seed: int, timeout: float = 30.0
    ) -> tuple[bytes, bytes, float, float] | None:
        """Execute a training step on the remote worker."""
        if not self._stub:
            return None

        try:
            request = tile_mesh_pb2.ExecuteStepRequest(  # type: ignore[union-attr]
                state_data=state_data,
                step=step,
                seed=seed,
            )
            response = await asyncio.wait_for(
                self._stub.ExecuteStep(request), timeout=timeout
            )

            if response.success:
                return (
                    response.pseudo_grads_data,
                    response.updates_data,
                    response.loss,
                    response.energy,
                )
            logger.debug("ExecuteStep failed: %s", response.error)
            return None
        except Exception as e:
            logger.debug("ExecuteStep error to %s: %s", self.target, e)
            return None


class GRPCConnectionPool:
    """Manages gRPC connections to multiple peers."""

    def __init__(
        self,
        geometry: TileGeometry,
        node_id: str,
        port: int = 50051,
        device: torch.device = torch.device("cpu"),
    ):
        self.geometry = geometry
        self.node_id = node_id
        self.port = port
        self.device = device
        self._server: GRPCServer | None = None
        self._clients: dict[str, GRPCClient] = {}
        self._peer_addresses: dict[str, str] = {}  # node_id -> host:port

    async def start_server(self) -> None:
        """Start the local gRPC server."""
        self._server = GRPCServer(self.geometry, self.node_id, self.port, self.device)
        await self._server.start()

    async def stop_server(self) -> None:
        """Stop the local gRPC server."""
        if self._server:
            await self._server.stop()

    def add_peer(self, node_id: str, address: str) -> None:
        """Add a peer to the connection pool."""
        self._peer_addresses[node_id] = address

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer from the connection pool."""
        self._peer_addresses.pop(node_id, None)
        client = self._clients.pop(node_id, None)
        if client:
            asyncio.create_task(client.close())  # ruff: ignore[asyncio-dangling-task] - Fire and forget

    async def get_client(self, node_id: str) -> GRPCClient | None:
        """Get or create a client for a peer."""
        if node_id == self.node_id:
            return None  # Don't connect to self

        if node_id in self._clients:
            return self._clients[node_id]

        address = self._peer_addresses.get(node_id)
        if not address:
            return None

        client = GRPCClient(address, self.node_id, self.device)
        await client.connect()
        self._clients[node_id] = client
        return client

    async def fetch_remote_activation(
        self, node_id: str, tile_id: int
    ) -> torch.Tensor | None:
        """Fetch activation from a remote node."""
        client = await self.get_client(node_id)
        if not client:
            return None
        return await client.fetch_tile_activation(tile_id)

    async def sync_boundary_with_peer(
        self,
        node_id: str,
        boundary_tile_ids: list[int],
        boundary_activations: list[torch.Tensor],
        step: int,
    ) -> bool:
        """Sync boundary tiles with a specific peer."""
        client = await self.get_client(node_id)
        if not client:
            return False
        return await client.sync_boundary_tiles(
            boundary_tile_ids, boundary_activations, step
        )

    async def close_all(self) -> None:
        """Close all connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        await self.stop_server()

    @property
    def servicer(self) -> TileMeshServicer | None:
        return self._server.servicer if self._server else None
