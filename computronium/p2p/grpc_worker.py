"""gRPC Worker Entry Point for Tile Mesh P2P Communication.

This module provides a standalone worker process that runs a gRPC server
for distributed training. It binds to port=0 (OS-assigned dynamic port),
prints the bound port to stdout for parent process to parse, and runs
until SIGTERM is received.
"""

from __future__ import annotations

import argparse
import logging
import signal

import torch

from computronium.core.logging import get_logger
from computronium.ontology import GeometryConfig, TileGeometry
from computronium.p2p.grpc_service import GRPCServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("GRPCWorker")


def _create_dummy_geometry(device: torch.device) -> TileGeometry:
    """Create a minimal TileGeometry for testing."""
    config = GeometryConfig(
        input_dim=32,
        output_dim=10,
        num_layers=4,
        topology_type="tile_mesh",
    )
    geometry = TileGeometry(
        config,
        neurons_per_tile=16,
        tiles_per_layer=2,
    )
    geometry.to(device)
    return geometry


class GRPCWorker:
    """gRPC worker that runs a TileMeshService server."""

    def __init__(
        self,
        node_id: str = "worker_0",
        device: torch.device = torch.device("cpu"),
    ):
        self.node_id = node_id
        self.device = device
        self._server: GRPCServer | None = None
        self._geometry: TileGeometry | None = None
        self._running = False

    async def start(self, port: int = 0) -> int:
        """Start the gRPC server and return the bound port."""
        self._geometry = _create_dummy_geometry(self.device)

        self._server = GRPCServer(
            geometry=self._geometry,
            node_id=self.node_id,
            port=port,
            device=self.device,
        )
        await self._server.start()

        # Get the actual bound port from the server
        if self._server._server is not None:
            # grpc.aio.Server doesn't expose port directly, so we need to track it
            # The port is known since we passed it to add_insecure_port
            # With port=0, the OS assigns a port; we need to extract it
            # For now, we'll use the port passed in (0 means OS-assigned)
            # In practice, the server socket's getsockname() would give us the actual port
            pass

        self._running = True
        return self._server.port if self._server.port > 0 else 0

    async def stop(self) -> None:
        """Stop the gRPC server."""
        if self._server:
            await self._server.stop()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


async def run_worker(node_id: str, port: int, device_str: str) -> int:
    """Run the worker and return the bound port."""
    device = torch.device(device_str)
    worker = GRPCWorker(node_id=node_id, device=device)
    actual_port = await worker.start(port=port)

    # If port was 0, we need to figure out the actual bound port
    # The server's servicer doesn't directly expose the port, but we can
    # get it from the server's sockets
    if actual_port == 0 and worker._server and worker._server._server:
        # Get the actual port from the server sockets
        try:
            for sock in worker._server._server._sockets:  # type: ignore[attr-defined]
                if hasattr(sock, "getsockname"):
                    actual_port = sock.getsockname()[1]
                    break
        except Exception:
            pass

    # Print the port for the parent process to parse
    print(f"GRPC_WORKER_PORT:{actual_port}", flush=True)

    # Wait for SIGTERM
    import asyncio

    stop_event = asyncio.Event()

    def _signal_handler(sig, frame):
        logger.info("Received signal %s, shutting down...", sig)
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    await stop_event.wait()

    await worker.stop()
    logger.info("Worker %s stopped", node_id)
    return actual_port


def main():
    parser = argparse.ArgumentParser(
        description="Bioplausible gRPC Worker for Tile Mesh P2P",
        prog="biopl-grpc-worker",
    )
    parser.add_argument(
        "--node-id", type=str, default="worker_0", help="Node identifier"
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Port to bind to (0 = OS assigned)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, cuda, cuda:0, etc.)"
    )

    args = parser.parse_args()

    import asyncio

    asyncio.run(run_worker(args.node_id, args.port, args.device))


if __name__ == "__main__":
    main()
