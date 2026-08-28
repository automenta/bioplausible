"""gRPC worker function for multiprocessing spawn."""

from __future__ import annotations

import asyncio
import signal

import torch

from computronium.ontology import GeometryConfig, TileGeometry
from computronium.p2p.grpc_service import GRPCServer


def run_grpc_worker(node_id: str, port: int, ready_pipe, device_str: str) -> None:
    """Run a gRPC worker in a separate process."""

    async def _run() -> None:
        device = torch.device(device_str)
        config = GeometryConfig(
            input_dim=16,
            output_dim=10,
            num_layers=3,
            topology_type="tile_mesh",
        )
        geometry = TileGeometry(
            config,
            neurons_per_tile=8,
            tiles_per_layer=2,
        ).to(device)

        server = GRPCServer(geometry, node_id, port=port, device=device)
        await server.start()

        ready_pipe.send(server.port)

        stop_event = asyncio.Event()

        def _sigterm_handler(sig, frame):
            stop_event.set()

        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGINT, _sigterm_handler)

        await stop_event.wait()
        await server.stop()

    asyncio.run(_run())
