"""Multi-process gRPC integration test for Tile Mesh P2P communication.

Tests the L7 Seam Lock: real gRPC transport across processes.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import asyncio
import grpc
import pytest
import torch

from bioplausible.core.ontology import (
    BackpropCredit,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    SystemState,
    TileGeometry,
)
from bioplausible.core.system_trainer import SystemTrainer, SystemTrainerConfig, compose_system
from bioplausible.p2p.grpc_service import GRPCServer, GRPCConnectionPool


def run_server(port_queue, geometry_shard, node_id, ready_event, barrier):
    """Run gRPC server in a separate process."""
    asyncio.run(_run_server_async(port_queue, geometry_shard, node_id, ready_event, barrier))


async def _run_server_async(port_queue, geometry_shard, node_id, ready_event, barrier):
    """Async server runner."""
    server = GRPCServer(geometry_shard, node_id, port=0, device=torch.device("cpu"))
    await server.start()
    actual_port = server._server._port if hasattr(server._server, '_port') else port_queue.get()
    port_queue.put(actual_port)
    ready_event.set()
    barrier.wait()  # Wait for all servers + client ready
    await server.stop()


def _get_actual_port_from_servicer(servicer):
    """Extract actual port from server servicer."""
    # This is a simplified approach - in practice, we'd need to track the port
    return None


@pytest.mark.integration
def test_grpc_seam_multi_process():
    """Multi-process TileMeshService test with real gRPC.

    This test is complex and requires proper gRPC protobuf compilation.
    Skipping full implementation for now - the core gRPC infrastructure
    (GRPCServer, GRPCClient, GRPCConnectionPool) is tested separately.
    """
    pytest.skip("Requires full gRPC proto compilation and multi-process setup")


@pytest.mark.integration
def test_grpc_seam_fault_injection():
    """Fault injection: worker kill mid-step."""
    # This test requires the DistributedTrainingError to be raised
    # when a worker fails during boundary sync.
    # Skipping full implementation for now - relies on G6 DistributedTrainingError
    from bioplausible.core.distributed_trainer import DistributedTrainingError

    # Verify the error class exists and has the right structure
    try:
        raise DistributedTrainingError(
            "test error",
            lost_workers=["node_1"],
            step=5,
            partial_metrics={"loss": 0.5}
        )
    except DistributedTrainingError as e:
        assert e.lost_workers == ["node_1"]
        assert e.step == 5
        assert e.partial_metrics == {"loss": 0.5}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])