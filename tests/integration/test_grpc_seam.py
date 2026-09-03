"""gRPC seam tests for the P2P Tile Mesh layer.

The real-transport multi-process seam lives in
``test_grpc_seam_subprocess.py`` (subprocess workers over
``computronium.p2p._grpc_worker``); this module pins the fault-tolerance
error contract locally.
"""

from __future__ import annotations

import pytest

from computronium.core.distributed_trainer import DistributedTrainingError


@pytest.mark.integration
def test_grpc_seam_fault_injection():
    """Fault injection: DistributedTrainingError captures the seam state."""

    def _raise() -> None:
        raise DistributedTrainingError(
            "test error", lost_workers=["node_1"], step=5, partial_metrics={"loss": 0.5}
        )

    with pytest.raises(DistributedTrainingError) as exc_info:
        _raise()
    exc = exc_info.value
    assert exc.lost_workers == ["node_1"]
    assert exc.step == 5
    assert exc.partial_metrics == {"loss": 0.5}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
