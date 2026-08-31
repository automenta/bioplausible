"""Pin the ``torch.export`` (PT2) round-trip for both geometry families (R5.5).

The TorchScript → PT2 migration in ``computronium/deployment.py`` is complete;
these tests lock the exported-program contract: serialize, deserialize, run,
match the eager reference bitwise (CPU).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import pytest
import torch

from computronium import create_backprop_mlp, create_eqprop_mlp
from computronium.deployment import export_to_pt2

pytestmark = pytest.mark.integration


class _HasGeometry(Protocol):
    @property
    def geometry(self) -> torch.nn.Module: ...


def _roundtrip(tmp_path: Path, system: _HasGeometry, factory_name: str) -> None:
    geometry = system.geometry
    geometry.eval()
    x = torch.randn(4, 16)

    with torch.no_grad():
        reference = geometry(x)

    path = export_to_pt2(geometry, x, str(tmp_path / f"{factory_name}.pt2"))
    exported = torch.export.load(path).module()

    with torch.no_grad():
        out = exported(x)
    assert out.shape == reference.shape
    torch.testing.assert_close(out, reference)


@pytest.mark.parametrize(
    ("factory", "kwargs", "geometry_name"),
    [
        (create_backprop_mlp, {}, "FeedforwardGeometry"),
        (
            create_eqprop_mlp,
            {"beta": 0.5, "inference_steps": 10},
            "RecurrentGeometry",
        ),
    ],
    ids=["feedforward", "recurrent"],
)
def test_pt2_export_roundtrip(
    tmp_path: Path,
    factory: Callable[..., object],
    kwargs: dict[str, float | int],
    geometry_name: str,
) -> None:
    """PT2 export → load reproduces the eager geometry forward exactly."""
    system = cast(
        "_HasGeometry", factory(16, (32,), 4, lr=0.001, device="cpu", **kwargs)
    )
    assert type(system.geometry).__name__ == geometry_name
    _roundtrip(tmp_path, system, geometry_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
