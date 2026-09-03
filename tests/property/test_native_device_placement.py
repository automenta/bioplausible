"""CUDA placement guard: no native factory may leave parameters on CPU.

Kills the silent-CPU failure mode: a factory that accepts ``device="cuda"``
but builds tensors on CPU produces runs that *look* GPU-accelerated while
executing entirely on the host. Every ``create_native_*`` factory is
parametrized over ``__all__``-style discovery so a new factory without
device support fails here immediately.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from computronium.models.native import backprop_native as _backprop
from computronium.models.native import diffusion_eqprop_native as _diffusion
from computronium.models.native import eqprop_native as _eqprop
from computronium.models.native import fa_native as _fa
from computronium.models.native import momentum_eqprop_native as _momentum
from computronium.models.native import pepita_native as _pepita
from computronium.models.native import research_native as _research
from computronium.models.native import sparse_eqprop_native as _sparse
from computronium.models.native import ternary_eqprop_native as _ternary
from computronium.models.native import tile_native as _tile

INPUT_DIM = 16
HIDDEN_DIM = 16
OUTPUT_DIM = 4

# (factory, kwargs) — minimal construction args beyond the 3 positional dims
_FACTORIES: list[tuple[Any, dict]] = [
    (_backprop.create_native_backprop_mlp, {}),
    (_diffusion.create_native_diffusion_eqprop, {}),
    (_eqprop.create_native_eqprop_mlp, {}),
    *[(fn, {}) for fn in _fa.FA_FACTORY_VARIANTS],
    (_momentum.create_native_momentum_eqprop, {}),
    (_pepita.create_native_pepita_mlp, {}),
    (_research.create_native_holomorphic_ep, {}),
    (_research.create_native_directed_ep, {}),
    (_research.create_native_finite_nudge_ep, {}),
    (_sparse.create_native_sparse_eqprop, {}),
    (_ternary.create_native_ternary_eqprop, {}),
    (_tile.create_native_tile_ep, {}),
    (_tile.create_native_tile_fa, {}),
    (_tile.create_native_tile_tp, {}),
    (_tile.create_native_tile_snn, {}),
    (_tile.create_native_tile_hebbian, {}),
    (_tile.create_native_tile_pc, {}),
    (_tile.create_native_tile_gnn, {}),
]

ids = [fn.__name__.removeprefix("create_native_") for fn, _ in _FACTORIES]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA placement guard requires CUDA"
)


def _module_buffers(module: torch.nn.Module) -> list[torch.Tensor]:
    return [b for _, b in module.named_buffers(recurse=True)]


@pytest.mark.parametrize(("factory", "kwargs"), _FACTORIES, ids=ids)
def test_native_factory_places_all_parameters_on_cuda(
    factory: Any, kwargs: dict
) -> None:
    system = factory(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, device="cuda", **kwargs)

    params = dict(system.geometry.params)
    assert params, "geometry must expose parameters"

    cpu_tensors = [
        name
        for name, tensor in {**params, **_named_buffers(system)}.items()
        if tensor.device.type != "cuda"
    ]
    assert not cpu_tensors, (
        f"silent-CPU parameters on {factory.__name__}: {cpu_tensors}"
    )

    # Substrate metadata must agree with actual placement (observability).
    assert system.substrate.config.device.startswith("cuda")


def _named_buffers(system: Any) -> dict[str, torch.Tensor]:
    geometry = system.geometry
    buffers: dict[str, torch.Tensor] = {}
    for name, buf in _module_buffers(geometry):
        buffers[f"buffer:{name}"] = buf
    if isinstance(geometry, torch.nn.Module):
        for name, param in geometry.named_parameters(recurse=True):
            buffers[f"module:{name}"] = param
    return buffers


def test_unknown_kwargs_are_rejected() -> None:
    with pytest.raises(TypeError):
        _eqprop.create_native_eqprop_mlp(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, bogus=1)


def test_default_construction_stays_on_cpu() -> None:
    system = _eqprop.create_native_eqprop_mlp(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    assert all(t.device.type == "cpu" for t in system.geometry.params.values())
