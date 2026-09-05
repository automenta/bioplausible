"""AdamUpdate locks: parity with torch.optim.Adam, state-scoping, clip.

The U-axis coverage map (D16) measured only the SGD family; D14 showed
Adam is load-bearing for deep local learning. These locks pin the
ontology's AdamUpdate to reference semantics before any coverage claim
uses it.
"""

import pytest
import torch

from computronium import AdamUpdate, EuclideanUpdate, ParameterUpdateConfig
from computronium.ontology.geometry import FeedforwardGeometry, GeometryConfig
from computronium.ontology.utils import _learnable_weight_names


def _geometry():
    return FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=4, output_dim=2, hidden_dims=(8,))
    )


def test_adam_matches_torch_optim_adam():
    """Reference parity: same gradient sequence must produce the same
    parameter trajectory as torch.optim.Adam (bitwise to fp32 tolerance)."""
    torch.manual_seed(0)
    geom = _geometry()
    params = {k: v.detach().clone() for k, v in geom.params.items()}
    torch_params = {
        k: v.detach().clone().requires_grad_(True) for k, v in geom.params.items()
    }
    opt = torch.optim.Adam(
        list(torch_params.values()), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
    )
    adam = AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3))

    torch.manual_seed(1)
    names = _learnable_weight_names(geom.params)
    for _ in range(5):
        grads = {k: torch.randn_like(v) * 0.1 for k, v in geom.params.items()}
        params = adam.step(params, [grads[n] for n in names], geom)
        opt.zero_grad()
        for n in names:
            torch_params[n].grad = grads[n].clone()
        opt.step()

    for n in geom.params:
        assert torch.allclose(params[n], torch_params[n].detach(), atol=1e-6), (
            f"AdamUpdate diverges from torch.optim.Adam on {n!r}"
        )


def test_adam_first_step_is_signed_sqrt_normalized():
    """Bias correction at t=1 makes the first step ≈ lr · sign-like move:
    |Δθ| ≈ lr for every coordinate regardless of gradient magnitude."""
    adam = AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3))
    params = {"layer_0_weight": torch.zeros(1, 10)}
    grads = [
        torch.tensor([[1e-3, 1e-1, 1.0, 2.0, 5.0, 1e-3, 1e-1, 1.0, 2.0, 5.0]])
    ]
    out = adam.step(params, grads, _geometry())["layer_0_weight"]
    assert torch.allclose(
        out.abs(), torch.full((1, 10), 1e-3), rtol=1e-4
    ), "first Adam step must be magnitude-normalized (bias-corrected)"


def test_adam_state_reuse_fails_loud():
    """Optimizer state is system-scoped: cross-geometry reuse must raise
    (the D13 momentum-buffer lesson), never silently corrupt."""
    adam = AdamUpdate(ParameterUpdateConfig.adam())
    params_a = {"layer_0_weight": torch.zeros(1, 4)}
    adam.step(params_a, [torch.ones(1, 4)], _geometry())
    with pytest.raises(RuntimeError, match="reused across different geometries"):
        adam.step(
            {"layer_0_weight": torch.zeros(1, 8)}, [torch.ones(1, 8)], _geometry()
        )


def test_adam_is_distinct_from_euclidean():
    """The optimizer families must produce different trajectories from the
    same gradient sequence — 'euclidean' must never silently alias 'adam'
    (the dead 'SGD/Adam' docstring claim this class retires)."""
    torch.manual_seed(2)
    g = [torch.randn(1, 6)]
    p0 = torch.zeros(1, 6)
    sgd = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=0.1, momentum=0.0)
    )
    adam = AdamUpdate(ParameterUpdateConfig.adam(step_size=0.1))
    p = {"layer_0_weight": p0.clone()}
    sgd_out = sgd.step(dict(p), g, _geometry())["layer_0_weight"]
    adam_out = adam.step(dict(p), g, _geometry())["layer_0_weight"]
    assert not torch.allclose(sgd_out, adam_out), (
        "Euclidean and Adam must be distinct update families"
    )
