"""Update-rule regression: gradient/parameter pairing over composed params.

The non-euclidean updates used to pair ``pseudo_grads`` with
``params.items()`` by index; bias interleaving made ``[hidden] - lr*[out,in]``
broadcast-crash. All rules now pair through ``apply_pseudo_gradients`` —
one gradient per learnable weight, biases pass through untouched.
"""

from __future__ import annotations

import pytest
import torch

from computronium.ontology import (
    ElasticConsolidationUpdate,
    EuclideanUpdate,
    NaturalGradientUpdate,
    ParameterUpdateConfig,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
    apply_pseudo_gradients,
)

INPUT_DIM = 8
OUTPUT_DIM = 4


def _composed_params() -> dict[str, torch.Tensor]:
    """Weight/bias interleaving as FeedforwardGeometry.named_parameters() yields."""
    torch.manual_seed(0)
    return {
        "0.weight": torch.randn(16, INPUT_DIM),
        "0.bias": torch.randn(16),
        "2.weight": torch.randn(OUTPUT_DIM, 16),
        "2.bias": torch.randn(OUTPUT_DIM),
    }


def _weight_grads(params: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    names = [n for n in params if "weight" in n and params[n].ndim == 2]
    return [torch.randn_like(params[n]) for n in names]


@pytest.mark.parametrize(
    "update",
    [
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.01)
        ),
        SpectralConstrainedUpdate(
            ParameterUpdateConfig.spectral_constrained(step_size=0.01)
        ),
        NaturalGradientUpdate(ParameterUpdateConfig.natural_gradient(step_size=0.01)),
        ElasticConsolidationUpdate(
            ParameterUpdateConfig.elastic_consolidation(step_size=0.01)
        ),
    ],
    ids=["euclidean", "riemannian", "spectral", "natural", "ewc"],
)
def test_update_applies_to_weights_and_spares_biases(update) -> None:
    params = _composed_params()
    grads = _weight_grads(params)
    updated = update.step(params, grads, geometry=None)  # type: ignore[arg-type]
    for name in params:
        if "bias" in name:
            assert torch.equal(updated[name], params[name]), f"{name} changed"
        else:
            delta = (updated[name] - params[name]).abs().max().item()
            assert delta > 0.0, f"{name} unchanged"
            assert updated[name].shape == params[name].shape


def test_euclidean_global_norm_clip_preserves_direction() -> None:
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=1.0, grad_clip=1.0)
    )
    params = _composed_params()
    big = [torch.full((16, INPUT_DIM), 100.0), torch.full((OUTPUT_DIM, 16), 100.0)]
    updated = update.step(params, big, geometry=None)  # type: ignore[arg-type]
    for grad, name in zip(big, ("0.weight", "2.weight"), strict=True):
        step = (params[name] - updated[name]).abs().mean().item()
        assert 0.0 < step <= 1.5  # clipped to global norm ~1 per tensor share


def test_euclidean_momentum_accumulates_per_parameter() -> None:
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=0.1, momentum=0.9)
    )
    params = _composed_params()
    grads = _weight_grads(params)
    first = update.step(params, grads, geometry=None)  # type: ignore[arg-type]
    second = update.step(first, grads, geometry=None)  # type: ignore[arg-type]
    d1 = (first["0.weight"] - params["0.weight"]).norm().item()
    d2 = (second["0.weight"] - first["0.weight"]).norm().item()
    assert d2 > d1 * 1.5  # momentum amplifies the repeated direction


def test_apply_pseudo_gradients_detaches_graphs() -> None:
    w = torch.randn(4, 4, requires_grad=True)
    params = {"0.weight": w.detach(), "0.bias": torch.randn(4)}
    loss = (w @ w.T).sum()
    (grad,) = torch.autograd.grad(loss, w)
    seen: list[torch.Tensor] = []

    def record(_name: str, param: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        seen.append(g)
        return param

    apply_pseudo_gradients(params, [grad], record)
    assert len(seen) == 1  # bias never receives a gradient
    assert not seen[0].requires_grad
