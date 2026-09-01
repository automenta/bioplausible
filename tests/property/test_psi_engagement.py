"""R7 probe #3 (imp-43): ψ-engagement locks — the hard gate for M-axis claims.

The benchmark harnesses (L1/L3) construct ψ but their adaptation/recovery
timing is θ-optimizer-driven, so plasticity cannot express itself in the
measured metrics — bit-identical M-arm means. Before any M-axis claim, the
engagement chain must hold end-to-end at the ontology-pipeline level
(``run_train_step`` with plasticity+ψ wiring, the path imp-22 opened):

  1. ψ moves: ``plasticity.step`` changes ψ on task input,
  2. ψ reaches behavior: ``modulate`` alters settled activations given the
     stepped ψ vs the initial ψ,
  3. metrics respond: identical θ/inputs, frozen-ψ control vs stepped arm
     produce different train-step metrics.

Suite-level verdicts and the Z3 gate checklist live in
``computronium/experiments/joint/_claims.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from computronium.core.campaign.evaluation import build_coordinate_system
from computronium.state import CompositeState

COORDINATES = {
    "routing": "digital/recurrent/instantaneous/routing/gradient/euclidean",
    "fast_weights": "digital/recurrent/instantaneous/fast_weights/gradient/euclidean",
}
_IDS = list(COORDINATES)


class FrozenPsi:
    """DI control: same modulate hook, but ``step`` never moves ψ."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def initial_psi(self, context: object, batch_size: int = 1) -> dict[str, Tensor]:
        return self._inner.initial_psi(context, batch_size)  # type: ignore[attr-defined]

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,  # ruff: ignore[unused-method-argument] - frozen control ignores the joint state
        context: object,  # ruff: ignore[unused-method-argument] - frozen control ignores the joint state
    ) -> dict[str, Tensor]:
        return psi

    def modulate(
        self, activations: list[Tensor] | Tensor, psi: dict[str, Tensor]
    ) -> list[Tensor] | Tensor:
        return self._inner.modulate(activations, psi)  # type: ignore[attr-defined]


def _key_tensor(psi: dict[str, Tensor]) -> Tensor:
    return psi[next(iter(psi))]


@pytest.fixture(params=COORDINATES.values(), ids=_IDS)
def coordinate(request: pytest.FixtureRequest) -> str:
    return request.param


def test_psi_moves_under_task_input(coordinate: str) -> None:
    system = build_coordinate_system(coordinate)
    plasticity = system.plasticity
    psi0 = plasticity.initial_psi(system.context, batch_size=4)
    z = CompositeState(
        activity={"x": torch.randn(4, 8), "y": torch.randn(4, 8)},
        plastic=psi0,
        substrate={},
    )
    psi1 = plasticity.step(psi0, z, system.context)
    assert not torch.equal(_key_tensor(psi0), _key_tensor(psi1)), (
        "ψ is constant under task input — M-axis inert"
    )


def test_modulate_reaches_activations(coordinate: str) -> None:
    system = build_coordinate_system(coordinate)
    plasticity = system.plasticity
    psi0 = plasticity.initial_psi(system.context, batch_size=4)
    z = CompositeState(
        activity={"x": torch.randn(4, 8), "y": torch.randn(4, 8)},
        plastic=psi0,
        substrate={},
    )
    psi1 = plasticity.step(psi0, z, system.context)
    acts = [torch.randn(4, 8), torch.randn(4, 8)]
    m0 = plasticity.modulate(acts, psi0)
    m1 = plasticity.modulate(acts, psi1)
    diff = sum((a - b).abs().max().item() for a, b in zip(m0, m1, strict=True))
    assert diff > 0.0, "modulate(ψ) is invariant to ψ change — ψ not causally engaged"


def test_metrics_respond_to_frozen_psi_control(coordinate: str) -> None:
    torch.manual_seed(7)
    x, y = torch.randn(16, 8), torch.randint(0, 8, (16,))

    torch.manual_seed(11)
    stepped_system = build_coordinate_system(coordinate)
    stepped = stepped_system.train_step(x, y)

    torch.manual_seed(11)
    frozen_system = build_coordinate_system(coordinate)
    frozen_system = dataclasses.replace(
        frozen_system,  # type: ignore[arg-type]
        plasticity=FrozenPsi(frozen_system.plasticity),
    )
    frozen = frozen_system.train_step(x, y)

    assert stepped["free_loss"] != pytest.approx(frozen["free_loss"]), (
        "metrics identical under frozen vs stepped ψ — suite cannot measure "
        "M-axis effects"
    )
