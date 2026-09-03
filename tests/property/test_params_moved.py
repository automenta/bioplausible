"""Params-moved learning locks (imp-26): every README-table factory either
demonstrates parameter movement through ``train_step`` or is pinned as a
documented non-learner via strict xfail (a fix flips xpass and forces
promotion to the movers list).

Ground truth measured 2026-09-03 at HEAD (tiny dims, 2 train_steps, lr=0.01):
movers = backprop, eqprop, fa, ff, tile; non-movers = pepita, tp, pc,
hebbian, snn. Non-mover reasons live in ``_NON_MOVERS`` and the TODO11
register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from computronium.core.presets import (
    create_backprop_mlp,
    create_eqprop_mlp,
    create_fa_mlp,
    create_ff_mlp,
    create_hebbian_mlp,
    create_pc_mlp,
    create_pepita_mlp,
    create_snn_mlp,
    create_tile_mlp,
    create_tp_mlp,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.ontology.system import System

_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, _BATCH = 16, (12,), 4, 8

_MOVERS = ("backprop", "eqprop", "fa", "ff", "tile")
_NON_MOVERS = {
    # Rate-coded surrogate carries no error signal; real STDP is unwired
    # (TODO11 R11.1.9, standing refutation candidate).
    "snn": "R11.1.9: pipeline-facing rate-coded surrogate has no error signal",
    # Same Hebbian-family plateau as snn (TODO11 R11.1.9).
    "hebbian": "R11.1.9: Hebbian-only plateau — no error signal in pipeline",
    # Non-learning LocalGoodness/TargetInversion pipeline paths surfaced by
    # this lock (finding recorded in TODO11 imp-26 entry, 2026-09-03).
    "pepita": "LocalGoodness pipeline path yields no param movement (finding)",
    "tp": "TargetInversion pipeline path yields no param movement (finding)",
    "pc": "PredictiveSettling+LocalGoodness path yields no movement (finding)",
}

_BUILDERS: dict[str, Callable[[], System]] = {
    "backprop": lambda: create_backprop_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01),
    "eqprop": lambda: create_eqprop_mlp(
        _INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01, inference_steps=3
    ),
    "fa": lambda: create_fa_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01),
    "ff": lambda: create_ff_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, layer_lr=0.03),
    "pepita": lambda: create_pepita_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01),
    "tp": lambda: create_tp_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01),
    "pc": lambda: create_pc_mlp(
        _INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01, settle_steps=5
    ),
    "hebbian": lambda: create_hebbian_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01),
    "snn": lambda: create_snn_mlp(_INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01),
    "tile": lambda: create_tile_mlp(
        _INPUT_DIM, _HIDDEN, _OUTPUT_DIM, lr=0.01, neurons_per_tile=4
    ),
}


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(8, _INPUT_DIM), torch.randint(0, _OUTPUT_DIM, (8,))


def _params_moved(system: System) -> tuple[int, int, float]:
    before = {k: v.detach().clone() for k, v in system.geometry.params.items()}
    x, y = _batch()
    for _ in range(2):
        metrics = system.train_step(x, y)
    assert metrics is not None
    moved = [k for k in before if not torch.equal(before[k], system.geometry.params[k])]
    max_delta = max(
        ((system.geometry.params[k] - before[k]).abs().max().item() for k in before),
        default=0.0,
    )
    return len(moved), len(before), max_delta


class TestParamsMovedLocks:
    @pytest.mark.parametrize("name", _MOVERS)
    def test_params_move(self, name: str) -> None:
        """The factory's train_step demonstrably updates geometry params."""
        torch.manual_seed(0)
        moved, total, max_delta = _params_moved(_BUILDERS[name]())
        assert moved > 0, f"{name}: 0/{total} params moved (max_delta={max_delta:.2e})"
        assert max_delta > 0.0

    @pytest.mark.parametrize("name", sorted(_NON_MOVERS))
    @pytest.mark.xfail(reason="pinned non-learner", strict=True)
    def test_pinned_non_learners(self, name: str) -> None:
        """Documented non-learners: xfail(strict) so a fix self-flags xpass."""
        torch.manual_seed(0)
        moved, total, _ = _params_moved(_BUILDERS[name]())
        assert moved > 0, f"{name}: still 0/{total} params moved ({_NON_MOVERS[name]})"
