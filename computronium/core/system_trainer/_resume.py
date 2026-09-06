"""Resumable training state (TODO11 R11.2.24).

``fold_in`` derives per-batch RNG seeds as a pure function of
``(base, epoch, batch[, domain])``; ``TrainerSnapshot`` carries everything
``SystemTrainer`` needs to resume. Together they make an interrupted run
bitwise identical to an uninterrupted one, independent of global RNG stream
position — the guarantee campaign checkpoint/resume (R11.3.1) builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_SEED_MAX = (1 << 63) - 1

DOMAIN_EPOCH = 1
"""Domain tag for the epoch-start draw (DataLoader shuffle permutation)."""


def _mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def fold_in(base: int, epoch: int, batch: int, *, domain: int = 0) -> int:
    """Derive a batch seed as a pure function of its coordinates.

    SplitMix64 keyed by (base, epoch, batch, domain). Reseeding the global
    torch RNG with the result at the top of each batch makes every downstream
    draw — shuffle permutation, substrate noise, projection masks — a pure
    function of the coordinates.
    """
    h = _mix64(base & _M64)
    for key in (domain, epoch, batch):
        h = _mix64((h + _GOLDEN * (key & _M64)) & _M64)
    return h & _SEED_MAX


@dataclass(frozen=True, slots=True)
class TrainerSnapshot:
    """Everything required to resume a ``SystemTrainer`` bitwise.

    Attributes:
        epoch: Number of completed epochs (resume starts here).
        global_step: Optimizer steps taken so far.
        history: Per-epoch metric dicts from the completed epochs.
        theta: Geometry parameter tensors (clones; device-agnostic).
        opt_state: Optimizer-side state as named groups of tensors per
            the update's ``get_state``/``load_state`` protocol (e.g.
            momentum buffers, Adam moments, step counters).
        credit_state: Credit-internal state as named groups per the
            credit's ``get_state``/``load_state`` protocol (e.g. B1's
            learned feedback matrices). Empty for stateless credits.
    """

    epoch: int
    global_step: int
    history: tuple[dict[str, float], ...]
    theta: dict[str, Tensor]
    opt_state: dict[str, dict[str, Tensor]]
    credit_state: dict[str, dict[str, Tensor]] = field(default_factory=dict)


__all__ = [
    "DOMAIN_EPOCH",
    "TrainerSnapshot",
    "fold_in",
]
