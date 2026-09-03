"""θ-invariance audit harness (PR-2, TODO11 R11.3.2).

Snapshot → run body → re-snapshot → exact-diff, as a reusable context
manager with per-seed reports. D5 (``test_demo_z3_frozen_theta.py``)
demonstrates the frozen-θ guarantee on registered machinery; this module
makes the same exact-diff instrument a library feature for Z3 /
Algorithm-Migration / continual-learning runs — any code path that
promises ‖θ_after − θ_before‖ = 0 gets audited, not trusted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["ParamsSource", "ThetaAuditReport", "theta_audit"]


class ParamsSource(Protocol):
    """Anything holding θ as a name -> tensor mapping under ``.geometry``."""

    @property
    def geometry(self) -> object:
        """Component owning the persistent parameters."""
        ...


@dataclass(frozen=True, slots=True)
class ThetaAuditReport:
    """Exact-diff verdict of one audited episode."""

    label: str
    seed: int | None
    theta_sha256_before: str
    theta_sha256_after: str
    moved: tuple[str, ...]

    @property
    def invariant(self) -> bool:
        return not self.moved and self.theta_sha256_before == self.theta_sha256_after

    def assert_invariant(self) -> None:
        """Raise with the moved-parameter list unless θ is bitwise-identical."""
        if not self.invariant:
            raise AssertionError(
                f"θ moved during audited episode {self.label!r}: {list(self.moved)}"
            )


def _tensor_sha256(params: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(params):
        t = params[name].detach()
        digest.update(name.encode())
        digest.update(str(t.device).encode())
        digest.update(str(t.dtype).encode())
        digest.update(t.cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class ThetaAuditSession:
    """Live audit handle: snapshots on entry, produces the report on demand."""

    _params: Mapping[str, Tensor]
    label: str
    seed: int | None
    _sha_before: str = field(init=False)
    _snapshots: dict[str, Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self._sha_before = _tensor_sha256(self._params)
        self._snapshots = {n: t.detach().clone() for n, t in self._params.items()}

    @property
    def report(self) -> ThetaAuditReport:
        moved = tuple(
            name
            for name, before in self._snapshots.items()
            if not _bitwise_equal(before, self._params[name])
        )
        return ThetaAuditReport(
            label=self.label,
            seed=self.seed,
            theta_sha256_before=self._sha_before,
            theta_sha256_after=_tensor_sha256(self._params),
            moved=moved,
        )


def _bitwise_equal(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape and bool((a == b).all())


@contextmanager
def theta_audit(
    system: ParamsSource | Mapping[str, Tensor],
    *,
    label: str = "",
    seed: int | None = None,
) -> Iterator[ThetaAuditSession]:
    """Audit θ invariance across the ``with`` body: snapshot, run, exact-diff.

    Accepts a composed System (audits ``system.geometry.params``) or a plain
    name -> tensor mapping. The session's ``report`` is meaningful once the
    block exits; call ``report.assert_invariant()`` to fail loud.
    """
    params: Mapping[str, Tensor] = (
        system if isinstance(system, Mapping) else system.geometry.params  # type: ignore[attr-defined]
    )
    yield ThetaAuditSession(params, label=label, seed=seed)
