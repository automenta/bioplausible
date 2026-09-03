"""θ-invariance audit harness (RESEARCH3 prerequisite PR-2).

Snapshot → verify-frozen → run → re-snapshot → exact-diff, as a reusable
context manager producing a per-run report. Consumed by Z3 evaluation,
algorithm-migration shakedowns, and continual-learning runs wherever an
exact-zero Δθ claim must be auditable end-to-end.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from types import TracebackType

ParamSelector = Callable[[str, nn.Parameter], bool]


@dataclass(frozen=True, slots=True)
class ThetaAuditReport:
    """Result of one θ-invariance audit window.

    Attributes:
        frozen_on_entry: All selected parameters had requires_grad=False on entry.
        max_abs_change: Maximum absolute elementwise drift across selected params.
        changed_keys: Names of parameters with any nonzero drift.
    """

    frozen_on_entry: bool
    max_abs_change: float
    changed_keys: tuple[str, ...]

    @property
    def invariant(self) -> bool:
        """True iff every selected parameter is bit-exactly unchanged."""
        return self.max_abs_change == 0.0

    def is_within(self, tolerance: float) -> bool:
        """True iff drift stays under ``tolerance``."""
        return self.max_abs_change <= tolerance


def require_frozen(params: dict[str, nn.Parameter]) -> None:
    """Raise unless every named parameter is frozen."""
    trainable = sorted(name for name, p in params.items() if p.requires_grad)
    if trainable:
        raise RuntimeError(f"frozen-θ violation: trainable parameters {trainable}")


class ThetaInvarianceAudit:
    """Context manager auditing that a parameter subset never moves.

    Args:
        module: Model whose named parameters are audited.
        selector: Predicate over ``(name, param)`` choosing the frozen subset;
            defaults to all parameters.
        expect_frozen: If True, entry raises unless the selection is frozen.

    Example:
        with ThetaInvarianceAudit(model, selector=is_theta_param) as audit:
            adapt(model)
        assert audit.report is not None and audit.report.invariant
    """

    def __init__(
        self,
        module: nn.Module,
        selector: ParamSelector | None = None,
        *,
        expect_frozen: bool = True,
    ):
        self._module = module
        self._selector: ParamSelector = selector or (lambda _name, _p: True)
        self._expect_frozen = expect_frozen
        self.report: ThetaAuditReport | None = None

    def _selected(self) -> dict[str, nn.Parameter]:
        return {
            name: p
            for name, p in self._module.named_parameters()
            if self._selector(name, p)
        }

    def __enter__(self) -> ThetaInvarianceAudit:  # ruff: ignore[non-self-return-type]
        selected = self._selected()
        # Frozen check reads live parameters (clones always report False)
        frozen = all(not p.requires_grad for p in selected.values())
        if self._expect_frozen and not frozen:
            raise RuntimeError(
                "ThetaInvarianceAudit entered with trainable selected parameters"
            )
        self._before = {name: p.detach().clone() for name, p in selected.items()}
        self._frozen_on_entry = frozen
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            return
        after = self._selected()
        changes = {
            name: float((after[name].detach() - before).abs().max().item())
            for name, before in self._before.items()
            if name in after
        }
        self.report = ThetaAuditReport(
            frozen_on_entry=self._frozen_on_entry,
            max_abs_change=max(changes.values(), default=0.0),
            changed_keys=tuple(sorted(k for k, c in changes.items() if c != 0.0)),
        )
