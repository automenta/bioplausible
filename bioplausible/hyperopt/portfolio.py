"""Portfolio revelation (Phase 1 of VALIDATE.md).

Implements the Phase 1.1 elimination / survival criterion and the Phase 1.3
portfolio ranking table.  Pure decision logic lives here so it is unit-testable
without a GPU; the ``biopl-hpo portfolio`` CLI subcommand wires it to the Optuna
study store and the component registry.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CONTINUAL_FAMILIES",
    "LOW_MEMORY_LOCALITIES",
    "PortfolioRow",
    "decide_status",
    "has_regime_advantage",
    "regime_advantage_label",
]

# Families whose credit-assignment regime is claimed to enable continual
# learning (VALIDATE.md Phase 1.1 bullet 3).
CONTINUAL_FAMILIES: frozenset[str] = frozenset({
    "eqprop",
    "fa",
    "hebbian",
    "forward_only",
})

# Locality levels that imply an O(1) / low-activation-memory regime (the
# structural advantage that can offset a raw-accuracy deficit).
LOW_MEMORY_LOCALITIES: frozenset[str] = frozenset({
    "equilibrium",
    "forward-only",
    "local",
    "layerwise",
})

# Phase thresholds (in percentage points of tuned accuracy vs backprop baseline).
_ELIMINATE_PP = 15.0
_SCALE_PP = 5.0


@dataclass(frozen=True, slots=True)
class PortfolioRow:
    """One algorithm family's portfolio judgement for a task scope."""

    family: str
    best_acc: float
    baseline_acc: float
    locality: set[str]
    wall_time_s: float | None = None

    @property
    def parity_gap_pp(self) -> float:
        """Gap below the best backprop baseline, in percentage points."""
        return (self.baseline_acc - self.best_acc) * 100.0

    @property
    def regime(self) -> bool:
        return has_regime_advantage(self.family, self.locality)

    @property
    def status(self) -> str:
        return decide_status(
            self.best_acc, self.baseline_acc, self.family, self.locality
        )


def has_regime_advantage(family: str, locality: set[str]) -> bool:
    """True if the family carries a structural (non-accuracy) advantage."""
    low_mem = bool(locality & LOW_MEMORY_LOCALITIES)
    continual = family in CONTINUAL_FAMILIES
    return low_mem or continual


def regime_advantage_label(family: str, locality: set[str]) -> str:
    """Human-readable description of the family's structural regime."""
    labels: list[str] = []
    if locality & {"equilibrium", "forward-only", "local", "layerwise"}:
        labels.append("O(1)/low-memory")
    if family in CONTINUAL_FAMILIES:
        labels.append("continual")
    if "forward-only" in locality:
        labels.append("forward-only")
    return ";".join(labels) if labels else "—"


def decide_status(
    best_acc: float,
    baseline_acc: float,
    family: str,
    locality: set[str],
) -> str:
    """Decide ``Scale`` / ``Hold`` / ``Eliminated`` for a family.

    Phase 1.1 criterion:
      * Eliminated  -> tuned accuracy > 15 pp below backprop baseline AND no
                       structural regime advantage.
      * Survives if ANY of:
            - parity gap < 5 pp (Scale),
            - gap < 10 pp AND O(1) memory (equilibrium/local/forward-only) or
              forward-only structure OR continual-learning family (Hold),
            - gap < 10 pp AND family in {eqprop, fa, hebbian, forward_only}.
    """
    gap_pp = (baseline_acc - best_acc) * 100.0
    regime = has_regime_advantage(family, locality)

    # Elimination requires being strictly MORE than 15pp below AND having no
    # structural regime to justify the deficit.  Epsilon absorbs the float
    # noise at the exact threshold (e.g. 15.000000000000002 vs 15.0).
    if gap_pp > _ELIMINATE_PP + 1e-9 and not regime:
        return "Eliminated"
    if gap_pp < _SCALE_PP:
        return "Scale"
    return "Hold"
