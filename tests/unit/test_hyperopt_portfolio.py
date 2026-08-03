"""Tests for the Phase 1 portfolio decision logic (VALIDATE.md)."""

import pytest

from bioplausible.hyperopt.portfolio import (
    PortfolioRow,
    decide_status,
    has_regime_advantage,
    regime_advantage_label,
)

EQUILIBRIUM = {"equilibrium"}
GLOBAL = {"global"}


@pytest.mark.parametrize(
    ("family", "locality", "expected"),
    [
        # No structural regime for pure-global families.
        ("backprop", GLOBAL, False),
        ("predictive_coding", GLOBAL, False),
        # O(1)/low-memory locality grants a regime.
        ("eqprop", EQUILIBRIUM, True),
        ("forward_only", {"forward-only"}, True),
        ("hebbian", {"local"}, True),
        # Continual-learning family grants a regime even at global locality.
        ("fa", GLOBAL, True),
    ],
)
def test_has_regime_advantage(family, locality, expected):
    assert has_regime_advantage(family, locality) is expected


@pytest.mark.parametrize(
    ("best_acc", "baseline_acc", "family", "locality", "expected"),
    [
        # Within 5pp -> Scale.
        (0.96, 0.98, "backprop", GLOBAL, "Scale"),
        (0.94, 0.97, "eqprop", EQUILIBRIUM, "Scale"),
        # 6-15pp below -> Hold (below the 15pp elimination threshold).
        (0.90, 0.97, "eqprop", EQUILIBRIUM, "Hold"),
        (0.88, 0.97, "predictive_coding", GLOBAL, "Hold"),
        # >15pp below AND no regime -> Eliminated.
        (0.80, 0.97, "predictive_coding", GLOBAL, "Eliminated"),
        # >15pp below BUT a regime exists -> Hold (revisit).
        (0.80, 0.97, "eqprop", EQUILIBRIUM, "Hold"),
        (0.80, 0.97, "hebbian", {"local"}, "Hold"),
        # Exactly at threshold (15pp) is not "more than 15pp" -> Hold.
        (0.82, 0.97, "predictive_coding", GLOBAL, "Hold"),
    ],
)
def test_decide_status(best_acc, baseline_acc, family, locality, expected):
    assert decide_status(best_acc, baseline_acc, family, locality) == expected


def test_parity_gap_pp_property():
    row = PortfolioRow(
        family="eqprop", best_acc=0.90, baseline_acc=0.97, locality=EQUILIBRIUM
    )
    assert row.parity_gap_pp == pytest.approx(7.0)


def test_regime_advantage_label_union():
    label = regime_advantage_label("eqprop", {"equilibrium", "global"})
    assert "O(1)/low-memory" in label
    assert "continual" in label


def test_regime_advantage_label_none():
    assert regime_advantage_label("predictive_coding", {"global"}) == "—"
