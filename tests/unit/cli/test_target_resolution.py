"""Tests for run_experiment target resolution from config.

Covers the ``model_budgets`` allowlist filter (FIX.md §43): when a config provides
``model_budgets``, only the models it lists are targeted — everything else in the
family is skipped. This is how run_phase1_5.py restricts the exact model set
without running every registered model of a family.
"""

from __future__ import annotations

from run_experiment import _resolve_targets_from_config


def _cfg(model_budgets: dict[str, int]) -> dict[str, object]:
    return {
        "families": ["backprop"],
        "tasks": ["digits"],
        "budget": 10,
        "model_budgets": model_budgets,
    }


def test_model_budgets_acts_as_allowlist() -> None:
    """Only models named in model_budgets are targeted (others skipped)."""
    config = _cfg({"backprop.backprop_mlp": 20})
    targets = _resolve_targets_from_config(config, [], ["digits"], 10, "standard")
    names = {t.model for t in targets}
    assert names == {"backprop_mlp"}
    # budget comes from the per-model override
    assert targets[0].budget == 20


def test_no_model_budgets_runs_all_compatible() -> None:
    """Without model_budgets, all compatible family models run with cfg budget."""
    config = {"families": ["backprop"], "tasks": ["digits"], "budget": 7}
    targets = _resolve_targets_from_config(config, [], ["digits"], 7, "standard")
    assert len(targets) >= 1
    assert all(t.budget == 7 for t in targets)
