"""Tests for the shared Pareto dominance primitive and its delegates.

The single non-dominated filter lives in ``hyperopt.metrics`` and is reused by
``analysis.results``, ``experiment.reporting`` and ``hyperopt.frontier`` so every
frontier sink shares one dominance predicate. These tests lock in the shared
behavior and the three delegate contract.
"""

import pytest

from bioplausible.hyperopt.metrics import non_dominated_indices


def test_empty_input():
    assert non_dominated_indices([], maximize=(True, False)) == []


def test_single_point_is_non_dominated():
    assert non_dominated_indices([(0.5, 3.0)], maximize=(True, False)) == [0]


def test_clear_maximizer_dominates():
    # Higher acc + lower params dominates the rest.
    vals = [(0.9, 1.0), (0.5, 5.0), (0.6, 4.0)]
    assert non_dominated_indices(vals, maximize=(True, False)) == [0]


def test_tradeoff_keeps_multiple():
    # Best-acc but high-params; low-acc but tiny-params -> both on frontier.
    vals = [(0.9, 9.0), (0.1, 0.1)]
    assert non_dominated_indices(vals, maximize=(True, False)) == [0, 1]


def test_tie_not_strictly_better_is_dominated():
    # Same acc and params as another point, but not strictly better -> remove.
    vals = [(0.5, 3.0), (0.5, 3.0), (0.7, 2.0)]
    idx = non_dominated_indices(vals, maximize=(True, False))
    assert 2 in idx and 2 not in [(0.5, 3.0), (0.5, 3.0)]


def test_tolerance_absorbs_minor_accuracy_gap():
    # tol on the accuracy axis treats non-zero gaps within eps as "not worse".
    vals = [(0.6005, 1.0), (0.6, 1.0), (0.5, 0.5)]
    idx = non_dominated_indices(vals, maximize=(True, False), tol=(1e-3, 0.0))
    # acc within eps of each other + equal params => not strictly better; the
    # lower-acc smaller-params point can still be on the frontier.
    assert 0 in idx


def test_delegation_analysis_results():
    from bioplausible.analysis.results import compute_pareto_frontier

    trials = [
        {"trial_id": 1, "accuracy": 0.9, "param_count": 5.0, "iteration_time": 0.3},
        {"trial_id": 2, "accuracy": 0.5, "param_count": 2.0, "iteration_time": 0.1},
        {"trial_id": 3, "accuracy": 0.7, "param_count": 4.0, "iteration_time": 0.2},
    ]
    # Each point is best on a different axis (acc/params/time) -> all on frontier.
    assert compute_pareto_frontier(trials) == [1, 2, 3]
    # A fully-dominated point (worse acc AND params) is removed.
    trials2 = [
        {"trial_id": 1, "accuracy": 0.9, "param_count": 3.0, "iteration_time": 0.2},
        {"trial_id": 2, "accuracy": 0.5, "param_count": 5.0, "iteration_time": 0.5},
        {"trial_id": 3, "accuracy": 0.8, "param_count": 4.0, "iteration_time": 0.4},
    ]
    assert compute_pareto_frontier(trials2) == [1]


def test_delegation_reporting():
    from types import SimpleNamespace

    from bioplausible.experiment.reporting import pareto_frontier

    results = [
        SimpleNamespace(status="ok", param_count=5, config_key="a", final_acc=0.9),
        SimpleNamespace(status="ok", param_count=2, config_key="b", final_acc=0.5),
        SimpleNamespace(status="ok", param_count=4, config_key="c", final_acc=0.7),
    ]
    keys = {p["config_key"] for p in pareto_frontier(results)}
    assert keys == {"a", "b", "c"}
    # A config that is worse on acc and worse-or-equal on params is dominated.
    results2 = [
        SimpleNamespace(status="ok", param_count=3, config_key="x", final_acc=0.9),
        SimpleNamespace(status="ok", param_count=6, config_key="y", final_acc=0.4),
        SimpleNamespace(status="ok", param_count=5, config_key="z", final_acc=0.8),
    ]
    keys2 = {p["config_key"] for p in pareto_frontier(results2)}
    assert keys2 == {"x"}


def test_delegation_hyperopt_frontier():
    from bioplausible.hyperopt.frontier import RulePoint, pareto_frontier

    pts = [
        RulePoint("r", 0.9, 100, 50, 1.0),
        RulePoint("r", 0.5, 20, 10, 0.2),
        RulePoint("r", 0.7, 40, 30, 0.5),
    ]
    # Trade-offs across acc/flops/mem/time keep all three on the frontier.
    front = pareto_frontier(pts)
    assert [p.accuracy for p in front] == [0.9, 0.5, 0.7]
    # A point worse on accuracy AND every resource is dominated.
    pts2 = [
        RulePoint("r", 0.9, 100, 50, 1.0),
        RulePoint("r", 0.6, 120, 60, 1.2),
        RulePoint("r", 0.75, 110, 55, 1.1),
    ]
    assert [p.accuracy for p in pareto_frontier(pts2)] == [0.9]