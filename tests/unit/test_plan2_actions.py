"""Unit tests for plan §10 surfaced changes: rule spaces, pruner support, data-loaders."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import optuna
import pytest
import torch
from torch.utils.data import TensorDataset

from bioplausible.cli.frontier import load_report_points, run_frontier_report
from bioplausible.data.vision import create_data_loaders, get_vision_dataset
from bioplausible.experiment.producer import OptunaBayesProducer
from bioplausible.hyperopt.frontier import (
    RulePoint,
    cost_of_plausibility,
    pareto_frontier,
)
from bioplausible.hyperopt.search_space import get_rule_space

if TYPE_CHECKING:
    from pathlib import Path


def test_rule_spaces_are_continuous():
    """§10: rule spaces use continuous ranges, not coarse discrete grids."""
    bp = get_rule_space("backprop")
    assert bp["hidden_dim"][2] == "log"
    assert bp["lr"][2] == "log"
    assert bp["weight_decay"][2] == "log"
    assert bp["num_layers"][2] == "int"


def test_rule_spaces_have_equilibrium_params():
    """§10: equilibrium rules expose rule-specific damping/iteration/tol knobs."""
    eq = get_rule_space("eqprop")
    for key in ("beta", "max_steps", "damping", "tol"):
        assert key in eq
    nc = get_rule_space("neural_cube")
    assert "cube_size" in nc
    assert "max_steps" in nc


def test_rule_space_unknown_raises():
    with pytest.raises(ValueError):
        get_rule_space("not_a_rule")


def test_bayes_producer_accepts_hyperband_pruner():
    """§10: OptunaBayesProducer accepts a HyperbandPruner."""
    producer = OptunaBayesProducer(
        n_candidates=5,
        seed=0,
        pruner=optuna.pruners.HyperbandPruner(),
    )
    assert isinstance(producer.pruner, optuna.pruners.HyperbandPruner)


def test_data_loader_pin_memory_flag():
    """§10: create_data_loaders forwards pin_memory to the loaders."""
    train_loader, _ = create_data_loaders("xor", batch_size=8, pin_memory=False)
    assert train_loader.pin_memory is False
    train_loader2, _ = create_data_loaders("xor", batch_size=8, pin_memory=True)
    assert train_loader2.pin_memory is True


@pytest.mark.parametrize("task", ["xor", "spiral", "circles"])
def test_data_loader_persistent_workers_noop_without_workers(task):
    """persistent_workers must stay False when num_workers==0."""
    train_loader, _ = create_data_loaders(
        task, batch_size=8, num_workers=0, persistent_workers=True
    )
    assert train_loader.persistent_workers is False


def test_cifar10_uses_tensor_cache():
    """§10: cifar10 must return a cached TensorDataset (per-epoch transform skipped)."""
    ds = get_vision_dataset("cifar10", root="./data", train=True, download=False)
    assert isinstance(ds, TensorDataset)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert tuple(x.shape) == (3, 32, 32)
    assert int(y) in range(10)


def _pt(rule, acc, flops, mem, time, **cfg):
    return RulePoint(rule, acc, flops, mem, time, tuple(cfg.items()))


def test_pareto_frontier_keeps_non_dominated():
    # Dominated: worse accuracy AND worse on all resources than point 0.
    points = [
        _pt("a", 0.9, 100, 50, 10),
        _pt("a", 0.8, 40, 20, 3),  # lower acc but strictly cheaper: non-dominated
        _pt("a", 0.88, 110, 55, 12),  # dominated by point 0 (worse everywhere)
        _pt("a", 0.95, 200, 100, 20),  # higher acc but more expensive: non-dominated
    ]
    front = pareto_frontier(points)
    assert _pt("a", 0.88, 110, 55, 12) not in front
    assert _pt("a", 0.9, 100, 50, 10) in front
    assert _pt("a", 0.95, 200, 100, 20) in front
    assert _pt("a", 0.8, 40, 20, 3) in front


def test_pareto_frontier_all_dominated_returns_one():
    points = [
        _pt("a", 0.9, 100, 50, 10),
        _pt("a", 0.8, 200, 100, 20),
        _pt("a", 0.7, 300, 150, 30),
    ]
    front = pareto_frontier(points)
    assert front == [_pt("a", 0.9, 100, 50, 10)]


def test_cost_of_plausibility_viable():
    bio = [_pt("bio", 0.95, 60, 30, 6)]
    bp = [_pt("backprop", 0.95, 50, 25, 5)]
    ratio = cost_of_plausibility(bio, bp)
    expected = (1.2 * 1.2 * 1.2) ** (1.0 / 3.0)
    assert ratio == pytest.approx(expected)
    assert ratio <= 1.5


def test_cost_of_plausibility_expensive():
    bio = [_pt("bio", 0.95, 500, 250, 50)]
    bp = [_pt("backprop", 0.95, 50, 25, 5)]
    ratio = cost_of_plausibility(bio, bp)
    expected = (10.0 * 10.0 * 10.0) ** (1.0 / 3.0)
    assert ratio == pytest.approx(expected)


def test_cost_of_plausibility_no_reference():
    assert cost_of_plausibility([], [_pt("backprop", 0.9, 1, 1, 1)]) == float("inf")
    assert cost_of_plausibility([_pt("bio", 0.9, 1, 1, 1)], []) == float("inf")


def test_cost_of_plausibility_picks_min_over_frontier():
    bio = [
        _pt("bio", 0.9, 300, 150, 30),  # expensive
        _pt("bio", 0.95, 100, 50, 10),  # best corner
    ]
    bp = [_pt("backprop", 0.95, 100, 50, 10)]
    ratio = cost_of_plausibility(bio, bp)
    assert ratio == pytest.approx(1.0)


def _write_probe(  # ruff: ignore[too-many-positional-arguments] - probe fields mirror ProbeResult
    tmp_path: Path, model: str, acc: float, ff: int, bf: int, mem: float, time_s: float
) -> None:
    line = {
        "model": model,
        "task": "mnist",
        "config": {"hidden_dim": 64},
        "config_key": "",
        "seed": 0,
        "status": "ok",
        "final_acc": acc,
        "forward_flops": ff,
        "backward_flops": bf,
        "peak_memory_mb": mem,
        "wall_time_s": time_s,
    }
    with (tmp_path / "r.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")


def test_load_report_points_ok_only(tmp_path):
    _write_probe(tmp_path, "backprop_mlp", 0.95, 100, 200, 10.0, 5.0)
    _write_probe(tmp_path, "backprop_mlp", 0.9, 90, 180, 9.0, 4.0)
    (tmp_path / "r.jsonl").open("a").write(
        '{"model": "x", "status": "error", "final_acc": 0.0}\n'
    )
    pts = load_report_points(str(tmp_path / "r.jsonl"))
    assert len(pts) == 2
    assert all(p.accuracy in {0.95, 0.9} for p in pts)


def test_run_frontier_report_computes_costs(tmp_path):
    _write_probe(tmp_path, "backprop_mlp", 0.95, 50, 50, 10.0, 5.0)
    _write_probe(tmp_path, "eqprop_mlp", 0.95, 100, 100, 20.0, 10.0)
    report = run_frontier_report(str(tmp_path / "r.jsonl"), "backprop_mlp")
    assert report["n_probes"] == 2
    assert report["models"]["backprop_mlp"]["n_frontier"] == 1
    # cost = geo-mean(2, 2, 2) = 2
    assert report["cost_of_plausibility"]["eqprop_mlp"] == pytest.approx(2.0)


def test_run_frontier_report_empty(tmp_path):
    (tmp_path / "r.jsonl").write_text("")
    report = run_frontier_report(str(tmp_path / "r.jsonl"), "backprop_mlp")
    assert report["n_probes"] == 0
    assert report["models"] == {}
