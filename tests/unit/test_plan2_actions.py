"""Unit tests for plan §10 surfaced changes: rule spaces, pruner support, data-loaders."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pytest
import torch
from torch.utils.data import TensorDataset

from bioplausible.cli.frontier import load_report_points, run_frontier_report
from bioplausible.data.vision import create_data_loaders, get_vision_dataset
from bioplausible.experiment.producer import OptunaBayesProducer
from bioplausible.hyperopt.comparator import compare_frontiers
from bioplausible.hyperopt.frontier import (
    RulePoint,
    cost_of_plausibility,
    pareto_frontier,
)
from bioplausible.hyperopt.ideal_backprop import (
    IdealBackpropDecision,
    IdealBackpropFinder,
)
from bioplausible.hyperopt.rule_frontier import (
    RuleFrontierDecision,
    RuleFrontierFinder,
    find_rule_frontier,
    sample_config_for_rule,
)
from bioplausible.hyperopt.scaling_law import (
    fit_accuracy_scaling,
    predict_flops_for_accuracy,
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


def test_rule_spaces_eqprop_expose_adaptive_early_stop():
    """§7/§4C: eqprop search space exposes the convergence early-stop knobs."""
    eq = get_rule_space("eqprop")
    assert eq["convergence_threshold"][2] == "log"
    assert eq["convergence_start"][2] == "int"


def test_equilibrium_early_stop_config_wires_to_model():
    """§7: convergence_threshold/start kwargs reach StandardEqProp."""
    from bioplausible.zoo.models.eqprop.standard_eqprop import StandardEqProp

    m = StandardEqProp(
        config=None,
        input_dim=10,
        output_dim=5,
        hidden_dim=8,
        num_layers=1,
        use_spectral_norm=False,
        convergence_threshold=1e-2,
        convergence_start=2,
    )
    assert m.convergence_threshold == 0.01
    assert m.convergence_start == 2
    # default when not passed
    m2 = StandardEqProp(
        config=None,
        input_dim=10,
        output_dim=5,
        hidden_dim=8,
        num_layers=1,
        use_spectral_norm=False,
    )
    assert m2.convergence_threshold == pytest.approx(1e-3)
    assert m2.convergence_start == 5


def test_forward_only_rule_spaces_exist():
    """§4D: forward-only and FA families have continuous spaces for search."""
    for rule in ("pepita", "forward_forward", "feedback_alignment"):
        space = get_rule_space(rule)
        assert space["lr"][2] == "log"
        assert space["hidden_dim"][2] == "log"
        assert space["num_layers"][2] == "int"


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


def test_cached_vision_forces_zero_workers():
    """§3: in-memory cached TensorDatasets must not spawn workers (IPC is pure overhead)."""
    train_loader, _ = create_data_loaders("mnist", batch_size=64, num_workers=4)
    assert train_loader.num_workers == 0


def test_disk_dataset_keeps_workers():
    """Generated (non-cached) datasets keep the operator's worker count."""
    train_loader, _ = create_data_loaders("xor", batch_size=16, num_workers=4)
    assert train_loader.num_workers == 4


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


class _FakeDriver:
    def __init__(self, acc_by_dim: dict[int, float] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.acc_by_dim = acc_by_dim or {}

    def train(
        self,
        *,
        model: str,
        task: str,
        config: dict[str, object],
        seed: int,
        epochs: int,
        device: str,
    ) -> dict[str, object]:
        del model, task, seed, epochs, device
        self.calls.append(dict(config))
        hidden = int(config.get("hidden_dim", 64))
        acc = self.acc_by_dim.get(hidden, 0.9)
        return {
            "final_acc": acc,
            "forward_flops": hidden * 100,
            "backward_flops": hidden * 50,
            "peak_memory_mb": hidden / 10.0,
            "wall_time_s": hidden / 100.0,
        }


def test_ideal_backprop_finder_searches_and_caches(tmp_path):
    driver = _FakeDriver({64: 0.95, 128: 0.98})
    finder = IdealBackpropFinder(
        driver,
        task="mnist",
        budget_probes=20,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    decision = finder.find()
    assert isinstance(decision, IdealBackpropDecision)
    assert decision.task == "mnist"
    assert len(decision.points) == 20
    assert decision.frontier  # non-empty
    # every probe trains through the driver
    assert len(driver.calls) == 20


def test_ideal_backprop_finder_uses_cache(tmp_path):
    driver = _FakeDriver()
    finder = IdealBackpropFinder(
        driver,
        task="mnist",
        budget_probes=10,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    first = finder.find()
    calls_after_first = len(driver.calls)
    second = finder.find()
    assert second.frontier == first.frontier
    # no additional training on cache hit
    assert len(driver.calls) == calls_after_first


def test_ideal_backprop_finder_force_reruns(tmp_path):
    driver = _FakeDriver()
    finder = IdealBackpropFinder(
        driver,
        task="mnist",
        budget_probes=8,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    finder.find()
    finder.find(force=True)
    assert len(driver.calls) == 16


def test_compare_frontiers_detects_dominating_point():
    # bio: higher accuracy AND cheaper on all 3 resources -> strictly dominates
    bio = [_pt("bio", 0.95, 100, 50, 10)]
    bp = [_pt("backprop", 0.9, 150, 60, 12)]
    cmp = compare_frontiers(bio, bp, rule="bio", backprop="backprop_mlp", task="xor")
    assert cmp.n_dominating_points == 1
    assert len(cmp.matches) == 1
    assert cmp.matches[0].accuracy_delta > 0
    assert cmp.matches[0].dominates()


def test_compare_frontiers_cheaper_at_same_accuracy():
    # bio: same accuracy, cheaper on all 3 resources -> cost < 1 (deployment-viable)
    bio = [_pt("bio", 0.9, 100, 50, 10)]
    bp = [_pt("backprop", 0.9, 150, 60, 12)]
    cmp = compare_frontiers(bio, bp, rule="bio", backprop="backprop_mlp", task="xor")
    assert cmp.cost_of_plausibility < 1.0
    assert cmp.n_dominating_points == 0  # equal accuracy is not a strict dominance


def test_compare_frontiers_worse_rule_reports_cost_above_one():
    bio = [_pt("bio", 0.9, 500, 250, 50)]
    bp = [_pt("backprop", 0.9, 50, 25, 5)]
    cmp = compare_frontiers(bio, bp, rule="bio", backprop="backprop_mlp", task="xor")
    assert cmp.n_dominating_points == 0
    assert cmp.cost_of_plausibility == pytest.approx(10.0)
    assert cmp.matches[0].flops_ratio == pytest.approx(10.0)


def test_compare_frontiers_empty_returns_inf():
    cmp = compare_frontiers([], [_pt("backprop", 0.9, 1, 1, 1)], rule="bio", task="xor")
    assert cmp.cost_of_plausibility == float("inf")
    assert cmp.matches == ()


def test_fit_accuracy_scaling_recovers_linear_log_law():
    rng = np.random.default_rng(0)
    flops = np.logspace(3, 7, 40)
    true_a, true_b = 0.05, 0.3
    acc = true_a * np.log(flops) + true_b + rng.normal(0, 0.001, flops.size)
    pts = [
        _pt("backprop_mlp", float(a), float(f), 10.0, 1.0) for f, a in zip(flops, acc)
    ]
    law = fit_accuracy_scaling(pts, rule="backprop_mlp", task="mnist")
    assert law is not None
    assert law.slope == pytest.approx(true_a, rel=0.1)
    assert law.intercept == pytest.approx(true_b, rel=0.1)
    assert law.r2 > 0.9
    assert law.n == 40


def test_fit_accuracy_scaling_too_few_points_returns_none():
    pts = [_pt("bio", 0.9, 100, 1, 1), _pt("bio", 0.91, 200, 1, 1)]
    assert fit_accuracy_scaling(pts, rule="bio", task="t") is None


def test_predict_flops_for_accuracy_matches_known_law():
    # True law: acc = 0.05 * log(F+1) + 0.3. For target 0.8 -> log(F+1) = 10
    # -> F = e^10 - 1 (the inverse must subtract the +1 offset).
    law = _mk_law(slope=0.05, slope_se=0.001, intercept=0.3, intercept_se=0.005)
    mean, lo, hi = predict_flops_for_accuracy(law, 0.8)
    assert mean == pytest.approx(np.exp(10.0) - 1.0, rel=0.01)
    assert lo < mean < hi


def test_predict_flops_nonpositive_slope_returns_nan():
    law = _mk_law(slope=0.0, slope_se=0.0, intercept=0.5, intercept_se=0.0)
    assert (
        predict_flops_for_accuracy(law, 0.8)[0]
        != predict_flops_for_accuracy(law, 0.8)[0]
    )


def _mk_law(slope, slope_se, intercept, intercept_se):
    from bioplausible.hyperopt.scaling_law import AccuracyScalingLaw

    return AccuracyScalingLaw(
        rule="backprop_mlp",
        task="mnist",
        slope=slope,
        slope_se=slope_se,
        intercept=intercept,
        intercept_se=intercept_se,
        r2=0.99,
        n=40,
    )


def test_sample_config_eqprop_has_equilibrium_params():
    trial = optuna.trial.FixedTrial({
        "lr": 0.01,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "beta": 0.5,
        "max_steps": 20,
        "damping": 0.3,
        "tol": 1e-4,
        "convergence_threshold": 1e-2,
        "convergence_start": 2,
    })
    cfg = sample_config_for_rule(trial, "eqprop")
    assert cfg["beta"] == 0.5
    assert cfg["max_steps"] == 20
    assert cfg["damping"] == 0.3
    assert cfg["tol"] == 1e-4
    assert cfg["convergence_threshold"] == 1e-2
    assert cfg["convergence_start"] == 2


def test_rule_frontier_finder_searches_bio_rule(tmp_path):
    driver = _FakeDriver({32: 0.92, 64: 0.95})
    finder = RuleFrontierFinder(
        driver,
        rule="eqprop",
        model="eqprop_mlp",
        task="mnist",
        budget_probes=15,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    decision = finder.find()
    assert isinstance(decision, RuleFrontierDecision)
    assert decision.rule == "eqprop"
    assert len(decision.points) == 15
    assert decision.frontier
    assert len(driver.calls) == 15


def test_rule_frontier_finder_unknown_rule_raises(tmp_path):
    finder = RuleFrontierFinder(
        _FakeDriver(),
        rule="not_a_rule",
        task="mnist",
        budget_probes=5,
        cache_dir=str(tmp_path),
    )
    with pytest.raises(ValueError):
        finder.find()


def test_find_rule_frontier_convenience(tmp_path):
    driver = _FakeDriver()
    decision = find_rule_frontier(
        driver,
        rule="neural_cube",
        model="neural_cube",
        task="mnist",
        budget_probes=7,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
    )
    assert decision.rule == "neural_cube"
    assert len(decision.points) == 7
