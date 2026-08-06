"""Unit tests for the experiment layer (schema, report, producer, staircase)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from bioplausible.experiment.probe import ProbeResult, config_key
from bioplausible.experiment.producer import (
    HyperoptGridProducer,
    grid_cardinality,
)
from bioplausible.experiment.report import Report
from bioplausible.experiment.schema import Stage, validate_yaml
from bioplausible.experiment.staircase import (
    StageMetrics,
    StaircaseRunner,
    Verdict,
    passes_stage,
)

CAMPAIGN_YAML = """
meta: {name: smoke, created: '2026-08-05'}
compute: {device: cpu, num_workers: 0}
arms:
  mlp: {max_params: 210000, models: [backprop_mlp]}
stages:
  - name: smoke
    task: xor
    epochs: 1
    seeds: 1
    configs: {hidden_dim: [16, 32], num_layers: [2]}
    pass_rule:
      min_seed_ok: 1
      rules: [{metric: acc, op: '>=', value: 0.5}]
"""


def _result(model: str, task: str, config: dict, seed: int, acc: float) -> ProbeResult:
    return ProbeResult(
        model=model,
        task=task,
        config=config,
        config_key=config_key(config),
        seed=seed,
        status="ok",
        final_acc=acc,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_valid_campaign():
    camp = validate_yaml(CAMPAIGN_YAML)
    assert camp.meta.name == "smoke"
    assert len(camp.stages) == 1
    assert camp.geometry("xor") == (2, 2)


def test_schema_rejects_unknown_task():
    with pytest.raises(Exception):
        validate_yaml(CAMPAIGN_YAML.replace("task: xor", "task: not_a_task"))


def test_schema_rejects_low_seeds_on_evidence():
    with pytest.raises(Exception):
        validate_yaml(
            """
meta: {name: p, created: x}
compute: {device: cpu}
arms: {a: {max_params: 1, models: [backprop_mlp]}}
stages:
  - name: parity
    task: xor
    epochs: 1
    seeds: 5
    baseline: backprop_mlp
            """
        )


def test_schema_rejects_evidence_without_matched_by():
    with pytest.raises(Exception):
        validate_yaml(
            """
meta: {name: p, created: x}
compute: {device: cpu}
arms: {a: {max_params: 1, models: [backprop_mlp]}}
stages:
  - name: parity
    task: xor
    epochs: 1
    seeds: 10
    baseline: backprop_mlp
            """
        )


def test_schema_rejects_evidence_without_energy():
    with pytest.raises(Exception):
        validate_yaml(
            """
meta: {name: p, created: x}
compute: {device: cpu}
arms: {a: {max_params: 1, models: [backprop_mlp]}}
stages:
  - name: parity
    task: xor
    epochs: 1
    seeds: 10
    baseline: backprop_mlp
    matched_by: {equal_budget: max_params, reported: [wall_time_s]}
            """
        )


# ---------------------------------------------------------------------------
# config_key / producer
# ---------------------------------------------------------------------------


def test_config_key_order_independent():
    a = config_key({"hidden_dim": 16, "num_layers": 2})
    b = config_key({"num_layers": 2, "hidden_dim": 16})
    assert a == b
    assert config_key({"hidden_dim": 32}) != a


def test_grid_cardinality_exact():
    stage = Stage(
        name="s",
        task="xor",
        epochs=1,
        seeds=1,
        configs={
            "hidden_dim": [16, 32],
            "num_layers": [2, 4],
        },
    )
    assert grid_cardinality(stage.configs) == 4


def test_producer_schedules_exact_probe_count():
    campaign = validate_yaml(CAMPAIGN_YAML)
    stage = campaign.stages[0]
    producer = HyperoptGridProducer(seed=7)
    configs = producer.configs_for(stage)
    assert len(configs) == grid_cardinality(stage.configs)
    assert all(isinstance(c, dict) for c in configs)


def test_producer_configs_are_distinct():
    campaign = validate_yaml(CAMPAIGN_YAML)
    stage = campaign.stages[0]
    producer = HyperoptGridProducer(seed=7)
    configs = producer.configs_for(stage)
    # The grid enumerates a unique combination per point — no duplicates.
    assert len({frozenset(c.items()) for c in configs}) == len(configs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_resume_noop(tmp_path: Path):
    report = Report(tmp_path / "r.jsonl")
    result = _result("backprop_mlp", "xor", {"hidden_dim": 16}, 0, 0.9)
    report.append("smoke", result)
    assert report.is_finished("smoke", result)

    reloaded = Report(tmp_path / "r.jsonl")
    assert reloaded.is_finished("smoke", result)


def test_report_error_does_not_resume():
    err_path = Path("/tmp/_bio_test_err.jsonl")
    if err_path.exists():
        err_path.unlink()
    report = Report(err_path)
    bad = _result("m", "xor", {"a": 1}, 0, 0.0)
    bad = ProbeResult(
        model="m",
        task="xor",
        config={"a": 1},
        config_key=bad.config_key,
        seed=0,
        status="error",
        error="boom",
    )
    report.append("s", bad)
    assert not report.is_finished("s", bad)


# ---------------------------------------------------------------------------
# Staircase / verdicts
# ---------------------------------------------------------------------------


def test_passes_stage_all_rules_ok():
    stage = Stage(
        name="smoke",
        task="xor",
        epochs=1,
        seeds=2,
        configs={"hidden_dim": [16]},
        pass_rule={
            "min_seed_ok": 1,
            "rules": [{"metric": "acc", "op": ">=", "value": 0.5}],
        },
    )
    results = [
        _result("m", "xor", {"hidden_dim": 16}, 0, 0.8),
        _result("m", "xor", {"hidden_dim": 16}, 1, 0.9),
    ]
    passed, reason = passes_stage(stage, results)
    assert passed
    assert "satisfied" in reason


def test_passes_stage_rejects_below_rule():
    stage = Stage(
        name="smoke",
        task="xor",
        epochs=1,
        seeds=1,
        configs={"hidden_dim": [16]},
        pass_rule={
            "min_seed_ok": 1,
            "rules": [{"metric": "acc", "op": ">=", "value": 0.9}],
        },
    )
    results = [_result("m", "xor", {"hidden_dim": 16}, 0, 0.3)]
    passed, reason = passes_stage(stage, results)
    assert not passed
    assert "fails" in reason


def test_passes_stage_respects_min_seed_ok():
    stage = Stage(
        name="smoke",
        task="xor",
        epochs=1,
        seeds=3,
        configs={"hidden_dim": [16]},
        pass_rule={"min_seed_ok": 3, "rules": []},
    )
    results = [
        _result("m", "xor", {"hidden_dim": 16}, 0, 0.9),
        _result("m", "xor", {"hidden_dim": 16}, 1, 0.9),
    ]
    passed, reason = passes_stage(stage, results)
    assert not passed
    assert "ok_seeds=2" in reason


def test_passes_stage_errored_seed_never_satisfies():
    stage = Stage(
        name="smoke",
        task="xor",
        epochs=1,
        seeds=1,
        configs={"hidden_dim": [16]},
        pass_rule={
            "min_seed_ok": 1,
            "rules": [{"metric": "acc", "op": ">=", "value": 0.5}],
        },
    )
    errored = ProbeResult(
        model="m",
        task="xor",
        config={"hidden_dim": 16},
        config_key="x",
        seed=0,
        status="error",
        error="boom",
    )
    results = [errored]
    passed, _ = passes_stage(stage, results)
    assert not passed


def test_staircase_only_survivors_advance():
    # Two stages: second only runs if first passes. backprop passes stage1.
    campaign = validate_yaml(
        """
meta: {name: s, created: x}
compute: {device: cpu}
arms: {mlp: {max_params: 100000, models: [backprop_mlp, eqprop_mlp]}}
stages:
  - name: A
    task: xor
    epochs: 1
    seeds: 1
    configs: {hidden_dim: [16]}
    pass_rule: {min_seed_ok: 1, rules: [{metric: acc, op: '>=', value: 0.5}]}
  - name: B
    task: xor
    epochs: 1
    seeds: 1
    configs: {hidden_dim: [16]}
    pass_rule: {min_seed_ok: 1, rules: []}
        """
    )

    class FakeDriver:
        def train(self, *, model, task, config, seed, epochs, device):
            # backprop_mlp passes, eqprop_mlp fails stage A.
            acc = 0.9 if model == "backprop_mlp" else 0.2
            return {"final_acc": acc, "epoch_time_s": 0.1}

    stair_path = Path("/tmp/_bio_test_stair.jsonl")
    if stair_path.exists():
        stair_path.unlink()
    report = Report(stair_path)
    runner = StaircaseRunner(
        campaign,
        report,
        FakeDriver(),
        HyperoptGridProducer(seed=1),
        param_counter=lambda model, config, input_dim, output_dim: 100,
    )
    outcomes = runner.run()

    # backprop survived both stages; eqprop rejected at A and never ran B.
    backprop_B = [
        o for o in outcomes if o.model == "backprop_mlp" and o.verdict is Verdict.PASS
    ]
    eqprop_B = [
        o for o in outcomes if o.model == "eqprop_mlp" and o.verdict is Verdict.PASS
    ]
    assert len(backprop_B) == 2  # passed A and B
    assert len(eqprop_B) == 0  # rejected at A, never ran B


def test_staircase_resume_noop_does_not_retrain(tmp_path: Path):
    """Re-running a finished campaign re-trains nothing and rehydrates verdicts."""
    campaign = validate_yaml(CAMPAIGN_YAML)
    report_path = tmp_path / "resume.jsonl"

    calls: list[tuple[str, int]] = []

    class CountingDriver:
        def train(self, *, model, task, config, seed, epochs, device):
            calls.append((model, seed))
            return {"final_acc": 0.9, "epoch_time_s": 0.1}

    Report(report_path)
    runner = StaircaseRunner(
        campaign,
        Report(report_path),
        CountingDriver(),
        HyperoptGridProducer(seed=1),
        param_counter=lambda model, config, input_dim, output_dim: 1,
    )
    runner.run()
    assert len(calls) == 2  # 2 configs x 1 seed

    calls.clear()
    reloaded = Report(report_path)
    runner2 = StaircaseRunner(
        campaign,
        reloaded,
        CountingDriver(),
        HyperoptGridProducer(seed=1),
        param_counter=lambda model, config, input_dim, output_dim: 1,
    )
    outcomes = runner2.run()
    assert calls == []  # nothing re-trained
    assert all(o.verdict is Verdict.PASS for o in outcomes)


def test_staircase_resume_noop_skips_param_construction(tmp_path: Path):
    """Re-running a finished config builds no models for the budget check.

    Regression: the budget filter used to construct every (model, config) on
    re-run, so a finished campaign was not a *true* no-op. Finished configs
    must be skipped before any param count (architecture §6.7 resume-noop).
    """
    campaign = validate_yaml(CAMPAIGN_YAML)
    report_path = tmp_path / "resume_budget.jsonl"

    class CountingDriver:
        def train(self, *, model, task, config, seed, epochs, device):
            return {"final_acc": 0.9, "epoch_time_s": 0.1}

    counts: list[int] = []

    def counter(model, config, _input_dim, _output_dim):
        counts.append(1)
        return 1

    Report(report_path)
    runner = StaircaseRunner(
        campaign,
        Report(report_path),
        CountingDriver(),
        HyperoptGridProducer(seed=1),
        param_counter=counter,
    )
    runner.run()
    assert len(counts) == 2  # 2 configs counted on first run

    counts.clear()
    runner2 = StaircaseRunner(
        campaign,
        Report(report_path),
        CountingDriver(),
        HyperoptGridProducer(seed=1),
        param_counter=counter,
    )
    runner2.run()
    assert counts == []  # neither config constructed on re-run (true no-op)


def test_metrics_aggregate_median():
    metrics = StageMetrics([
        _result("m", "xor", {}, 0, 0.8),
        _result("m", "xor", {}, 1, 0.9),
        _result("m", "xor", {}, 2, 0.85),
    ])
    assert metrics.value("acc", "median") == pytest.approx(0.85)
    assert metrics.value("acc", "mean") == pytest.approx(0.85)


def _result_full(
    model: str,
    *,
    seed: int,
    acc: float = 0.0,
    loss: float = 0.0,
    flops: int = 0,
    memory: float = 0.0,
    epoch: float = 0.0,
) -> ProbeResult:
    return ProbeResult(
        model=model,
        task="xor",
        config={},
        config_key="k",
        seed=seed,
        status="ok",
        final_acc=acc,
        final_train_loss=loss,
        forward_flops=flops,
        peak_memory_mb=memory,
        epoch_time_s=epoch,
    )


def test_metrics_aggregate_loss_flops_memory():
    metrics = StageMetrics([
        _result_full("m", seed=0, loss=1.0, flops=10, memory=5.0, epoch=2.0),
        _result_full("m", seed=1, loss=3.0, flops=30, memory=7.0, epoch=4.0),
    ])
    assert metrics.value("loss", "median") == pytest.approx(2.0)
    assert metrics.value("loss", "mean") == pytest.approx(2.0)
    assert metrics.value("flops", "mean") == pytest.approx(20.0)
    assert metrics.value("memory", "mean") == pytest.approx(6.0)
    assert metrics.value("epoch_time_s", "mean") == pytest.approx(3.0)
    # Unknown metric aggregates to NaN (never satisfies a rule).
    assert math.isnan(metrics.value("nope", "mean"))


def test_passes_stage_non_acc_rule():
    stage = Stage(
        name="perf",
        task="xor",
        epochs=1,
        seeds=1,
        configs={"hidden_dim": [16]},
        pass_rule={
            "min_seed_ok": 1,
            "rules": [{"metric": "flops", "op": "<=", "value": 50}],
        },
    )
    below = [
        ProbeResult(
            model="m",
            task="xor",
            config={"hidden_dim": 16},
            config_key="a",
            seed=0,
            status="ok",
            forward_flops=20,
        )
    ]
    over = [
        ProbeResult(
            model="m",
            task="xor",
            config={"hidden_dim": 16},
            config_key="a",
            seed=0,
            status="ok",
            forward_flops=500,
        )
    ]
    assert passes_stage(stage, below)[0] is True
    assert passes_stage(stage, over)[0] is False


def test_passes_stage_nonfinite_never_satisfies():
    stage = Stage(
        name="s",
        task="xor",
        epochs=1,
        seeds=1,
        configs={"hidden_dim": [16]},
        pass_rule={
            "min_seed_ok": 1,
            "rules": [{"metric": "acc", "op": ">=", "value": 0.5}],
        },
    )
    nan_result = ProbeResult(
        model="m",
        task="xor",
        config={"hidden_dim": 16},
        config_key="a",
        seed=0,
        status="ok",
        final_acc=float("nan"),
    )
    passed, reason = passes_stage(stage, [nan_result])
    assert not passed
    assert "fails" in reason


# ---------------------------------------------------------------------------
# max_params budget enforcement (architecture §6.3, §5.3)
# ---------------------------------------------------------------------------


BOARD_CAMPAIGN_YAML = """
meta: {name: budget, created: x}
compute: {device: cpu}
arms:
  a: {max_params: 100, models: [m_low, m_high]}
stages:
  - name: smoke
    task: xor
    epochs: 1
    seeds: 2
    configs: {hidden_dim: [8, 16]}
    pass_rule: {min_seed_ok: 1, rules: []}
"""


def test_staircase_skips_over_budget_configs(tmp_path: Path):
    """Configs whose param count exceeds the arm budget never train."""
    campaign = validate_yaml(BOARD_CAMPAIGN_YAML)
    report = Report(tmp_path / "r.jsonl")
    called: list[tuple[str, dict]] = []

    class Driver:
        def train(self, *, model, task, config, seed, epochs, device):
            called.append((model, config))
            return {"final_acc": 0.9, "epoch_time_s": 0.1}

    def counter(model, config, _input_dim, _output_dim):
        # m_low fits every config; m_high only fits hidden_dim=8.
        hidden = config["hidden_dim"]
        if model == "m_high" and hidden == 16:
            return 101
        return 50

    runner = StaircaseRunner(
        campaign,
        report,
        Driver(),
        HyperoptGridProducer(seed=1),
        param_counter=counter,
    )
    outcomes = runner.run()

    m_low, m_high = (
        {o.model: o for o in outcomes}["m_low"],
        {o.model: o for o in outcomes}["m_high"],
    )
    # m_low trains both configs x 2 seeds; m_high trains only hidden_dim=8 (2 seeds).
    assert m_low.verdict is Verdict.PASS
    assert m_high.verdict is Verdict.PASS
    trained_high = [c for (m, c) in called if m == "m_high"]
    assert len(trained_high) == 2  # 1 in-budget config x 2 seeds
    assert all(c["hidden_dim"] == 8 for c in trained_high)


def test_staircase_rejects_model_entirely_over_budget(tmp_path: Path):
    """A model whose every config blows the budget is REJECTED, not trained."""
    campaign = validate_yaml(BOARD_CAMPAIGN_YAML)
    report = Report(tmp_path / "r.jsonl")
    called: list[str] = []

    class Driver:
        def train(self, *, model, task, config, seed, epochs, device):
            called.append(model)
            return {"final_acc": 0.9, "epoch_time_s": 0.1}

    def counter(model, _config, _input_dim, _output_dim):
        return 500 if model == "m_high" else 50

    runner = StaircaseRunner(
        campaign,
        report,
        Driver(),
        HyperoptGridProducer(seed=1),
        param_counter=counter,
    )
    outcomes = runner.run()
    m_high = next(o for o in outcomes if o.model == "m_high")
    assert m_high.verdict is Verdict.REJECT
    assert "all configs exceed max_params=100" in m_high.reason
    assert all(m != "m_high" for m in called)


def test_plan_run_budget_consistency(tmp_path: Path):
    """plan's probe count matches what run schedules (both filter by budget)."""
    campaign = validate_yaml(
        """
meta: {name: b, created: x}
compute: {device: cpu}
arms: {a: {max_params: 100, models: [m_low, m_high]}}
stages:
  - name: smoke
    task: xor
    epochs: 1
    seeds: 2
    configs: {hidden_dim: [8, 16]}
    pass_rule: {min_seed_ok: 1, rules: []}
        """
    )
    from bioplausible.experiment.cli import _in_budget_pairs

    models = [m for arm in campaign.arms.values() for m in arm.models]

    def counter(model, config, _input_dim, _output_dim):
        if model == "m_high" and config["hidden_dim"] == 16:
            return 101
        return 50

    pairs = _in_budget_pairs(
        campaign, campaign.stages[0], models, param_counter=counter
    )
    # m_low: 2 configs; m_high: 1 config (hidden=16 filtered) -> 3 pairs x 2 seeds.
    assert len(pairs) == 3
    assert all(p.config["hidden_dim"] != 16 or p.model != "m_high" for p in pairs)
    assert {p.model for p in pairs} == {"m_low", "m_high"}
