"""Unit tests for the experiment layer (schema, report, producer, staircase)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioplausible.experiment.probe import ProbeResult, config_key, run_probe
from bioplausible.experiment.producer import (
    ConfigProducer,
    HyperoptGridProducer,
    ProbeWork,
    grid_cardinality,
)
from bioplausible.experiment.report import Report
from bioplausible.experiment.schema import Stage, validate_yaml
from bioplausible.experiment.staircase import (
    Outcome,
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
        validate_yaml(
            CAMPAIGN_YAML.replace("task: xor", "task: not_a_task")
        )


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
    stage = Stage(name="s", task="xor", epochs=1, seeds=1, configs={
        "hidden_dim": [16, 32],
        "num_layers": [2, 4],
    })
    assert grid_cardinality(stage.configs) == 4


def test_producer_schedules_exact_probe_count():
    campaign = validate_yaml(CAMPAIGN_YAML)
    stage = campaign.stages[0]
    producer = HyperoptGridProducer(seed=7)
    works = list(producer.schedule(stage, ["backprop_mlp"]))
    assert len(works) == grid_cardinality(stage.configs)
    assert all(isinstance(w, ProbeWork) for w in works)


def test_producer_skips_finished():
    campaign = validate_yaml(CAMPAIGN_YAML)
    stage = campaign.stages[0]
    producer = HyperoptGridProducer(seed=7)
    all_works = list(producer.schedule(stage, ["backprop_mlp"]))
    finished = {w.config_key for w in all_works[:1]}
    remaining = list(producer.schedule(stage, ["backprop_mlp"], finished=finished))
    assert len(remaining) == len(all_works) - 1


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
        model="m", task="xor", config={"a": 1}, config_key=bad.config_key,
        seed=0, status="error", error="boom",
    )
    report.append("s", bad)
    assert not report.is_finished("s", bad)


# ---------------------------------------------------------------------------
# Staircase / verdicts
# ---------------------------------------------------------------------------


def test_passes_stage_all_rules_ok():
    stage = Stage(
        name="smoke", task="xor", epochs=1, seeds=2,
        configs={"hidden_dim": [16]},
        pass_rule={"min_seed_ok": 1, "rules": [{"metric": "acc", "op": ">=", "value": 0.5}]},
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
        name="smoke", task="xor", epochs=1, seeds=1,
        configs={"hidden_dim": [16]},
        pass_rule={"min_seed_ok": 1, "rules": [{"metric": "acc", "op": ">=", "value": 0.9}]},
    )
    results = [_result("m", "xor", {"hidden_dim": 16}, 0, 0.3)]
    passed, reason = passes_stage(stage, results)
    assert not passed
    assert "fails" in reason


def test_passes_stage_respects_min_seed_ok():
    stage = Stage(
        name="smoke", task="xor", epochs=1, seeds=3,
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
        name="smoke", task="xor", epochs=1, seeds=1,
        configs={"hidden_dim": [16]},
        pass_rule={"min_seed_ok": 1, "rules": [{"metric": "acc", "op": ">=", "value": 0.5}]},
    )
    errored = ProbeResult(
        model="m", task="xor", config={"hidden_dim": 16},
        config_key="x", seed=0, status="error", error="boom",
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
arms: {mlp: {max_params: 1, models: [backprop_mlp, eqprop_mlp]}}
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
    backprop_B = [o for o in outcomes if o.model == "backprop_mlp" and o.verdict is Verdict.PASS]
    eqprop_B = [o for o in outcomes if o.model == "eqprop_mlp" and o.verdict is Verdict.PASS]
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


def test_metrics_aggregate_median():
    metrics = StageMetrics([
        _result("m", "xor", {}, 0, 0.8),
        _result("m", "xor", {}, 1, 0.9),
        _result("m", "xor", {}, 2, 0.85),
    ])
    assert metrics.value("acc", "median") == pytest.approx(0.85)
    assert metrics.value("acc", "mean") == pytest.approx(0.85)
