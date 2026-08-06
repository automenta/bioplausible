"""Unit tests for the experiment CLI (validate / plan / report)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioplausible.experiment.cli import main, main_report
from bioplausible.experiment.probe import ProbeResult, config_key
from bioplausible.experiment.report import Report

if TYPE_CHECKING:
    from pathlib import Path

_SMOKE_YAML = """
meta: {name: smoke, created: '2026-08-05'}
compute: {device: cpu, num_workers: 0}
arms:
  mlp: {max_params: 210000, models: [backprop_mlp]}
stages:
  - name: smoke
    task: xor
    epochs: 1
    seeds: 1
    configs: {hidden_dim: [16], num_layers: [2]}
    pass_rule: {min_seed_ok: 1, rules: [{metric: acc, op: '>=', value: 0.5}]}
reproducibility: {seed: 7}
"""

BAD_EVIDENCE_YAML = """
meta: {name: bad, created: '2026-08-05'}
compute: {device: cpu}
arms:
  mlp: {max_params: 210000, models: [backprop_mlp]}
stages:
  - name: parity
    task: xor
    epochs: 1
    seeds: 5
    baseline: backprop_mlp
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_validate_ok(tmp_path: Path):
    assert main(["validate", str(_write(tmp_path, _SMOKE_YAML))]) == 0


def test_validate_rejects_low_evidence_seeds(tmp_path: Path):
    assert main(["validate", str(_write(tmp_path, BAD_EVIDENCE_YAML))]) == 1


def test_validate_rejects_unknown_task(tmp_path: Path):
    yaml_text = _SMOKE_YAML.replace("task: xor", "task: not_a_task")
    assert main(["validate", str(_write(tmp_path, yaml_text))]) == 1


def test_plan_reports_probe_count(tmp_path: Path, capsys):
    rc = main(["plan", str(_write(tmp_path, _SMOKE_YAML))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "total probes: 1" in out
    assert "estimated total time" in out


def test_plan_unknown_config_fails(tmp_path: Path):
    assert main(["plan", str(tmp_path / "missing.yaml")]) == 1


def test_report_renders_parity_and_pareto(tmp_path: Path, capsys):
    report_path = tmp_path / "r.jsonl"
    result = ProbeResult(
        model="backprop_mlp",
        task="xor",
        config={"hidden_dim": 16, "num_layers": 2},
        config_key=config_key({"hidden_dim": 16, "num_layers": 2}),
        seed=0,
        status="ok",
        final_acc=0.8,
        param_count=82,
        epoch_time_s=1.0,
    )
    Report(report_path).append("smoke", result)

    assert main_report([str(report_path)]) == 0
    out = capsys.readouterr().out
    assert "stage: smoke" in out
    assert "backprop_mlp" in out
    assert "Pareto frontier" in out


def test_report_failure_manifesto(tmp_path: Path, capsys):
    report_path = tmp_path / "r.jsonl"
    result = ProbeResult(
        model="m",
        task="xor",
        config={},
        config_key="k",
        seed=0,
        status="error",
        error="boom",
    )
    Report(report_path).append("smoke", result)

    assert main_report([str(report_path)]) == 0
    out = capsys.readouterr().out
    assert "failure manifesto" in out
    assert "boom" in out
