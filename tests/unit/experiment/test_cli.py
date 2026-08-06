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


def test_plan_unexpected_error_resume_hint(tmp_path: Path, capsys, monkeypatch):
    """A mid-run driver crash must not lose the resume contract.

    Regression for the overnight run: an unexpected exception inside the
    cascade used to escape as a bare traceback. It must instead tell the
    operator the Report is resumable and return a non-zero exit.
    """
    import bioplausible.experiment.cli as cli

    class ExplodingRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        @staticmethod
        def run():
            raise RuntimeError("boom")

    # Swap only the runner construction; Report is still real so we get a path.
    original = cli.StaircaseRunner
    cli.StaircaseRunner = ExplodingRunner  # type: ignore[assignment]
    try:
        rc = main(["run", str(_write(tmp_path, _SMOKE_YAML))])
    finally:
        cli.StaircaseRunner = original  # type: ignore[assignment]

    out = capsys.readouterr().out
    assert rc == 1
    assert "resumable" in out
    assert "rerun to continue" in out


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


def _ok_probe(
    model: str, config: dict[str, object], seed: int, acc: float, n_params: int
) -> ProbeResult:
    return ProbeResult(
        model=model,
        task="xor",
        config=config,
        config_key=config_key(config),
        seed=seed,
        status="ok",
        final_acc=acc,
        param_count=n_params,
        epoch_time_s=1.0,
    )


def test_report_dummy_data_full_render(tmp_path: Path, capsys):
    """Render a multi-model dummy Report: parity table, effect sizes, Pareto, failures.

    Regression against the full report path using synthetic probes (no training):
    backprop is the baseline; eqprop_mlp has worse acc + more params (dominated);
    a high-acc low-param config sits on the frontier; one probe errors.
    """
    report_path = tmp_path / "r.jsonl"
    report = Report(report_path)

    config_lo = {"hidden_dim": 16, "num_layers": 2}
    config_hi = {"hidden_dim": 64, "num_layers": 4}
    for seed in range(3):
        report.append(
            "parity",
            _ok_probe(
                "backprop_mlp", config_lo, seed, acc=0.80 + seed * 0.01, n_params=82
            ),
        )
        report.append(
            "parity",
            _ok_probe(
                "eqprop_mlp", config_lo, seed, acc=0.60 + seed * 0.01, n_params=3000
            ),
        )
    # A dominant config: higher acc, fewer params than both — must be on the frontier.
    report.append(
        "parity",
        _ok_probe("backprop_mlp", config_hi, 0, acc=0.95, n_params=58),
    )
    # A failing probe — must appear in the failure manifesto, never on a table row.
    report.append(
        "parity",
        ProbeResult(
            model="broken",
            task="xor",
            config={},
            config_key="k",
            seed=0,
            status="error",
            error="boom",
        ),
    )
    report.append(
        "parity",
        ProbeResult(
            model="backprop_mlp",
            task="xor",
            config={},
            config_key="k2",
            seed=9,
            status="error",
            error="epoch NaN",
        ),
    )

    assert main_report([str(report_path), "--baseline", "backprop_mlp"]) == 0
    out = capsys.readouterr().out

    assert "stage: parity" in out
    assert "effect sizes vs baseline backprop_mlp" in out
    assert "eqprop_mlp" in out
    assert "Pareto frontier" in out
    # The dominant config's hash is the frontier point.
    assert config_key(config_hi) in out
    # The dominated config hash never appears on the frontier.
    assert config_key(config_lo) not in out.split("Pareto frontier")[1]
    assert "failure manifesto" in out
    assert "epoch NaN" in out


def test_report_pareto_frontier_dominance():
    """Dummy-data check of the dominance rule: fewer params + higher acc wins."""
    from bioplausible.experiment.reporting import pareto_frontier

    outcomes = [
        _ok_probe("m", {"a": 1}, 0, 0.70, 500),
        _ok_probe("m", {"a": 2}, 0, 0.90, 60),  # dominates the 500-param point
    ]
    frontier = pareto_frontier(outcomes)
    assert len(frontier) == 1
    assert frontier[0]["config_key"] == config_key({"a": 2})
    # The 500-param, lower-acc config is dominated and absent.
    assert config_key({"a": 1}) not in {p["config_key"] for p in frontier}


# ---------------------------------------------------------------------------
# --time-budget auto-scale
# ---------------------------------------------------------------------------


def test_parse_time_budget_units():
    from bioplausible.experiment.cli import _parse_time_budget

    assert _parse_time_budget("1h") == 3600
    assert _parse_time_budget("90m") == 5400
    assert _parse_time_budget("300s") == 300
    assert _parse_time_budget("7200") == 7200


def test_auto_scale_reduces_epochs_to_fit_budget():
    from bioplausible.experiment.cli import _auto_scale_campaign
    from bioplausible.experiment.producer import HyperoptGridProducer
    from bioplausible.experiment.schema import validate_yaml

    campaign = validate_yaml(_SMOKE_YAML)
    models = [m for arm in campaign.arms.values() for m in arm.models]
    producer = HyperoptGridProducer(seed=42)
    # 1 epoch on xor ~ cheap; use a deliberately slow epoch to force scaling.
    epoch_times = {("backprop_mlp", "xor"): 100.0}
    scaled = _auto_scale_campaign(campaign, 50, epoch_times, models, producer)
    assert scaled.stages[0].epochs == 1  # clamped to the floor
    assert scaled.stages[0].seeds == 1


def test_auto_scale_noop_when_already_fits():
    from bioplausible.experiment.cli import _auto_scale_campaign
    from bioplausible.experiment.producer import HyperoptGridProducer
    from bioplausible.experiment.schema import validate_yaml

    campaign = validate_yaml(_SMOKE_YAML)
    models = [m for arm in campaign.arms.values() for m in arm.models]
    producer = HyperoptGridProducer(seed=42)
    epoch_times = {("backprop_mlp", "xor"): 0.01}
    scaled = _auto_scale_campaign(campaign, 1000, epoch_times, models, producer)
    assert scaled.stages[0].epochs == campaign.stages[0].epochs
    assert scaled.stages[0].configs == campaign.stages[0].configs
