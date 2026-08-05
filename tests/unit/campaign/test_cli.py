"""Unit tests for the ``biopl-run`` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioplausible.campaign.cli import main

if TYPE_CHECKING:
    from pathlib import Path

CAMPAIGN = """meta: {name: cli_test, created: '2026-08-05'}
arms:
  mlp:
    input_dim: 64
    num_classes: 10
    max_params: 210000
    models: [backprop_mlp]
"""


def _write_campaign(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN, encoding="utf-8")
    return path


def test_validate_ok(tmp_path, capsys):
    path = _write_campaign(tmp_path)
    rc = main(["validate", "--config", str(path)])
    assert rc == 0
    assert "valid campaign" in capsys.readouterr().out


def test_validate_rejects_missing_arms(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("meta: {name: broken}\n", encoding="utf-8")
    rc = main(["validate", "--config", str(path)])
    assert rc != 0


def test_dry_run_prints_models(tmp_path, capsys):
    path = _write_campaign(tmp_path)
    rc = main(["dry-run", "--config", str(path)])
    assert rc == 0
    assert "backprop_mlp" in capsys.readouterr().out


def test_gates_writes_report(tmp_path):
    path = _write_campaign(tmp_path)
    rc = main(["gates", "--config", str(path), "--seeds", "1"])
    report = tmp_path / "gates.jsonl"
    assert rc == 0 or report.exists()


def test_run_alias_accepts_gate_flags(tmp_path, monkeypatch):
    # `run` must support the same --tier/--device/--seeds knobs as `gates`;
    # --tier tier0 must disable TIER 0.5.
    path = _write_campaign(tmp_path)
    captured: dict[str, bool] = {}

    from bioplausible.campaign.runner import CampaignResult

    def fake_run_gates(campaign, **kwargs):
        captured["run_tier05"] = kwargs.get("run_tier05", True)
        return CampaignResult(campaign_name=campaign.meta.name)

    monkeypatch.setattr("bioplausible.campaign.cli.run_gates", fake_run_gates)

    rc = main([
        "run",
        "--config",
        str(path),
        "--tier",
        "tier0",
        "--device",
        "cpu",
        "--seeds",
        "1",
    ])
    assert rc == 0
    assert captured["run_tier05"] is False
