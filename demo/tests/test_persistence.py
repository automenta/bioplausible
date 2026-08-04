"""Experiment persistence tests (Sprint 3.6)."""

import json

import pytest
from persistence import (
    config_from_url,
    config_to_url,
    export_run_csv,
    export_run_png,
    export_summary,
    load_config,
    save_config,
)
from runner import default_trainer_config


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        cfg = default_trainer_config(
            model="eqprop_mlp", task="digits", epochs=3, lr=0.01
        )
        path = save_config(cfg, tmp_path / "cfg.json")
        back = load_config(path)
        assert back.model == cfg.model
        assert back.epochs == cfg.epochs
        assert back.task == cfg.task

    def test_json_is_loadable(self, tmp_path):
        cfg = default_trainer_config()
        path = save_config(cfg, tmp_path / "cfg.json")
        data = json.loads(path.read_text())
        assert data["model"] == "backprop_mlp"

    def test_export_summary(self):
        s = export_summary([1.0, 0.5], [0.2, 0.9], "eqprop_mlp", "mnist")
        assert s["final_accuracy"] == 0.9
        assert s["final_loss"] == 0.5
        assert s["n_epochs"] == 2

    def test_export_run_csv_roundtrip(self, tmp_path):
        p = export_run_csv([1.0, 0.5, 0.25], [0.2, 0.8, 0.95], tmp_path / "run.csv")
        lines = p.read_text().splitlines()
        assert lines[0] == "step,loss,accuracy"
        assert len(lines) == 4  # header + 3 steps

    def test_export_run_csv_with_header(self, tmp_path):
        p = export_run_csv([1.0], [], tmp_path / "run.csv", header={"model": "eqprop"})
        lines = p.read_text().splitlines()
        assert lines[0] == "#,model"
        assert lines[1] == "#,eqprop"


class TestShareableUrl:
    def test_roundtrip(self):
        cfg = default_trainer_config(
            model="eqprop_mlp", task="digits", epochs=4, lr=0.02
        )
        back = config_from_url(config_to_url(cfg))
        assert back.model == cfg.model
        assert back.task == cfg.task
        assert back.epochs == cfg.epochs
        assert back.optimizer_kwargs["lr"] == cfg.optimizer_kwargs["lr"]

    def test_url_prefix_and_no_secrets(self):
        url = config_to_url(default_trainer_config())
        assert url.startswith("bioplausible://")
        # Only compact knobs (optimizer_kwargs/model_kwargs), no secrets.
        assert "token" not in url

    def test_invalid_url_rejected(self):
        with pytest.raises(ValueError):
            config_from_url("https://example.com/x")


class TestPngExport:
    def test_writes_valid_png(self, tmp_path):
        p = export_run_png([1.0, 0.5], [0.2, 0.9], tmp_path / "run.png")
        assert p.exists()
        assert p.stat().st_size > 100
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
