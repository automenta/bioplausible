"""Experiment persistence tests (Sprint 3.6)."""

import json

from persistence import export_summary, load_config, save_config
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
