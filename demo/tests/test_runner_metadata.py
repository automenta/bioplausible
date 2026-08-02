"""Registry-metadata surfacing tests (Sprint 2.5 -> demo 3.2 tooltips)."""

import pytest

from runner import (
    TRAINABLE_MODELS,
    DemoPanel,
    default_trainer_config,
    model_metadata,
    run_headless,
)
from charts import parity_explanation


class TestModelMetadata:
    def test_known_model_returns_calibrated_fields(self):
        m = model_metadata("equitile")
        assert m["family"] == "equitile"
        assert 0.0 <= m["bio_plausibility_score"] <= 1.0
        assert m["locality_level"] in {"local", "global", "equilibrium", "forward-only"}
        assert m["requires_backward"] is False

    def test_backprop_low_bio_score(self):
        m = model_metadata("backprop_mlp")
        assert m["bio_plausibility_score"] <= 0.5
        assert m["requires_backward"] is True

    def test_unknown_model_degrades_to_empty(self):
        assert model_metadata("not_a_model") == {}


class TestParityExplanation:
    def test_no_note_for_small_gap(self):
        a = DemoPanel(trainer_config=default_trainer_config(model="equitile"))
        b = DemoPanel(trainer_config=default_trainer_config(model="backprop_mlp"))
        assert parity_explanation(a, b, gap=2.0) == ""

    def test_note_when_backward_free_model_lags(self):
        a = DemoPanel(trainer_config=default_trainer_config(model="equitile"))
        b = DemoPanel(trainer_config=default_trainer_config(model="backprop_mlp"))
        note = parity_explanation(a, b, gap=12.0)
        assert "gap expected" in note
        assert "equitile" in note

    def test_no_note_when_both_backward(self):
        a = DemoPanel(trainer_config=default_trainer_config(model="backprop_mlp"))
        b = DemoPanel(trainer_config=default_trainer_config(model="backprop_mlp"))
        assert parity_explanation(a, b, gap=12.0) == ""

    def test_threshold_drives_note_not_hardcoded_5pp(self):
        # pepita's documented parity_threshold is 0.2 (20 pp) — a 12 pp gap is
        # *above* the old hardcoded 5 pp but *below* pepita's ceiling, so no
        # note must appear. Proves the note reads the YAML-backed threshold.
        a = DemoPanel(trainer_config=default_trainer_config(model="pepita"))
        b = DemoPanel(trainer_config=default_trainer_config(model="backprop_mlp"))
        assert parity_explanation(a, b, gap=12.0) == ""
        assert parity_explanation(a, b, gap=25.0) != ""

    def test_model_metadata_surfaces_parity_threshold(self):
        assert model_metadata("pepita")["parity_threshold"] == 0.2
        assert model_metadata("eqprop_mlp")["parity_threshold"] == 0.05
        # Backprop has no documented bio-gap ceiling; default applies.
        assert model_metadata("backprop_mlp")["parity_threshold"] == 0.05


@pytest.mark.parametrize("model", TRAINABLE_MODELS)
def test_demo_model_trains_headless(model):
    """Sprint 3.3 CI smoke: every advertised demo model trains end-to-end.

    Guards the curated list so a broken CoreTrainer integration can't silently
    ship in the demo selector (this is the failure EquiTile/pepita/FF/FA hit).
    """
    # Unknown models in the metadata table would fail the tooltip lookup.
    assert model_metadata(model), f"{model} missing registry metadata"
    cfg = default_trainer_config(model=model, task="digits", epochs=1, lr=0.01)
    panel = DemoPanel(trainer_config=cfg, epochs=1)
    run_headless(panel)
    assert panel.error is None, f"{model} failed: {panel.error}"
    assert panel.finished
    assert panel.accuracies and all(x == x for x in panel.accuracies), (
        f"{model} produced no valid accuracies"
    )
