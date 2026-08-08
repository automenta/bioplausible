"""Sweep self-diagnosis defect flag (EXPERIMENT_PLAN5 §1).

A bio-family probe that fell back to the silent BPTT path is a defect and
must be auto-surfaced in the sweep report — never audited by a human.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[3] / "scripts" / "broad_sweep.py"
_spec = importlib.util.spec_from_file_location("broad_sweep", _SCRIPT)
bs = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(bs)  # type: ignore[union-attr]


def _probe(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {"ok": True, "loss_epoch_0": 1.0, "loss_epoch_final": 0.5}
    base.update(kw)
    return base


def test_bio_defect_detected() -> None:
    """A bio probe whose dominant path is bptt is flagged as defective."""
    runs = [_probe(training_path="bptt")]
    assert bs._bio_defect(runs)


def test_bio_correct_local_rule_is_not_defect() -> None:
    runs = [
        _probe(training_path="propagator"),
        _probe(training_path="model_train_step"),
    ]
    assert not bs._bio_defect(runs)


def test_failed_probe_is_not_a_defect_signal() -> None:
    runs = [_probe(ok=False, error="boom")]
    assert not bs._bio_defect(runs)


def test_bio_family_flags_bptt_model() -> None:
    """eqprop is a bio family; a model that fell back to BPTT is a defect."""
    family_runs = {"looped_mlp": [_probe(training_path="bptt")]}
    out = bs._summarize_family(family_runs, determined=True, family="eqprop")
    assert out["defects"] == ["looped_mlp"]
    assert out["models"]["looped_mlp"]["defect"] is True


def test_non_bio_family_never_flagged() -> None:
    """backprop uses BPTT by design — it is not a defect."""
    family_runs = {"backprop_mlp": [_probe(training_path="bptt")]}
    out = bs._summarize_family(family_runs, determined=True, family="backprop")
    assert out["defects"] == []
    assert out["models"]["backprop_mlp"]["defect"] is False


def test_default_family_is_not_flagged() -> None:
    """Without a family name the defect gate stays off (non-bio default)."""
    family_runs = {"model_a": [_probe(training_path="bptt")]}
    out = bs._summarize_family(family_runs, determined=True)
    assert out["defects"] == []
    assert out["models"]["model_a"]["defect"] is False
