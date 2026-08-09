"""Compute-matched parity smoke test (Plan 8 Track D3).

Runs the full parity pipeline at the smallest possible scale (1 depth, 1 seed,
1 epoch, tiny width) on CPU and asserts the contract: no crash, both report
files are written, note-worthy out-of-budget parameter mismatches are surfaced,
and the statistics utilities are exercised (bootstrap CI appears in JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioplausible.validation import backprop_parity as bp


def test_parity_smoke_completes_and_writes_reports(tmp_path: Path):
    report = bp.build_report(
        task="digits",
        depths=(2,),
        hidden_dims=(16,),
        seeds=1,
        epochs=1,
        device="cpu",
        families=("backprop", "target_prop"),
        output_dir=str(tmp_path),
    )

    results = tmp_path / "results.json"
    md = tmp_path / "report.md"
    assert results.exists(), "results.json must be written"
    assert md.exists(), "report.md must be written"

    data = json.loads(results.read_text(encoding="utf-8"))
    assert data["task"] == "digits"
    assert data["depths"] == [2]
    # Baseline model present.
    assert "backprop_mlp@2" in data["models"]
    # At least one comparison cell (target_prop vs backprop).
    assert len(data["comparisons"]) >= 1
    assert any(c["family"] == "target_prop" for c in data["comparisons"])


def test_parity_baseline_present_in_markdown(tmp_path: Path):
    bp.build_report(
        task="digits",
        depths=(2,),
        hidden_dims=(16,),
        seeds=1,
        epochs=1,
        device="cpu",
        families=("backprop",),
        output_dir=str(tmp_path),
    )
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Compute-Matched Parity Report" in md
    assert "backprop_mlp" in md


def test_parity_uses_statistics_utilities(tmp_path: Path):
    """The bootstrap CI utility feeds the per-model accuracy_ci95 field."""
    report = bp.build_report(
        task="digits",
        depths=(2,),
        hidden_dims=(16,),
        seeds=2,
        epochs=1,
        device="cpu",
        families=("backprop", "target_prop"),
        output_dir=str(tmp_path),
    )
    target = next(
        v for k, v in report["models"].items() if v["model"] == "diff_target_prop"
    )
    lo, hi = target["accuracy_ci95"]
    assert lo <= target["mean_accuracy"] <= hi


def test_parity_tier_classification():
    base = 0.80
    assert bp.parity_tier(0.81, base) == "strong"  # within 2% -> strong
    # within 5% AND advantage -> acceptable
    assert bp.parity_tier(0.77, base, advantage=True) == "acceptable"
    # within 5% but no advantage -> negative (Tier 2 requires the advantage)
    assert bp.parity_tier(0.77, base, advantage=False) == "negative"
    # more than 5% below, even with advantage -> negative
    assert bp.parity_tier(0.70, base, advantage=True) == "negative"


@pytest.mark.parametrize(
    "acc, baseline, advantage, expected",
    [
        (0.81, 0.80, False, "strong"),
        (0.80, 0.80, True, "strong"),
        (0.79, 0.80, False, "strong"),  # within 2% -> strong regardless of advantage
        (0.78, 0.80, False, "strong"),  # exactly 2% below
        (0.77, 0.80, True, "acceptable"),
        (0.77, 0.80, False, "negative"),
        (0.74, 0.80, True, "negative"),  # >5% below
    ],
)
def test_parity_tier_table(acc, baseline, advantage, expected):
    assert bp.parity_tier(acc, baseline, advantage=advantage) == expected


def test_parity_fails_loudly_on_out_of_budget_params(tmp_path: Path):
    """A param mismatch beyond tolerance is surfaced in notes (not hidden)."""
    report = bp.build_report(
        task="digits",
        depths=(2,),
        hidden_dims=(8,),  # tiny width forces target_prop params far from backprop
        seeds=1,
        epochs=1,
        device="cpu",
        families=("backprop", "target_prop"),
        output_dir=str(tmp_path),
    )
    notes = "\n".join(report["notes"])
    assert "does not match baseline" in notes, notes
