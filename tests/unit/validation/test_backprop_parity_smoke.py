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


# ---------------------------------------------------------------------------
# Three-contract report (Plan 8 §15.4): width_matched / compute_matched /
# capacity_controlled. Effect sizes (Cohen's d, Cliff's δ, bootstrap_p) are
# part of the §C2 contract.
# ---------------------------------------------------------------------------


def test_parity_emits_three_contracts_per_cell(tmp_path: Path):
    """Each non-baseline cell produces the three §15.4 comparison contracts.

    Capacity-controlled is gated by ``width_ladder`` being non-empty — the
    existing smoke build passes the default ladder, so the row count per cell
    is exactly 3. (Dropping the ladder keeps width/compute-matched — only the
    capacity-controlled arm is optional.)
    """
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
    contracts = {c["contract"] for c in report["comparisons"]}
    assert contracts == {
        bp.Contract.WIDTH_MATCHED.value,
        bp.Contract.COMPUTE_MATCHED.value,
        bp.Contract.CAPACITY_CONTROLLED.value,
    }, (
        "All three §15.4 contracts must appear in a full-ladder run; got "
        f"{sorted(contracts)}"
    )


def test_parity_compute_matched_carries_flops(tmp_path: Path):
    """The compute-matched row reports forward+backward FLOPs for both arms.

    FLOPs are the §15.4 honest currency for PC/eqprop families whose settling
    steps cost more compute than backprop's single forward+backward — the
    ``model_total_flops``/``baseline_total_flops`` fields expose that
    discrepancy rather than hiding it in the wall-clock column.
    """
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
    compute_rows = [
        c
        for c in report["comparisons"]
        if c["contract"] == bp.Contract.COMPUTE_MATCHED.value
    ]
    assert compute_rows, "compute_matched contract row missing"
    row = compute_rows[0]
    assert "model_total_flops" in row
    assert "baseline_total_flops" in row
    assert "flops_advantage" in row
    assert isinstance(row["flops_advantage"], bool)


def test_parity_capacity_controlled_reports_baseline_width(tmp_path: Path):
    """The capacity-controlled arm reports which backprop width it searched to.

    That ``baseline_width`` makes the §15.4 Tertiary arm auditable: someone
    reading the markdown can verify what backprop architecture the bio model
    was compared against, without re-deriving the width ladder.
    """
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
    cap_rows = [
        c
        for c in report["comparisons"]
        if c["contract"] == bp.Contract.CAPACITY_CONTROLLED.value
    ]
    assert cap_rows, "capacity_controlled row missing"
    row = cap_rows[0]
    assert "baseline_width" in row
    assert isinstance(row["baseline_width"], int)
    assert row["baseline_width"] > 0


def test_parity_effect_sizes_appear_when_seeds_ge_2(tmp_path: Path):
    """Effect sizes (Cohen's d, Cliff's δ, bootstrap_p) require ≥2 seeds.

    They are ``nan`` for 1-seed probes (the stats primitives are undefined);
    the parity runner degrades gracefully so a 1-seed smoke probe still
    produces a valid row, and ≥2-seed runs populate the fields.
    """
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
    rows_with_effects = [
        c
        for c in report["comparisons"]
        if c.get("cohen_d") is not None
        and c.get("cliff_delta") is not None
        and c.get("bootstrap_p") is not None
    ]
    assert rows_with_effects, (
        "≥2-seed parity rows must populate the §C2 effect-size fields"
    )
    import math

    cohen = float(rows_with_effects[0]["cohen_d"])
    assert not math.isnan(cohen), "cohen_d should be a finite number for ≥2 seeds"


def test_parity_recovery_when_ladder_disabled(tmp_path: Path):
    """Passing an empty width ladder disables only the capacity-controlled arm.

    The other two contracts still produce rows; this is the supported route for
    very small cells where a backprop retrain would dominate the elapsed time.
    """
    report = bp.run_parity(
        task="digits",
        depths=(2,),
        hidden_dims=(16,),
        seeds=1,
        epochs=1,
        device="cpu",
        families=("backprop", "target_prop"),
        output_dir=tmp_path,
        width_ladder=(),
    )
    contracts = {c["contract"] for c in report["comparisons"]}
    assert contracts == {
        bp.Contract.WIDTH_MATCHED.value,
        bp.Contract.COMPUTE_MATCHED.value,
    }, (
        "Empty width_ladder disables the capacity-controlled arm only; got "
        f"{sorted(contracts)}"
    )
