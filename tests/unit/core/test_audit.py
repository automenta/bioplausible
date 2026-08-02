"""Registry audit CLI tests — Sprint 2.5.

Validates that ``biopl-registry-audit`` enumerates every component with
complete critical calibration metadata, and that the CSV / markdown / JSON
emitters behave as documented.
"""

import csv
import io
import json

# Importing the zoo/equitile packages is what registers the components the
# audit enumerates; the module does this lazily inside ``audit_rows``.
from bioplausible import zoo  # ruff: ignore[unused-import]
from bioplausible.core.audit import (
    CRITICAL_FIELDS,
    AuditRow,
    _missing_critical,
    audit_rows,
    emit_csv,
    emit_markdown,
    main,
)

MIN_EXPECTED_COMPONENTS = 50


def test_audit_enumerates_all_components():
    rows = audit_rows()
    # Model zoo + equitile registration should populate well over fifty entries,
    # spanning every category the registry supports.
    assert len(rows) >= MIN_EXPECTED_COMPONENTS
    categories = {row.category for row in rows}
    assert {"model", "propagator", "optimizer"} <= categories


def test_no_component_missing_critical_fields():
    rows = audit_rows()
    missing = _missing_critical(rows)
    assert missing == [], f"{len(missing)} components missing {CRITICAL_FIELDS}"


def test_every_algorithm_component_has_a_family():
    # Tracks/metrics are experiment scaffolding, not learning algorithms, so
    # ``family`` only has meaning for rule-bearing categories.
    algorithm_categories = {
        "model",
        "propagator",
        "optimizer",
        "update_strategy",
        "constraint",
        "controller",
        "sparsity",
    }
    rows = audit_rows()
    no_family = [
        row for row in rows if row.category in algorithm_categories and not row.family
    ]
    assert no_family == [], [f"{r.category}/{r.name}" for r in no_family]


def test_critical_field_values_are_calibrated():
    rows = audit_rows()
    for row in rows:
        assert 0.0 <= float(row.bio_plausibility_score) <= 1.0, row.name
        assert row.locality_level in {
            "global",
            "layerwise",
            "local",
            "equilibrium",
            "forward-only",
        }


def test_emit_csv_roundtrip():
    rows = audit_rows()
    out = io.StringIO()
    emit_csv(rows, out)
    out.seek(0)
    reader = csv.DictReader(out)
    parsed = list(reader)
    assert len(parsed) == len(rows)
    assert {"name", "family", "parity_status", "test_coverage"} <= set(parsed[0].keys())


def test_emit_markdown_table():
    rows = audit_rows()
    md = emit_markdown(rows)
    assert md.startswith("| Model")
    header_lines = md.splitlines()
    assert "---" in header_lines[1]
    # Every row appears exactly once as a table row.
    for row in rows:
        assert f"| {row.name} |" in md


def test_main_metadata_csv_exit_zero(capsys):
    code = main(["--metadata"])
    captured = capsys.readouterr()
    assert code == 0
    assert "name,category,family,bio_plausibility_score" in captured.out


def test_main_json_output(capsys):
    code = main(["--json"])
    captured = capsys.readouterr()
    assert code == 0
    first = json.loads(captured.out.splitlines()[0])
    assert "bio_plausibility_score" in first


def test_main_fails_when_critical_field_empty(monkeypatch):
    bad = AuditRow(
        name="bad",
        category="model",
        family="x",
        bio_plausibility_score="",  # missing — must trip the gate
        locality_level="global",
        memory_complexity="O(N)",
        requires_backward=True,
        credit_assignment_type="gradient",
        parity_status="n/a",
        test_coverage="n/a",
    )
    monkeypatch.setattr("bioplausible.core.audit.audit_rows", lambda: [bad])
    assert main([]) == 1
