"""Registry audit tooling — Sprint 2.5.

Enumerates every registered component and exposes its calibrated metadata as
CSV / markdown / JSON. The outputs feed three artifacts:

* ``biopl-registry-audit --metadata`` — the Sprint 2.5 calibration CSV
  (name, family, bio_plausibility_score, locality_level, memory_complexity,
  requires_backward, credit_assignment_type, parity_status, test_coverage).
* ``biopl-registry-audit --markdown`` — the live component table injected into
  the README (Sprint 4.6).
* ``biopl-registry-audit`` (default) — a completeness gate that fails when any
  component is missing a critical calibration field (``bio_plausibility_score``
  or ``locality_level``).

``test_coverage`` is reported as ``n/a`` here because per-component coverage
is not tracked at audit time; the global coverage floor is enforced separately
by the pytest ``--cov-fail-under`` gate in CI (Sprint 5.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import TextIO, cast

from bioplausible.core.registry import ComponentCategory, ComponentMetadata, Registry

# Critical calibration fields that must never be empty (Sprint 2.5 CI gate).
CRITICAL_FIELDS: tuple[str, ...] = ("bio_plausibility_score", "locality_level")

# Uniform parity threshold used by the backprop-parity suite (Sprint 1.5.2).
PARITY_THRESHOLD = 0.05

# Model names that participate in the backprop-parity suite, mapped to the
# ``parity_threshold`` (>0.05 ⇒ a documented biological gap, Sprint 1.5.3).
_PARITY_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tests"
    / "unit"
    / "validation"
    / "hyperparams"
)


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One serializable row of the registry audit."""

    name: str
    category: str
    family: str
    bio_plausibility_score: float | str
    locality_level: str
    memory_complexity: str
    requires_backward: bool
    credit_assignment_type: str
    parity_status: str
    test_coverage: str

    def as_dict(self) -> dict[str, object]:
        """Return the row as an ordered mapping for CSV/JSON serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "family": self.family,
            "bio_plausibility_score": self.bio_plausibility_score,
            "locality_level": self.locality_level,
            "memory_complexity": self.memory_complexity,
            "requires_backward": self.requires_backward,
            "credit_assignment_type": self.credit_assignment_type,
            "parity_status": self.parity_status,
            "test_coverage": self.test_coverage,
        }


def _load_registry() -> None:
    """Import the registration modules so every component is present."""
    import bioplausible.zoo  # ruff: ignore[unused-import]


def _parity_status(name: str) -> str:
    """Classify a model's parity status from its hyperparameter YAML.

    Models absent from the parity directory report ``n/a``; those with a
    ``parity_threshold`` of 0.05 report ``pass``; anything stricter than the
    uniform 0.05 default is a documented biological gap.
    """
    yaml_path = _PARITY_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        return "n/a"
    for line in yaml_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("parity_threshold:"):
            try:
                threshold = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                return "unknown"
            return "pass" if threshold <= PARITY_THRESHOLD else "documented-gap"
    return "unknown"


def collect_rows() -> list[AuditRow]:
    """Enumerate every registered component as :class:`AuditRow`."""
    _load_registry()
    rows: list[AuditRow] = []
    for category in ComponentCategory:
        for name, info in Registry._components.get(category.value, {}).items():
            meta = cast("ComponentMetadata | None", info.get("metadata"))
            rows.append(
                AuditRow(
                    name=name,
                    category=category.value,
                    family=meta.family if meta and meta.family else "",
                    bio_plausibility_score=(
                        meta.bio_plausibility_score
                        if meta and meta.bio_plausibility_score is not None
                        else ""
                    ),
                    locality_level=(
                        meta.locality_level.value
                        if meta and meta.locality_level is not None
                        else ""
                    ),
                    memory_complexity=(
                        meta.memory_complexity
                        if meta and meta.memory_complexity
                        else ""
                    ),
                    requires_backward=bool(meta and meta.requires_backward),
                    credit_assignment_type=(
                        meta.credit_assignment_type
                        if meta and meta.credit_assignment_type
                        else ""
                    ),
                    parity_status=_parity_status(name),
                    test_coverage="n/a",
                )
            )
    return rows


def audit_rows() -> list[AuditRow]:
    """Public alias kept for test convenience; returns all component rows."""
    return collect_rows()


def _missing_critical(rows: list[AuditRow]) -> list[AuditRow]:
    """Return rows missing any critical calibration field."""
    return [
        row
        for row in rows
        if str(row.bio_plausibility_score) == "" or row.locality_level == ""
    ]


def emit_csv(rows: list[AuditRow], out: TextIO) -> None:
    """Write the audit to an open CSV file object."""
    writer = csv.DictWriter(out, fieldnames=list(rows[0].as_dict().keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())


def emit_markdown(rows: list[AuditRow]) -> str:
    """Render a README-friendly component table."""
    header = ["Model", "Family", "Bio Score", "Locality", "Credit Assignment", "Parity"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in sorted(rows, key=lambda r: (r.category, r.name)):
        lines.append(
            "| "
            + " | ".join([
                row.name,
                row.family or "—",
                str(row.bio_plausibility_score),
                row.locality_level,
                row.credit_assignment_type,
                row.parity_status,
            ])
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``biopl-registry-audit``."""
    parser = argparse.ArgumentParser(description="Audit the bioplausible registry")
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Emit the full Sprint 2.5 calibration CSV (name, family, locality, ...)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a README-ready component table",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit rows as JSON lines",
    )
    args = parser.parse_args(argv)

    rows = audit_rows()
    missing = _missing_critical(rows)
    if missing:
        print(
            f"Registry audit FAILED: {len(missing)} component(s) missing a critical "
            f"field ({'/'.join(CRITICAL_FIELDS)}):",
            file=sys.stderr,
        )
        for row in missing:
            print(f"  - {row.category}/{row.name}", file=sys.stderr)
        return 1

    if args.markdown:
        print(emit_markdown(rows))
    elif args.json:
        print("\n".join(json.dumps(row.as_dict()) for row in rows))
    else:
        emit_csv(rows, sys.stdout)
    print(f"# {len(rows)} components, 0 missing critical fields", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
