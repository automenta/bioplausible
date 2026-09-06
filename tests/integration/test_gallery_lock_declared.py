"""The declared-figure migration ratchet (Demo API roadmap item 1).

Every demo run record must DECLARE its figure (``data["figure"]`` spec,
the common demo API) — bespoke per-capability figure factories are dead
code. This lock walks the on-disk records and fails on any undeclared
record, so a new demo cannot register a bespoke factory or skip the
declaration.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = REPO_ROOT / "docs" / "figures" / "run_records"


def test_every_record_declares_its_figure() -> None:
    undeclared = [
        path.name
        for path in sorted(RECORDS_DIR.glob("*.json"))
        if "figure" not in json.loads(path.read_text(encoding="utf-8"))["data"]
    ]
    assert not undeclared, (
        f"{len(undeclared)} record(s) declare no figure spec: {undeclared} — "
        "declare panels under data['figure'] (computronium.visualization "
        "builders); bespoke factories are retired"
    )
