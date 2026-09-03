"""Figure lock with teeth (TODO10 R10.1.4).

The gallery cannot silently drift from what the code actually demonstrates.
The data layer of every demo run record (recorded metric values, not
pixels) is checksummed into the committed manifest; this test regenerates
the gallery and compares. A mismatch means one of two things, both caught:
the code changed what it demonstrates (review the diff, re-pin the manifest
deliberately) or the demo became nondeterministic (a bug — fix it).

Runs after the demo tests (alphabetical file order in this directory), so
the records on disk are from the same gate run.
"""

import hashlib
import json
from pathlib import Path

from computronium.visualization.gallery import render_gallery

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
RECORDS_DIR = FIGURES_DIR / "run_records"

EXPECTED = {
    "compose_6axis": "D1",
    "swap_credit": "D2",
    "swap_plasticity": "D3",
    "memory_budget": "D4",
    "substrate_swap": "D6",
    "spike_settle": "D7",
    "z3_frozen_theta": "D5",
    "geometry_swap": "D8",
    "graph_geometry_swap": "D9",
    "attention_geometry_swap": "D10",
    "spatial_lattice_geometry_swap": "D11",
}


def _records() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(RECORDS_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["capability_name"]] = record
    return records


def test_figure_lock(tmp_path: Path) -> None:
    records = _records()
    assert set(records) == set(EXPECTED), (
        f"missing demo records: {sorted(set(EXPECTED) - set(records))}"
    )

    for name, record in records.items():
        assert (REPO_ROOT / record["demo_test"]).exists(), (
            f"stale record for {name}: demo test deleted, claim must retire"
        )
        data_sha = hashlib.sha256(
            json.dumps(record["data"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert data_sha == record["provenance"]["config_sha256"], (
            f"record for {name} was tampered with or the emitter changed"
        )

    metas = render_gallery(RECORDS_DIR, tmp_path)
    assert {m.capability_name for m in metas} == set(EXPECTED)

    manifest = json.loads((FIGURES_DIR / "manifest.json").read_text(encoding="utf-8"))
    pinned = {f["capability_name"]: f for f in manifest["figures"]}
    assert set(pinned) == set(EXPECTED), "manifest has orphaned or missing capabilities"
    for name, entry in pinned.items():
        fresh = next(m for m in metas if m.capability_name == name)
        assert entry["data_sha256"] == fresh.data_sha256, (
            f"figure data for {name} drifted from the pinned manifest — "
            "review the demo diff, then re-pin deliberately"
        )
        assert entry["demo_test"] == fresh.demo_test
