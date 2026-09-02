"""Shared fixtures for the demonstrative integration suite (TODO10 R10.1.2).

Each demo test emits one deterministic JSON run record per capability:
same seed -> same record -> same figure. Records carry the run's provenance
(git commit, config hash) alongside the deterministic data payload; the
figure lock (R10.1.4) compares only the data layer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

RECORDS_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures" / "run_records"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except OSError, subprocess.SubprocessError:
        return "unknown"


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@pytest.fixture()
def emit_run_record(request: pytest.FixtureRequest) -> Callable[[str, str, dict], Path]:
    """Return an emitter writing ``docs/figures/run_records/<capability>.json``.

    Called before assertions so the record exists even on failure paths.
    """

    def emit(capability_id: str, capability_name: str, data: dict) -> Path:
        demo_test = str(request.path.relative_to(RECORDS_DIR.parents[2]))
        record = {
            "capability": capability_id,
            "capability_name": capability_name,
            "demo_test": demo_test,
            "provenance": {
                "git_commit": _git_commit(),
                "config_sha256": hashlib.sha256(_canonical(data).encode()).hexdigest(),
            },
            "data": data,
        }
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDS_DIR / f"{capability_id.lower()}_{capability_name}.json"
        path.write_text(_canonical(record), encoding="utf-8")
        return path

    return emit
