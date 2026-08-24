"""Artifact path resolution (REFACTOR5 ROOT HYGIENE).

Experiment databases are outputs, not source code; they live under
``artifacts/`` (or the ``BIOPL_DB_DIR`` override) so the project root stays
clean. ``mkdir`` is idempotent so callers can write through the returned path
immediately.
"""

import os
from pathlib import Path


def artifacts_dir() -> Path:
    base = Path(os.environ.get("BIOPL_DB_DIR", "artifacts"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path(name: str) -> str:
    """Resolve an artifact database file path, creating its parent directory."""
    return str(artifacts_dir() / name)
