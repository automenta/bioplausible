"""``comp gallery``: render the demo suite's figures (TODO10 R10.2.10).

Reads the demo run records (re-running the demo suite first with ``--run``),
renders one figure per capability into ``docs/figures/`` plus a manifest,
and exits nonzero on any missing or drifted record — a record whose data
checksum differs from the previous manifest means the demo changed what it
shows or became nondeterministic.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

from computronium.visualization.gallery import (
    _FACTORIES,
    RECORDS_DIRNAME,
    render_gallery,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
DEMO_TESTS = ("tests/integration", "-k", "demo")


def _run_demo_suite() -> int:
    cmd = [sys.executable, "-m", "pytest", *DEMO_TESTS, "-q"]
    return subprocess.run(cmd, check=False).returncode  # ruff: ignore[subprocess-without-shell-equals-true]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="comp gallery", description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the demo suite first to regenerate the run records",
    )
    args = parser.parse_args(argv)

    if args.run and _run_demo_suite() != 0:
        print("gallery: demo suite failed; refusing to render", file=sys.stderr)
        return 1

    records_dir = FIGURES_DIR / RECORDS_DIRNAME
    missing = sorted(
        name for name in _FACTORIES if not any(records_dir.glob(f"*_{name}.json"))
    )
    if missing:
        print(
            f"gallery: missing run records for {missing}; run `comp gallery --run`",
            file=sys.stderr,
        )
        return 1

    previous = _previous_data_hashes()
    metas = render_gallery(records_dir, FIGURES_DIR)
    drifted = [
        m.capability_name
        for m in metas
        if previous.get(m.capability_name, m.data_sha256) != m.data_sha256
    ]
    if drifted:
        print(
            f"gallery: data layer changed for {drifted} — either the demo "
            "changed what it demonstrates (review and re-pin deliberately) "
            "or it became nondeterministic (a bug)",
            file=sys.stderr,
        )
        return 1

    rendered = [m.figure_png for m in metas]
    print(f"gallery: rendered {len(rendered)} figures -> {FIGURES_DIR}")
    for png in rendered:
        print(f"  {png}")
    return 0


def _previous_data_hashes() -> dict[str, str]:
    manifest = FIGURES_DIR / "manifest.json"
    if not manifest.exists():
        return {}
    figures: list[dict] = json.loads(manifest.read_text(encoding="utf-8"))["figures"]
    return {f["capability_name"]: f["data_sha256"] for f in figures}


if __name__ == "__main__":
    raise SystemExit(main())
