"""Validation tracks CLI (``biopl validate``).

Runs the comprehensive verification suite and optionally records results
to the knowledge base for integration with ``biopl report`` and
``biopl failure-manifesto``.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biopl validate",
        description="Run validation tracks and record to knowledge base",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (smoke test, ~2 min)"
    )
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Intermediate mode (directional, ~1 hour)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Full mode (statistically significant, ~4+ hr)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="Comma-separated track IDs to run (default: all)",
    )
    parser.add_argument(
        "--record-kb",
        action="store_true",
        help="Record track results to knowledge base (for biopl report / failure-manifesto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for verification notebook",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run tracks in parallel"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tracks and exit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``biopl validate``."""
    args = _build_parser().parse_args(argv)

    # Determine mode
    if args.full:
        quick_mode = False
        intermediate_mode = False
    elif args.intermediate:
        quick_mode = False
        intermediate_mode = True
    else:
        quick_mode = True
        intermediate_mode = False

    # Import here to avoid loading heavy modules at dispatcher level
    from bioplausible.validation.core import Verifier

    verifier = Verifier(
        quick_mode=quick_mode,
        intermediate_mode=intermediate_mode,
        seed=args.seed,
        output_dir=args.output_dir,
        record_to_kb=args.record_kb,
    )

    if args.list:
        verifier.list_tracks()
        return 0

    track_ids = None
    if args.tracks:
        track_ids = [int(t.strip()) for t in args.tracks.split(",")]

    try:
        results = verifier.run_tracks(track_ids=track_ids, parallel=args.parallel)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1

    # Print summary
    passed = sum(1 for r in results.values() if r.status == "pass")
    total = len(results)
    print(f"\nResults: {passed}/{total} tracks passed")

    if passed < total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())