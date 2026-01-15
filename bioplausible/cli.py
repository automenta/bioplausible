#!/usr/bin/env python3
"""
TorEqProp Comprehensive Verification Suite (CLI)

Entry point for the installed `eqprop-verify` command.
"""

import argparse
from bioplausible.validation import Verifier


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Comprehensive Verification Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick mode (~2 min, smoke test)"
    )
    parser.add_argument(
        "--intermediate",
        "-i",
        action="store_true",
        help="Intermediate mode (~1 hr, directional validation)",
    )
    parser.add_argument(
        "--track", "-t", type=int, nargs="+", help="Run specific track(s)"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List all tracks")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds for robustness checks (redundancy)",
    )
    parser.add_argument("--export", action="store_true", help="Export raw data to CSV")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save verification results (defaults to ./results)",
    )

    args = parser.parse_args()

    verifier = Verifier(
        quick_mode=args.quick,
        intermediate_mode=args.intermediate,
        seed=args.seed,
        n_seeds_override=args.seeds,
        export_data=args.export,
        output_dir=args.output_dir,
    )

    if args.list:
        verifier.list_tracks()
    else:
        verifier.run_tracks(args.track)


if __name__ == "__main__":
    main()
