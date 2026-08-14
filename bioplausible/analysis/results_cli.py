"""CLI for experiment result analysis.

Usage:
    python -m bioplausible.analysis.results_cli rank --db path/to/study.db
"""

import argparse

from bioplausible.analysis.results import get_rankings, load_trials, print_rankings
from bioplausible.core._paths import db_path

__all__ = [
    "main",
]


def _filter_trials(trials, tier=None, task=None):
    """Filter trials by tier and/or task."""
    filtered = []
    for t in trials:
        if tier:
            trial_tier = t.get("user_attrs", {}).get("tier")
            if not trial_tier:
                parts = t["study_name"].split("_")
                trial_tier = parts[-1] if parts else None
            if trial_tier != tier:
                continue

        if task and task not in t["study_name"]:
            continue

        filtered.append(t)
    return filtered


def main():
    """Run the CLI."""
    parser = argparse.ArgumentParser(description="Bioplausible Experiment Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    rank_parser = subparsers.add_parser("rank", help="Show algorithm rankings")
    rank_parser.add_argument(
        "--db", default=db_path("bioplausible.db"), help="Path to database"
    )
    rank_parser.add_argument("--tier", help="Filter by patience tier")
    rank_parser.add_argument("--task", help="Filter by task (e.g. mnist, lm)")

    args = parser.parse_args()

    if args.command == "rank":
        trials = load_trials(args.db)
        if args.tier or args.task:
            trials = _filter_trials(trials, tier=args.tier, task=args.task)
        rankings = get_rankings(trials)
        print_rankings(rankings)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
