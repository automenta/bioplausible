"""
CLI Leaderboard Viewer
"""

import argparse

from tabulate import tabulate  # Assuming installed, or use simple formatter

from computronium.analysis.results import get_rankings, load_trials
from computronium.core.logging import get_logger

__all__ = [
    "logger",
    "main",
    "view_rankings",
]
logger = get_logger()


def view_rankings(args):
    db_path = args.db
    logger.info("[DATA]  Loading rankings from %s...", db_path)

    trials = load_trials(db_path)
    if not trials:
        logger.warning("No trials found.")
        return

    rankings = get_rankings(trials)

    # Prepare table
    data = []
    headers = ["Rank", "Family", "Best Acc", "Gap", "Trials"]

    for r in rankings:
        if args.family and args.family.lower() not in r.family.lower():
            continue

        gap_str = f"{r.gap_to_baseline:+.1f}%" if r.gap_to_baseline != 0 else "Base"
        data.append([
            f"#{r.rank}",
            r.family,
            f"{r.best_value * 100:.2f}%",
            gap_str,
            r.n_trials,
        ])

    logger.info("\n" + tabulate(data, headers=headers, tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Leaderboard Viewer")
    parser.add_argument("--db", default="shallow_benchmark.db", help="Path to database")
    parser.add_argument("--family", help="Filter by algorithm family")

    args = parser.parse_args()
    view_rankings(args)


if __name__ == "__main__":
    main()
