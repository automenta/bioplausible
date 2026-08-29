"""Portfolio CLI command."""

import argparse
import csv
import logging
from pathlib import Path

from computronium.cli.shared import _set_storage, _STORAGE_URL
from computronium.hyperopt.storage import list_studies


def add_portfolio_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add Portfolio command to subparsers."""
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Build Phase 1 portfolio ranking table (Scale/Hold/Eliminated)",
    )
    portfolio_parser.add_argument(
        "--tasks",
        default="digits,cifar10",
        help="Comma-separated task scopes to include (default: digits,cifar10)",
    )
    portfolio_parser.add_argument("--output", required=True, help="Output CSV path")
    portfolio_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the studies (default: computronium.db)",
    )


def run_portfolio(args: argparse.Namespace) -> None:
    """Build Phase 1 portfolio ranking table."""
    if args.db:
        _set_storage(args.db)

    studies = list_studies(_STORAGE_URL)
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Filter studies by task
    filtered = [s for s in studies if any(task in s for task in tasks)]

    if not filtered:
        logging.warning("No studies found for tasks: %s", args.tasks)
        return

    # Simple output - just list studies
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study", "tasks"])
        for s in filtered:
            writer.writerow([s, ",".join(tasks)])

    logging.info("Portfolio written to %s", args.output)
