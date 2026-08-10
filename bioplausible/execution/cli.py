"""
CLI Entry Points for AutoScientist.

This module provides the command-line interface for the AutoScientist system.
It supports two main modes:
1. `run`: Start the autonomous discovery agent.
2. `report`: Generate scientific reports from the database.
"""

import argparse

from bioplausible.analysis.reporting import ReportOrchestrator
from bioplausible.core.logging import get_logger
from bioplausible.execution.engine import ExecutionEngine

__all__ = [
    "logger",
    "main",
    "main_reporter",
    "main_scientist",
]
logger = get_logger()


def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="AutoScientist CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Subcommand: run (Default)
    run_parser = subparsers.add_parser("run", help="Start the autonomous scientist")
    run_parser.add_argument(
        "--task", type=str, default=None, help="Filter tasks (e.g. 'vision', 'mnist')"
    )
    run_parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Max trials (not strictly enforced yet)",
    )
    run_parser.add_argument(
        "--tier-limit",
        type=str,
        default=None,
        help="Limit maximum tier (smoke, shallow, standard, deep)",
    )
    run_parser.add_argument("--db", default="bioplausible.db", help="Path to database")
    run_parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )

    # Subcommand: report
    report_parser = subparsers.add_parser("report", help="Generate scientific report")
    report_parser.add_argument(
        "--db", default="bioplausible.db", help="Path to database"
    )
    report_parser.add_argument(
        "--out", default="reports", help="Output directory for reports"
    )

    args = parser.parse_args()

    if args.command == "report":
        _run_reporter(args)
    else:
        _run_scientist(args)


def _run_scientist(args: argparse.Namespace) -> None:
    """Execute the scientist runner."""
    logger.info(
        "Initializing AutoScientist (Task Filter: %s, Workers: %d)...",
        args.task,
        args.workers,
    )
    engine = ExecutionEngine(
        db_path=args.db,
        task_filter=args.task,
        tier_limit=args.tier_limit,
        num_workers=args.workers,
    )
    engine.run()


def _run_reporter(args: argparse.Namespace) -> None:
    """Execute the report generator."""
    logger.info("Generating report from %s to %s...", args.db, args.out)
    orchestrator = ReportOrchestrator(args.db, args.out)
    orchestrator.generate_reports()
    logger.info("Done.")


def main_scientist() -> None:
    """Entry point for ``biopl-scientist``: run the auto-scientist without arg parsing.

    Mirrors the default branch of :func:`main` (``run`` subcommand) so a
    bare ``biopl-scientist`` invocation starts the autonomous discovery loop.
    """
    engine = ExecutionEngine()
    engine.run()


def main_reporter() -> None:
    """Entry point for ``biopl-report``: generate reports without arg parsing."""
    orchestrator = ReportOrchestrator("bioplausible.db", "reports")
    orchestrator.generate_reports()


if __name__ == "__main__":
    main()
