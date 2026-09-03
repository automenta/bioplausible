"""
CLI Runner for Bioplausible Experiments - Entry Point

Commands:
    train        Run a single training session (``--config`` for YAML).
    core-train   Train via the new CoreTrainer API.
    from-config  Train from a YAML config file.
    search       Compute-matched HPO across a propagator family.
    compare      Rank families from completed HPO studies into a CSV.
    verify       Re-run the top-k configs of a study with n seeds.
    pareto       Emit Pareto frontier plots/data for a study.
    portfolio    Build Phase 1 portfolio ranking table (Scale/Hold/Eliminated).
    benchmark    Cross-domain benchmark suite.
"""

import argparse
import logging

from computronium.cli.benchmark import main as run_benchmark_cli
from computronium.cli.commands.compare import add_compare_subparsers, run_compare
from computronium.cli.commands.pareto import add_pareto_subparsers, run_pareto
from computronium.cli.commands.portfolio import add_portfolio_subparsers, run_portfolio
from computronium.cli.commands.search import add_search_subparsers, run_search
from computronium.cli.commands.train import (
    add_train_subparsers,
    run_core_train,
    run_from_yaml,
    run_training,
)
from computronium.cli.commands.verify import add_verify_subparsers, run_verify


def main() -> None:
    parser = argparse.ArgumentParser(description="Bioplausible Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    add_train_subparsers(subparsers)
    add_search_subparsers(subparsers)
    add_compare_subparsers(subparsers)
    add_verify_subparsers(subparsers)
    add_pareto_subparsers(subparsers)
    add_portfolio_subparsers(subparsers)
    # benchmark delegates to comp benchmark CLI
    subparsers.add_parser("benchmark", help="Run cross-domain benchmark suite")

    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if getattr(args, "db", None):
        from computronium.cli.shared import _set_storage

        _set_storage(args.db)

    command_map = {
        "train": run_training,
        "core-train": run_core_train,
        "from-config": run_from_yaml,
        "search": run_search,
        "compare": run_compare,
        "verify": run_verify,
        "pareto": run_pareto,
        "portfolio": run_portfolio,
        "benchmark": lambda args: run_benchmark_cli(),  # ruff: ignore[unused-lambda-argument]
    }

    if args.command in command_map:
        command_map[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
