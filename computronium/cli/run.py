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
    list         List registered models.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from computronium.cli.commands.train import add_train_subparsers, run_training, run_core_train, run_from_yaml
from computronium.cli.commands.search import add_search_subparsers, run_search
from computronium.cli.commands.compare import add_compare_subparsers, run_compare
from computronium.cli.commands.verify import add_verify_subparsers, run_verify
from computronium.cli.benchmark import main as run_benchmark_cli
from computronium.core.registry import Registry, ComponentCategory


def add_pareto_subparsers(subparsers: argparse._SubParsersAction) -> None:
    pareto_parser = subparsers.add_parser(
        "pareto", help="Generate Pareto frontier plots/data for a study"
    )
    pareto_parser.add_argument("--study", required=True, help="Study name")
    pareto_parser.add_argument(
        "--output-dir", default="results/pareto", help="Output directory"
    )
    pareto_parser.add_argument(
        "--format",
        choices=["html", "png", "json"],
        default="html",
        help="Output format",
    )
    pareto_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the study (default: computronium.db)",
    )


def add_portfolio_subparsers(subparsers: argparse._SubParsersAction) -> None:
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


def add_benchmark_subparsers(subparsers: argparse._SubParsersAction) -> None:
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run cross-domain benchmark suite"
    )
    benchmark_parser.add_argument(
        "--models",
        help="Comma-separated model names (default: all registered)",
    )
    benchmark_parser.add_argument(
        "--domains",
        help="Comma-separated domains (default: all)",
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (3 epochs, smoke test)",
    )
    benchmark_parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Intermediate mode (10 epochs)",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )


def list_models(_args: argparse.Namespace) -> None:
    """List available models."""
    logger = logging.getLogger(__name__)
    logger.info("Available Models (Zoo Registry):")
    for entry in Registry.query(category=ComponentCategory.MODEL):
        bio = entry.get("bio_plausibility", 0)
        logger.info("  %-30s bio=%.1f", entry["name"], bio)


def run_pareto(args: argparse.Namespace) -> None:
    """Generate Pareto frontier plots/data for a study."""
    import optuna
    from computronium.cli.shared import _set_storage

    if args.db:
        _set_storage(args.db)

    from computronium.cli.shared import _STORAGE_URL
    study = optuna.load_study(study_name=args.study, storage=_STORAGE_URL)

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        logging.warning("No completed trials in study %s", args.study)
        return

    # Extract metrics
    data = []
    for t in trials:
        data.append({
            "number": t.number,
            "value": t.value,
            **t.params,
            **t.user_attrs,
        })

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        with open(Path(args.output_dir) / f"{args.study}_pareto.json", "w") as f:
            json.dump(data, f, indent=2)
    elif args.format == "html":
        # Minimal HTML output
        html = f"<html><body><h1>Pareto: {args.study}</h1><pre>{json.dumps(data, indent=2)}</pre></body></html>"
        with open(Path(args.output_dir) / f"{args.study}_pareto.html", "w") as f:
            f.write(html)
    logging.info("Pareto data written to %s", args.output_dir)


def run_portfolio(args: argparse.Namespace) -> None:
    """Build Phase 1 portfolio ranking table."""
    if args.db:
        from computronium.cli.shared import _set_storage
        _set_storage(args.db)

    from computronium.cli.shared import _STORAGE_URL
    from computronium.hyperopt.storage import list_studies

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


def run_benchmark(args: argparse.Namespace) -> None:
    """Run cross-domain benchmark suite - delegates to biopl benchmark."""
    # Delegate to the biopl benchmark CLI
    import sys
    sys.argv = ["biopl", "benchmark"] + sys.argv[2:]
    run_benchmark_cli()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bioplausible Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    add_train_subparsers(subparsers)
    add_search_subparsers(subparsers)
    add_compare_subparsers(subparsers)
    add_verify_subparsers(subparsers)
    add_pareto_subparsers(subparsers)
    add_portfolio_subparsers(subparsers)
    add_benchmark_subparsers(subparsers)
    subparsers.add_parser("list", help="List available models")

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
        "benchmark": run_benchmark,
        "list": list_models,
    }

    if args.command in command_map:
        command_map[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()