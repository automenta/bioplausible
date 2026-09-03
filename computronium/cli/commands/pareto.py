"""Pareto frontier CLI command."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.cli.shared import _STORAGE_URL, _set_storage

if TYPE_CHECKING:
    import argparse


def add_pareto_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add Pareto command to subparsers."""
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


def run_pareto(args: argparse.Namespace) -> None:
    """Generate Pareto frontier plots/data for a study."""
    import optuna

    if args.db:
        _set_storage(args.db)

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
        with Path(Path(args.output_dir) / f"{args.study}_pareto.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2)
    elif args.format == "html":
        # Minimal HTML output
        html = f"<html><body><h1>Pareto: {args.study}</h1><pre>{json.dumps(data, indent=2)}</pre></body></html>"
        with Path(Path(args.output_dir) / f"{args.study}_pareto.html").open(
            "w", encoding="utf-8"
        ) as f:
            f.write(html)
    logging.info("Pareto data written to %s", args.output_dir)
