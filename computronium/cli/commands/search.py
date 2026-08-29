"""Search (HPO) commands for the CLI."""

import argparse
import json

from computronium.cli.shared import (
    FAMILY_MAP,
    _make_objective,
    _resolve_targets,
    _set_storage,
    _tier_for_args,
    _TrialContext,
    logger,
)

__all__ = ["add_search_subparsers", "run_search"]


def add_search_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add search subparser."""
    search_parser = subparsers.add_parser(
        "search", help="Compute-matched HPO across a propagator family"
    )
    search_group = search_parser.add_mutually_exclusive_group()
    search_group.add_argument(
        "--family",
        choices=[*list(FAMILY_MAP), "all", "survivors"],
        help="Propagator family to search (one study per family).",
    )
    search_group.add_argument(
        "--models",
        help="Comma-separated model names (legacy per-model path)",
    )
    search_parser.add_argument(
        "--survivors-csv",
        default="results/portfolio.csv",
        help="Portfolio CSV read by --family survivors (Phase 1.2 gate)",
    )
    search_parser.add_argument(
        "--task",
        default="digits",
        choices=["digits", "cifar10", "tiny_shakespeare", "mnist"],
        help="Task/dataset name",
    )
    search_parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Optuna trials per model (0 = use tier default)",
    )
    search_parser.add_argument(
        "--budget-tier",
        dest="tier",
        default="standard",
        choices=["smoke", "shallow", "standard", "deep"],
        help="Compute-matching tier (controls epochs, batch size, sampler warmup)",
    )
    search_parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Seeds for top-k verification (metadata only here; used by verify)",
    )
    search_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for the Optuna sampler + RNGs",
    )
    search_parser.add_argument(
        "--method",
        choices=["bayesian", "random"],
        default="bayesian",
        help="Optuna sampler: bayesian (TPE) or random",
    )
    search_parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    search_parser.add_argument(
        "--output", type=str, help="JSONL output path for trial records"
    )
    search_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file for this run (default: computronium.db)",
    )


def run_search(args: argparse.Namespace) -> None:
    """Compute-matched HPO across a propagator family."""
    from computronium.hyperopt import create_optuna_space, create_study
    from computronium.hyperopt.eval_tiers import get_evaluation_config

    # Override storage if --db provided
    global _DB_PATH, _STORAGE_URL
    if getattr(args, "db", None):
        _DB_PATH, _STORAGE_URL = _set_storage(args.db)
        logger.info("Using storage: %s", _STORAGE_URL)

    targets = _resolve_targets(args)
    if not targets:
        logger.warning("No targets resolved; nothing to do")
        return

    tier = _tier_for_args(args)
    eval_cfg = get_evaluation_config(tier)

    for study_name, reg_family, cli_family, models in targets:
        logger.info("Starting study: %s", study_name)
        study = create_study(study_name, storage=_STORAGE_URL)

        for model in models:
            logger.info("  Optimizing %s", model)
            ctx = _TrialContext(
                model=model,
                family=reg_family,
                task=args.task,
                eval_cfg=eval_cfg,
                quick_mode=(tier == "smoke"),
                device=args.device,
                tier_name=tier.value,
            )

            space = create_optuna_space(model, args.task)
            objective = _make_objective(ctx, search_space=space)

            n_trials = args.budget or eval_cfg.n_trials
            study.optimize(objective, n_trials=n_trials, timeout=None)

            # Save best trial
            best = study.best_trial
            logger.info("  Best trial: %s (value=%.4f)", best.number, best.value)

        if args.output:
            # Export all trials as JSONL
            trials = study.trials
            with open(args.output, "w") as f:
                for t in trials:
                    f.write(
                        json.dumps({
                            "number": t.number,
                            "value": t.value,
                            "params": t.params,
                            "user_attrs": t.user_attrs,
                        })
                        + "\n"
                    )
            logger.info("Exported %d trials to %s", len(trials), args.output)
