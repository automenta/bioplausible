"""Verify commands for the CLI."""

import argparse
import json
from pathlib import Path

from computronium.cli.shared import logger

__all__ = ["add_verify_subparsers", "run_verify"]


def add_verify_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add verify subparser."""
    verify_parser = subparsers.add_parser(
        "verify", help="Re-run top-k configs of a study with n seeds"
    )
    verify_parser.add_argument("--study", required=True, help="Study name to verify")
    verify_parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top trials to re-run"
    )
    verify_parser.add_argument("--seeds", type=int, default=5, help="Seeds per config")
    verify_parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for verification runs"
    )
    verify_parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for verification"
    )
    verify_parser.add_argument(
        "--task", default="digits", help="Task name (fallback if not in study attrs)"
    )
    verify_parser.add_argument(
        "--output", type=str, help="JSONL output path for verified runs"
    )
    verify_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the study (default: computronium.db)",
    )


def run_verify(args: argparse.Namespace) -> None:
    """Re-run top-k configs of a study with n seeds."""
    import optuna
    from computronium.cli.shared import _DB_PATH, _STORAGE_URL, _set_storage

    if getattr(args, "db", None):
        _DB_PATH, _STORAGE_URL = _set_storage(args.db)

    study = optuna.load_study(study_name=args.study, storage=_STORAGE_URL)

    # Get top-k trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)
    top_trials = completed[: args.top_k]

    logger.info("Verifying top %d trials from %s", args.top_k, args.study)

    results = []
    for trial in top_trials:
        params = trial.params
        model_name = trial.user_attrs.get("model_name")
        task_name = trial.user_attrs.get("task", args.task)
        family = trial.user_attrs.get("family")

        logger.info("  Trial %d: value=%.4f, model=%s", trial.number, trial.value, model_name)

        for seed_offset in range(args.seeds):
            seed = args.seed + seed_offset * 1000 + trial.number

            # Run verification
            from computronium.hyperopt.experiment import run_single_trial
            from computronium.hyperopt.eval_tiers import get_evaluation_config, PatientLevel

            eval_cfg = get_evaluation_config(PatientLevel.STANDARD)
            if args.epochs:
                eval_cfg = eval_cfg.__class__(
                    n_trials=eval_cfg.n_trials,
                    n_epochs=args.epochs,
                    batch_size=eval_cfg.batch_size,
                    sampler_warmup=eval_cfg.sampler_warmup,
                    min_epochs=eval_cfg.min_epochs,
                )

            result = run_single_trial(
                model_name=model_name,
                task_name=task_name,
                config=params,
                eval_cfg=eval_cfg,
                device="auto",
                seed=seed,
            )

            results.append({
                "trial_number": trial.number,
                "seed": seed,
                "model": model_name,
                "task": task_name,
                "family": family,
                **result,
            })

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        logger.info("Verification results written to %s", args.output)