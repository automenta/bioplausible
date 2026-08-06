"""``biopl-run`` — the experiment-layer campaign CLI (architecture §10).

Subcommands:

* ``validate`` — parse and validate a campaign YAML (schema + task registry +
  evidence gates: ``seeds >= 10`` on ``baseline:`` stages, ``matched_by``,
  dual ``energy``).
* ``plan``     — dry-run: exact scheduled probe count (grid x models x seeds,
  assuming all survivors advance) plus a wall-time budget estimate via
  ``hyperopt.eval_tiers.estimate_total_time``.
* ``run``      — idempotent staircase execution (resume by default) into an
  append-only JSONL Report.

``main_report`` backs ``biopl-report``: render the experiment Report.

Register the entry point as ``biopl-run = bioplausible.experiment.cli:main`` and
``biopl-report = bioplausible.experiment.cli:main_report``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from bioplausible.experiment.probe import CoreTrainerDriver
from bioplausible.experiment.producer import HyperoptGridProducer, grid_cardinality
from bioplausible.experiment.report import Report
from bioplausible.experiment.reporting import render_report
from bioplausible.experiment.schema import load_campaign
from bioplausible.experiment.staircase import StaircaseRunner, Verdict
from bioplausible.hyperopt.eval_tiers import PatientLevel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioplausible.experiment.schema import Campaign

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biopl-run", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate a campaign YAML")
    p_validate.add_argument("config", help="Path to the campaign YAML")

    p_plan = sub.add_parser("plan", help="Print the exact probe plan (dry-run)")
    p_plan.add_argument("config", help="Path to the campaign YAML")

    p_run = sub.add_parser("run", help="Run the campaign staircase (resume by default)")
    p_run.add_argument("config", help="Path to the campaign YAML")
    p_run.add_argument("--report", default=None, help="Report JSONL path")
    p_run.add_argument("--device", default=None, help="cpu / cuda:0 / auto")
    return parser


def _report_path(campaign: Campaign, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(f"{campaign.meta.name}.report.jsonl")


def _all_models(campaign: Campaign) -> list[str]:
    return [m for arm in campaign.arms.values() for m in arm.models]


def _probe_count(campaign: Campaign) -> int:
    return sum(
        len(_all_models(campaign)) * grid_cardinality(stage.configs) * stage.seeds
        for stage in campaign.stages
    )


_SMOKE_MAX_EPOCHS = 3
_SHALLOW_MAX_EPOCHS = 15
_STANDARD_MAX_EPOCHS = 60


def _patience_for_epochs(epochs: int) -> PatientLevel:
    if epochs <= _SMOKE_MAX_EPOCHS:
        return PatientLevel.SMOKE
    if epochs <= _SHALLOW_MAX_EPOCHS:
        return PatientLevel.SHALLOW
    if epochs <= _STANDARD_MAX_EPOCHS:
        return PatientLevel.STANDARD
    return PatientLevel.DEEP


def _time_budget(campaign: Campaign) -> dict[str, object]:
    from bioplausible.hyperopt.eval_tiers import estimate_total_time

    deepest = max(stage.epochs for stage in campaign.stages)
    return estimate_total_time(_patience_for_epochs(deepest), n_models=len(_all_models(campaign)))


def _cmd_validate(config: str) -> int:
    campaign = load_campaign(config)
    arms = ", ".join(sorted(campaign.arms))
    stage_summary = ", ".join(f"{s.name}:{s.task}" for s in campaign.stages)
    print(
        f"OK: campaign {campaign.meta.name!r} valid\n"
        f"  arms: [{arms}]\n  stages: {stage_summary}"
    )
    return 0


def _cmd_plan(config: str) -> int:
    campaign = load_campaign(config)
    total = _probe_count(campaign)
    print(f"campaign: {campaign.meta.name!r}")
    for stage in campaign.stages:
        n_models = len(_all_models(campaign))
        n_configs = grid_cardinality(stage.configs)
        n_seeds = stage.seeds
        print(
            f"  stage {stage.name:<16}{n_models} models x {n_configs} configs "
            f"x {n_seeds} seeds = {n_models * n_configs * n_seeds} probes"
        )
    budget = _time_budget(campaign)
    print(f"total probes: {total}")
    print(
        f"estimated total time: {budget['estimated_completion']} "
        f"({budget['minutes']:.0f} min across {budget['trials_per_model']} "
        "trials_per_model)"
    )
    return 0


def _cmd_run(config: str, report_override: str | None, device: str | None) -> int:
    campaign = load_campaign(config)
    report_path = _report_path(campaign, report_override)
    if report_path.exists() and report_path.stat().st_size:
        print(f"resuming report {report_path} (finished probes are no-ops)")

    wants_energy = any(bool(stage.energy) for stage in campaign.stages)
    report = Report(report_path)
    runner = StaircaseRunner(
        campaign,
        report,
        CoreTrainerDriver(track_energy=wants_energy),
        HyperoptGridProducer(seed=campaign.reproducibility.seed),
        compute=campaign.compute,
    )
    if device and runner.compute is not None:
        runner.compute.device = device

    start = time.time()
    outcomes = runner.run()
    elapsed = time.time() - start

    for outcome in outcomes:
        verdict = outcome.verdict.value
        print(f"  [{verdict:>6}] {outcome.model:<24} {outcome.reason}")
    n_pass = sum(1 for o in outcomes if o.verdict is Verdict.PASS)
    print(f"report written: {report_path}")
    print(f"{n_pass}/{len(outcomes)} PASS; completed in {elapsed:.1f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``biopl-run``."""
    args = _build_parser().parse_args(argv)
    try:
        match args.command:
            case "validate":
                return _cmd_validate(args.config)
            case "plan":
                return _cmd_plan(args.config)
            case "run":
                return _cmd_run(args.config, args.report, args.device)
            case _:
                return 2
    except (yaml.YAMLError, ValueError, FileNotFoundError) as exc:
        logger.error("experiment error: %s", exc)  # ruff: ignore[error-instead-of-exception]  # user-facing CLI: a traceback is noise
        return 1


def main_report(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``biopl-report`` (experiment reporter)."""
    parser = argparse.ArgumentParser(
        prog="biopl-report",
        description="Render a parity/Pareto/failure report from an experiment Report.",
    )
    parser.add_argument("report", help="Path to the experiment Report JSONL")
    parser.add_argument("--baseline", default=None, help="Baseline model for effect sizes")
    args = parser.parse_args(argv)
    try:
        print(render_report(args.report, baseline=args.baseline))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("report error: %s", exc)  # ruff: ignore[error-instead-of-exception]  # user-facing CLI: a traceback is noise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
