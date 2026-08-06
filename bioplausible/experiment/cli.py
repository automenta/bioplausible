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

from bioplausible.experiment.probe import CoreTrainerDriver, config_key
from bioplausible.experiment.producer import (
    HyperoptGridProducer,
    OptunaBayesProducer,
    ProbeWork,
)
from bioplausible.experiment.report import Report
from bioplausible.experiment.reporting import render_report
from bioplausible.experiment.schema import load_campaign
from bioplausible.experiment.staircase import StaircaseRunner, Verdict
from bioplausible.hyperopt.eval_tiers import PatientLevel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from bioplausible.experiment.schema import Campaign, Stage

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
    p_plan.add_argument(
        "--producer",
        choices=("grid", "bayes"),
        default="grid",
        help="Config sampler: grid (exhaustive) or bayes (TPE, n_candidates)",
    )
    p_plan.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Bayes producer n_candidates (default 50)",
    )

    p_run = sub.add_parser("run", help="Run the campaign staircase (resume by default)")
    p_run.add_argument("config", help="Path to the campaign YAML")
    p_run.add_argument("--report", default=None, help="Report JSONL path")
    p_run.add_argument("--device", default=None, help="cpu / cuda:0 / auto")
    p_run.add_argument(
        "--producer",
        choices=("grid", "bayes"),
        default="grid",
        help="Config sampler: grid (exhaustive) or bayes (TPE, n_candidates)",
    )
    p_run.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Bayes producer n_candidates (default 50)",
    )
    return parser


def _report_path(campaign: Campaign, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(f"{campaign.meta.name}.report.jsonl")


def _all_models(campaign: Campaign) -> list[str]:
    return [m for arm in campaign.arms.values() for m in arm.models]


def _producer(
    campaign: Campaign, kind: str, candidates: int | None
) -> HyperoptGridProducer | OptunaBayesProducer:
    """Build the config producer from a ``--producer`` selection.

    ``grid`` enumerates the stage's full grid exactly; ``bayes`` draws
    ``candidates`` (default 50) TPE-sampled points from it, so ``plan``'s
    probe count tracks ``run`` under either choice.
    """
    seed = campaign.reproducibility.seed
    if kind == "bayes":
        return OptunaBayesProducer(
            n_candidates=candidates if candidates is not None else 50,
            seed=seed,
        )
    return HyperoptGridProducer(seed=seed)


def _in_budget_pairs(
    campaign: Campaign,
    stage: Stage,
    models: list[str],
    producer: HyperoptGridProducer | OptunaBayesProducer,
    param_counter: Callable[[str, dict[str, object], int, int], int] | None = None,
) -> list[ProbeWork]:
    """Enumerate (model, config) pairs within each model's arm ``max_params``.

    Uses the same schedule-time budget rule as the staircase
    (architecture §6.3): a config is dropped when its training-free parameter
    count exceeds the model's arm budget. Keeps ``plan``'s probe count exactly
    consistent with what ``run`` will actually train.
    """
    from bioplausible.experiment.param_estimator import estimate_param_count

    def _count(
        model: str,
        config: dict[str, object],
        dims: tuple[int, int],
    ) -> int:
        if param_counter is not None:
            return param_counter(model, config, dims[0], dims[1])
        return estimate_param_count(
            model, config, input_dim=dims[0], output_dim=dims[1]
        )

    geom = campaign.geometry(stage.task)
    configs = producer.configs_for(stage)
    budget_by_model = {m: campaign.max_params_for(m) for m in models}
    pairs: list[ProbeWork] = []
    for model in models:
        budget = budget_by_model[model]
        for config in configs:
            try:
                over = budget is not None and _count(model, config, geom) > budget
            except Exception as exc:  # broad: a non-constructible (model, config) is simply not schedulable
                logger.warning(
                    "skipping non-constructible pair %s/%s: %s",
                    model,
                    config,
                    exc,
                )
                continue
            if over:
                continue
            pairs.append(
                ProbeWork(model=model, config=config, config_key=config_key(config))
            )
    return pairs


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
    return estimate_total_time(
        _patience_for_epochs(deepest), n_models=len(_all_models(campaign))
    )


def _cmd_validate(config: str) -> int:
    campaign = load_campaign(config)
    arms = ", ".join(sorted(campaign.arms))
    stage_summary = ", ".join(f"{s.name}:{s.task}" for s in campaign.stages)
    print(
        f"OK: campaign {campaign.meta.name!r} valid\n"
        f"  arms: [{arms}]\n  stages: {stage_summary}"
    )
    return 0


def _cmd_plan(config: str, producer_kind: str, candidates: int | None) -> int:
    campaign = load_campaign(config)
    models = _all_models(campaign)
    producer = _producer(campaign, producer_kind, candidates)
    total = 0
    print(f"campaign: {campaign.meta.name!r}")
    for stage in campaign.stages:
        pairs = _in_budget_pairs(campaign, stage, models, producer)
        n_seeds = stage.seeds
        n_probes = len(pairs) * n_seeds
        total += n_probes
        print(
            f"  stage {stage.name:<16}{len(pairs)} in-budget (model, config) "
            f"pairs x {n_seeds} seeds = {n_probes} probes"
        )
    budget = _time_budget(campaign)
    print(f"total probes: {total}")
    print(
        f"estimated total time: {budget['estimated_completion']} "
        f"({budget['minutes']:.0f} min across {budget['trials_per_model']} "
        "trials_per_model)"
    )
    return 0


def _cmd_run(
    config: str,
    report_override: str | None,
    device: str | None,
    producer_kind: str,
    candidates: int | None,
) -> int:
    campaign = load_campaign(config)
    report_path = _report_path(campaign, report_override)
    if report_path.exists() and report_path.stat().st_size:
        print(f"resuming report {report_path} (finished probes are no-ops)")

    wants_energy = any(bool(stage.energy) for stage in campaign.stages)
    report = Report(report_path)
    track = campaign.compute.track
    runner = StaircaseRunner(
        campaign,
        report,
        CoreTrainerDriver(
            num_workers=campaign.compute.num_workers,
            track_energy=wants_energy,
            track_flops=track.flops,
            track_memory=track.memory,
        ),
        _producer(campaign, producer_kind, candidates),
        compute=campaign.compute,
    )
    if device and runner.compute is not None:
        runner.compute.device = device

    start = time.time()
    try:
        outcomes = runner.run()
    except KeyboardInterrupt:
        print(  # '--report' path is the resume contract; a partial run must be re-runnable
            f"interrupted: {report_path} holds {len(report.finished_keys())} "
            f"finished probes; rerun to resume"
        )
        return 130
    except Exception as exc:  # broad: a long overnight run must not lose the resume contract to a single probe/driver crash
        logger.error(
            "run aborted at %d finished probe(s): %s",
            len(report.finished_keys()),
            exc,
            exc_info=True,
        )
        print(
            f"run aborted: {report_path} holds {len(report.finished_keys())} "
            f"finished probes and is resumable (rerun to continue)"
        )
        return 1
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
                return _cmd_plan(args.config, args.producer, args.candidates)
            case "run":
                return _cmd_run(
                    args.config,
                    args.report,
                    args.device,
                    args.producer,
                    args.candidates,
                )
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
    parser.add_argument(
        "--baseline", default=None, help="Baseline model for effect sizes"
    )
    args = parser.parse_args(argv)
    try:
        print(render_report(args.report, baseline=args.baseline))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("report error: %s", exc)  # ruff: ignore[error-instead-of-exception]  # user-facing CLI: a traceback is noise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
