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


def _parse_time_budget(text: str) -> int:
    """Parse a time budget string like '1h', '30m', '3600s' into seconds."""
    text = text.strip().lower()
    if text.endswith("h"):
        return int(float(text[:-1]) * 3600)
    if text.endswith("m"):
        return int(float(text[:-1]) * 60)
    if text.endswith("s"):
        return int(text[:-1])
    # Assume seconds if no suffix
    return int(text)


def _resolve_device(campaign: Campaign, override: str | None) -> str:
    """Resolve the device string, handling 'auto'."""
    if override:
        return override
    device = campaign.compute.device
    if device == "auto":
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _rep_models(models: list[str]) -> list[str]:
    """Pick a small representative subset for calibration.

    Always the backprop baseline plus one equilibrium/FA/hebbian model (or any
    second model if none of those exist), to keep calibration cheap without
    pretending all models cost the same.
    """
    reps: list[str] = []
    for m in models:
        if "backprop" in m:
            reps.append(m)
            break
    for m in models:
        if m not in reps and any(k in m for k in ("eqprop", "pepita", "fa", "hebbian")):
            reps.append(m)
            break
    if len(reps) < 2:  # ruff: ignore[magic-value-comparison]  # reps is a fixed 2-model calibration subset
        for m in models:
            if m not in reps:
                reps.append(m)
                break
    return reps


def _calib_batches(target_seconds: float | None) -> int:
    """Calibration batch count, scaled with the budget (proportional planning).

    Roughly 1% of the budget in batches, clamped between a 10-batch noise floor
    and the default 100-batch epoch. The floor matters: extrapolating fewer
    batches amplifies per-probe setup noise into a garbage per-epoch figure.
    """
    epoch_batches = 100
    if target_seconds is None:
        n = 10
    else:
        n = int(epoch_batches * min(1.0, target_seconds / 3600))
    return max(10, min(epoch_batches, n))


def _task_cost_factor(task: str, bottleneck_task: str) -> float:
    """Relative per-epoch cost of ``task`` vs the calibrated bottleneck task.

    xor is far cheaper than cifar10 on a GPU; unknown pairs are treated as
    equal to the bottleneck (factor 1.0).
    """
    if task == "xor" and bottleneck_task == "cifar10":
        return 1 / 50.0
    if task == "cifar10" and bottleneck_task == "xor":
        return 50.0
    return 1.0


def _measure_epoch(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]  # passes the full probe identity + calibration knobs
    driver: CoreTrainerDriver,
    model: str,
    task: str,
    config: dict[str, object],
    device: str,
    calib_batches: int,
    epoch_batches: int,
) -> float:
    """Run one 1-epoch calibration probe and extrapolate to a full epoch.

    Returns seconds per full (epoch_batches-batch) epoch, or 0.0 on failure.
    """
    from bioplausible.experiment.probe import run_probe

    try:
        result = run_probe(
            driver,
            model=model,
            task=task,
            config=config,
            seed=0,
            epochs=1,
            device=device,
            param_count=0,
        )
    except Exception as exc:  # broad: a broken calibration probe is a 0.0, not an abort
        print(f"    {model}: ERROR {exc}")
        return 0.0
    if result.status != "ok":
        print(f"    {model}: FAILED ({result.error})")
        return 0.0
    per_batch = max(result.epoch_time_s, 0.0) / calib_batches
    return per_batch * epoch_batches


def _calibrate_epoch_times(
    campaign: Campaign,
    device: str,
    models: list[str],
    producer: HyperoptGridProducer | OptunaBayesProducer,
    target_seconds: float | None = None,
) -> dict[tuple[str, str], float]:
    """
    Run quick calibration probes on the target device.

    Calibrates only the bottleneck task (most epochs) using a tiny config and
    2 representative models. The number of calibration batches scales with the
    time budget (proportional planning) so planning stays cheap for short runs
    and accurate for long ones. Returns (model, task) -> seconds per full
    (100-batch) epoch.
    """
    from bioplausible.experiment.probe import CoreTrainerDriver

    # Pick the bottleneck task: the stage with the most epochs (usually the
    # evidence/parity stage), since that dominates the schedule.
    bottleneck_stage = max(campaign.stages, key=lambda s: s.epochs)
    bottleneck_task = bottleneck_stage.task

    configs = producer.configs_for(bottleneck_stage) or [{}]

    def config_size(cfg: dict[str, object]) -> int:
        h = int(cfg.get("hidden_dim", 64))
        layers = int(cfg.get("num_layers", 1))
        return h * layers

    rep_config = min(configs, key=config_size)
    reps = _rep_models(models)
    calib_batches = _calib_batches(target_seconds)

    driver = CoreTrainerDriver(
        num_workers=campaign.compute.num_workers,
        track_energy=False,
        track_flops=False,
        track_memory=False,
        batches_per_epoch=calib_batches,
    )

    print(
        f"  Calibrating bottleneck task '{bottleneck_task}' on {device} "
        f"with config: {rep_config} ({calib_batches} batches)"
    )
    epoch_batches = 100
    epoch_times: dict[tuple[str, str], float] = {}

    for model in reps:
        epoch_times[model, bottleneck_task] = _measure_epoch(
            driver,
            model,
            bottleneck_task,
            rep_config,
            device,
            calib_batches,
            epoch_batches,
        )
        reported = epoch_times[model, bottleneck_task]
        print(f"    {model}: {reported:.1f}s/epoch (est)")

    # Erred/failed reps get the other rep's time (or a defensive 30s fallback).
    for model in reps:
        if epoch_times.get((model, bottleneck_task), 0.0) <= 0:
            alt = next(
                (
                    epoch_times[m, bottleneck_task]
                    for m in reps
                    if m != model and epoch_times.get((m, bottleneck_task), 0) > 0
                ),
                30.0,
            )
            epoch_times[model, bottleneck_task] = alt

    # Other tasks: scale relative to the bottleneck task's dataset cost ratio.
    max_ref = max(
        (epoch_times.get((m, bottleneck_task), 0.0) for m in reps), default=30.0
    )
    for stage in campaign.stages:
        if stage.task == bottleneck_task:
            continue
        factor = _task_cost_factor(stage.task, bottleneck_task)
        for model in models:
            epoch_times[model, stage.task] = max_ref * factor

    # Populate the bottleneck task for every model not directly calibrated.
    measured = [epoch_times.get((m, bottleneck_task), 0.0) for m in reps]
    avg = sum(measured) / len(measured) if measured else 30.0
    for model in models:
        if (model, bottleneck_task) not in epoch_times:
            epoch_times[model, bottleneck_task] = avg

    return epoch_times


def _estimate_total_wall_time(
    campaign: Campaign,
    epoch_times: dict[tuple[str, str], float],
    models: list[str],
    producer: HyperoptGridProducer | OptunaBayesProducer,
) -> float:
    """Estimate total wall time for the full campaign based on per-model per-task epoch times."""
    total = 0.0
    for stage in campaign.stages:
        # Reuse the same in-budget filter as `plan` so the estimate matches the
        # real schedule (over-budget pairs are excluded once, not per iteration).
        pairs = _in_budget_pairs(campaign, stage, models, producer)
        n_pairs = len(pairs)
        n_seeds = stage.seeds
        # Use the max per-epoch time across models in this stage as the stage's
        # per-epoch cost (a conservative upper bound for the whole stage).
        per_epoch = 0.0
        for model in models:
            et = epoch_times.get((model, stage.task), 0.0)
            per_epoch = max(per_epoch, et)
        if per_epoch <= 0:
            per_epoch = 30.0
        # Per probe: epochs * per_epoch (train) + fixed setup overhead (~8s).
        probe_time = stage.epochs * per_epoch + 8.0
        total += probe_time * n_pairs * n_seeds
    return total


def _config_size(cfg: dict[str, object]) -> int:
    """Cheap proxy for a config's training cost: hidden_dim * num_layers."""
    h = int(cfg.get("hidden_dim", 64))
    layers = int(cfg.get("num_layers", 1))
    return h * layers


def _min_seeds_for(stage: Stage) -> int:
    """Evidence (baseline) stages must keep 10 seeds; others can go to 1."""
    return 10 if stage.baseline is not None else 1


def _reduce_stage_epochs(stage: Stage, scale_factor: float) -> bool:
    """Clamp a stage's epochs toward the budget; returns True if changed."""
    new_epochs = max(1, int(stage.epochs * scale_factor))
    if new_epochs >= stage.epochs:
        return False
    print(f"    Stage {stage.name}: epochs {stage.epochs} -> {new_epochs}")
    stage.epochs = new_epochs
    return True


def _reduce_stage_configs(
    stage: Stage,
    scale_factor: float,
    producer: HyperoptGridProducer | OptunaBayesProducer,
) -> bool:
    """Keep the smallest-N configs of a stage; returns True if any were dropped."""
    configs = producer.configs_for(stage)
    if len(configs) <= 1:
        return False
    n_keep = max(1, int(len(configs) * scale_factor))
    if n_keep >= len(configs):
        return False
    kept = sorted(configs, key=_config_size)[:n_keep]
    print(f"    Stage {stage.name}: configs {len(configs)} -> {n_keep}")
    new_configs: dict[str, list[object]] = {}
    for key in stage.configs:
        values = {cfg[key] for cfg in kept if key in cfg}
        if values:
            new_configs[key] = sorted(values)
    stage.configs = new_configs
    return True


def _reduce_stage_seeds(stage: Stage, scale_factor: float) -> bool:
    """Clamp a stage's seeds toward the budget (never below its minimum)."""
    new_seeds = max(_min_seeds_for(stage), int(stage.seeds * scale_factor))
    if new_seeds >= stage.seeds:
        return False
    print(f"    Stage {stage.name}: seeds {stage.seeds} -> {new_seeds}")
    stage.seeds = new_seeds
    return True


def _auto_scale_campaign(
    campaign: Campaign,
    target_seconds: float,
    epoch_times: dict[tuple[str, str], float],
    models: list[str],
    producer: HyperoptGridProducer | OptunaBayesProducer,
) -> Campaign:
    """
    Return a deep copy of the campaign with epochs/configs/seeds scaled to fit the time budget.

    Scaling priority:
    1. Reduce epochs (minimum 1)
    2. Reduce configs via bayes producer (if grid)
    3. Reduce seeds (respecting minimums: 1 for smoke, 10 for evidence)
    """
    import copy

    scaled = copy.deepcopy(campaign)

    current = _estimate_total_wall_time(scaled, epoch_times, models, producer)
    print(
        f"  Current estimate: {current / 60:.1f}min, "
        f"target: {target_seconds / 60:.1f}min"
    )
    if current <= target_seconds:
        return scaled  # already fits

    # Iteratively reduce epochs -> configs -> seeds until the estimate fits the
    # budget or every resource is at its floor. A single pass is not enough:
    # reducing epochs changes the cost linearly, and re-solving after each step
    # converges on the floor instead of overshooting the budget.
    for _ in range(64):  # bounded: each reduction strictly decreases cost
        current = _estimate_total_wall_time(scaled, epoch_times, models, producer)
        if current <= target_seconds:
            return scaled
        scale_factor = target_seconds / current
        for stage in scaled.stages:
            _reduce_stage_epochs(stage, scale_factor)
        if (
            _estimate_total_wall_time(scaled, epoch_times, models, producer)
            <= target_seconds
        ):
            return scaled
        for stage in scaled.stages:
            _reduce_stage_configs(stage, scale_factor, producer)
        if (
            _estimate_total_wall_time(scaled, epoch_times, models, producer)
            <= target_seconds
        ):
            return scaled
        for stage in scaled.stages:
            _reduce_stage_seeds(stage, scale_factor)

    current = _estimate_total_wall_time(scaled, epoch_times, models, producer)
    if current > target_seconds:
        print(
            f"  WARNING: cannot fit {target_seconds / 60:.1f}min with these "
            f"minimums; estimate {current / 60:.1f}min"
        )
    return scaled


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
    p_plan.add_argument(
        "--time-budget",
        type=str,
        default=None,
        help="Target wall-time budget (e.g. '1h', '30m', '3600s'); runs calibration probes on the target device and auto-scales epochs/configs to fit",
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
    p_run.add_argument(
        "--time-budget",
        type=str,
        default=None,
        help="Target wall-time budget (e.g. '1h', '30m', '3600s'); runs calibration probes on the target device and auto-scales epochs/configs to fit",
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


def _cmd_plan(
    config: str,
    producer_kind: str,
    candidates: int | None,
    time_budget: str | None,
) -> int:
    campaign = load_campaign(config)
    models = _all_models(campaign)
    producer = _producer(campaign, producer_kind, candidates)

    # If time budget specified, calibrate and auto-scale
    if time_budget:
        target_seconds = _parse_time_budget(time_budget)
        device = _resolve_device(campaign, None)
        print(f"Calibrating on {device} to fit {time_budget} budget...")
        epoch_times = _calibrate_epoch_times(
            campaign, device, models, producer, target_seconds
        )
        campaign = _auto_scale_campaign(
            campaign, target_seconds, epoch_times, models, producer
        )
        producer = _producer(campaign, producer_kind, candidates)
        print("Scaled campaign:")

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
    print(f"total probes: {total}")

    if time_budget:
        # Report the device-calibrated estimate (matches run's actual cost),
        # not the heuristic tier table.
        est = _estimate_total_wall_time(campaign, epoch_times, models, producer)
        print(
            f"calibrated total time: {est:.0f}s "
            f"({est / 60:.1f} min) vs {time_budget} target"
        )
    else:
        budget = _time_budget(campaign)
        print(
            f"estimated total time: {budget['estimated_completion']} "
            f"({budget['minutes']:.0f} min across {budget['trials_per_model']} "
            "trials_per_model)"
        )
    return 0


def _cmd_run(  # ruff: ignore[too-many-arguments,too-many-locals,too-many-positional-arguments]  # run threads every operator knob; grouped by the parser, not the signature
    config: str,
    report_override: str | None,
    device: str | None,
    producer_kind: str,
    candidates: int | None,
    time_budget: str | None,
) -> int:
    campaign = load_campaign(config)

    # If time budget specified, calibrate and auto-scale BEFORE creating runner
    if time_budget:
        target_seconds = _parse_time_budget(time_budget)
        resolved_device = _resolve_device(campaign, device)
        models = _all_models(campaign)
        producer = _producer(campaign, producer_kind, candidates)
        print(f"Calibrating on {resolved_device} to fit {time_budget} budget...")
        epoch_times = _calibrate_epoch_times(
            campaign, resolved_device, models, producer, target_seconds
        )
        campaign = _auto_scale_campaign(
            campaign, target_seconds, epoch_times, models, producer
        )
        print("Scaled campaign will be executed.")

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
                return _cmd_plan(
                    args.config, args.producer, args.candidates, args.time_budget
                )
            case "run":
                return _cmd_run(
                    args.config,
                    args.report,
                    args.device,
                    args.producer,
                    args.candidates,
                    args.time_budget,
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
