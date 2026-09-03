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

Register the entry point as ``biopl-run = computronium.experiment.cli:main`` and
``biopl-report = computronium.experiment.cli:main_report``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from computronium.core.logging import get_logger
from computronium.core.utils.device import get_device
from computronium.experiment.probe import CoreTrainerDriver, config_key
from computronium.experiment.producer import (
    HyperoptGridProducer,
    OptunaBayesProducer,
    ProbeWork,
)
from computronium.experiment.report import Report
from computronium.experiment.reporting import render_report
from computronium.experiment.schema import load_campaign
from computronium.experiment.staircase import StaircaseRunner, Verdict
from computronium.hyperopt.eval_tiers import PatientLevel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from computronium.experiment.schema import Campaign, Stage

logger = get_logger()


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
        return str(get_device("cuda:0"))
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
    if len(reps) < 2:  # reps is a fixed 2-model calibration subset
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


_CALIB_EPOCHS = 3  # amortize the first-epoch warmup (kernel compile / allocator)


def _measure_epoch(  # passes the full probe identity + calibration knobs
    driver: CoreTrainerDriver,
    model: str,
    task: str,
    config: dict[str, object],
    device: str,
    calib_batches: int,
    epoch_batches: int,
) -> float:
    """Measure seconds per full epoch via a warm-up-amortized calibration probe.

    Runs ``_CALIB_EPOCHS`` short epochs and divides by the count, so the
    first-epoch warmup (CUDA kernel compile, allocator) does not inflate the
    per-epoch cost. Returns seconds per full (``epoch_batches``-batch) epoch,
    or 0.0 on failure.
    """
    from computronium.experiment.probe import run_probe

    try:
        result = run_probe(
            driver,
            model=model,
            task=task,
            config=config,
            seed=0,
            epochs=_CALIB_EPOCHS,
            device=device,
            param_count=0,
        )
    except Exception as exc:  # broad: a broken calibration probe is a 0.0, not an abort
        logger.error("Model %s: ERROR %s", model, exc)  # ruff: ignore[error-instead-of-exception]
        return 0.0
    if result.status != "ok":
        logger.error(t"    {model}: FAILED ({result.error})")
        return 0.0
    per_batch = max(result.epoch_time_s, 0.0) / (_CALIB_EPOCHS * calib_batches)
    return per_batch * epoch_batches


def _calibrate_epoch_times(
    campaign: Campaign,
    device: str,
    models: list[str],
    producer: HyperoptGridProducer | OptunaBayesProducer,
    target_seconds: float | None = None,
) -> dict[tuple[str, str], float]:
    """
    Calibrate seconds per full epoch for every (model, task) in the campaign.

    Measures each distinct task independently (not just the costliest) using a
    tiny config and the representative models, so cheap gating tasks (e.g. xor)
    get a real, much-smaller per-epoch figure rather than being assumed equal to
    the bottleneck. The number of calibration batches scales with the budget
    (proportional planning); per-epoch time is warm-up amortized. Returns
    ``(model, task) -> seconds per full (100-batch) epoch``.
    """
    from computronium.experiment.probe import CoreTrainerDriver

    tasks = sorted({s.task for s in campaign.stages})
    reps = _rep_models(models)
    calib_batches = _calib_batches(target_seconds)
    epoch_batches = 100
    driver = CoreTrainerDriver(
        num_workers=campaign.compute.num_workers,
        track_energy=False,
        track_flops=False,
        track_memory=False,
        batches_per_epoch=calib_batches,
    )

    epoch_times: dict[tuple[str, str], float] = {}
    for task in tasks:
        stage = next(s for s in campaign.stages if s.task == task)
        rep_config = min(producer.configs_for(stage) or [{}], key=_config_size)
        logger.info(
            t"Calibrating task '{task}' on {device} with config: {rep_config} ({calib_batches} batches)"
        )
        task_times: dict[str, float] = {}
        for model in reps:
            task_times[model] = _measure_epoch(
                driver, model, task, rep_config, device, calib_batches, epoch_batches
            )
            logger.info(t"    {model}: {task_times[model]:.1f}s/epoch (est)")

        valid = [t for t in task_times.values() if t > 0]
        avg = sum(valid) / len(valid) if valid else 30.0  # defensive floor
        for model in reps:
            if task_times.get(model, 0.0) <= 0:
                task_times[model] = avg
        for model in models:
            epoch_times[model, task] = task_times.get(model, avg)

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
        n_seeds = stage.seeds
        # Sum per (model, config) using that model's own calibrated per-epoch
        # time — fast models (e.g. a 0.07s/epoch hebbian) must not be charged at
        # the slowest model's cost. Per probe: epochs * per_epoch plus a
        # setup-overhead term bounded to the epoch cost (a flat 8s dominated
        # subsecond GPU probes and made the auto-scaler degenerate).
        for pair in pairs:
            per_epoch = epoch_times.get((pair.model, stage.task), 0.0)
            if per_epoch <= 0:  # uncalibrated pair: defensive upper bound
                per_epoch = 30.0
            setup = min(6.0, max(0.5, per_epoch))
            total += (stage.epochs * per_epoch + setup) * n_seeds
    return total


def _config_size(cfg: dict[str, object]) -> int:
    """Cheap proxy for a config's training cost: hidden_dim * num_layers."""
    h = int(cfg.get("hidden_dim", 64))
    layers = int(cfg.get("num_layers", 1))
    return h * layers


def _min_seeds_for(stage: Stage) -> int:
    """Evidence (baseline) stages must keep 10 seeds; others can go to 1."""
    return 10 if stage.baseline is not None else 1


# Evidence (parity) stages must keep enough epochs that accuracy is meaningful
# (learned) rather than chance-level noise, so the report's parity/effect-size
# table is analyzable. Below this floor a probe is under-trained and the parity
# comparison is meaningless (the smoke run at 1 epoch showed acc ~ chance).
_MIN_EVIDENCE_EPOCHS = 5


def _reduce_stage_epochs(
    stage: Stage, scale_factor: float, min_epochs: int = 1
) -> bool:
    """Clamp a stage's epochs toward the budget; returns True if changed."""
    new_epochs = max(min_epochs, int(stage.epochs * scale_factor))
    if new_epochs >= stage.epochs:
        return False
    logger.info(t"    Stage {stage.name}: epochs {stage.epochs} -> {new_epochs}")
    stage.epochs = new_epochs
    return True


def _reduce_epochs_keeping_gates(stages: Sequence[Stage], scale_factor: float) -> None:
    """Reduce epochs on evidence stages only; gating stages keep theirs.

    A gating (non-baseline) stage whose pass-rule threshold becomes
    unreachable under aggressive epoch cuts (e.g. xor ``acc >= 0.60`` in 1
    epoch) would reject the whole model set and produce a degenerate
    single-survivor report, so its epochs are never scaled down.
    """
    for stage in stages:
        if stage.baseline is None:
            continue
        _reduce_stage_epochs(stage, scale_factor, _MIN_EVIDENCE_EPOCHS)


def _grid_len(grid: dict[str, list[object]]) -> int:
    """Cardinality of a parallel value grid (product of per-key sizes)."""
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


def _reduce_stage_configs(stage: Stage, scale_factor: float) -> bool:
    """Drop costliest config choices until the stage grid fits ``scale_factor``.

    A parallel grid can't be pruned to an arbitrary config subset (the smallest
    N configs of a 3x3 grid still span every value in every key, so re-expanding
    reconstructs the full grid). Instead, greedily remove the highest-cost value
    choice from the key whose removal shrinks the grid the most, so big configs
    are dropped first (cheapest and fewest remain). Returns True if changed.
    """
    from itertools import product

    def expand(g: dict[str, list[object]]) -> list[dict[str, object]]:
        keys = list(g)
        if not keys:
            return []
        return [dict(zip(keys, combo)) for combo in product(*(g[k] for k in keys))]

    grid: dict[str, list[object]] = {k: list(v) for k, v in stage.configs.items()}
    target = max(1, int(_grid_len(grid) * scale_factor))
    if target >= _grid_len(grid):
        return False

    while _grid_len(grid) > target:
        best: tuple[int, str, object] | None = None  # (shrink, key, value)
        for key, values in grid.items():
            if len(values) <= 1:
                continue
            drop = max(values, key=lambda v: _config_size({key: v}))
            candidate = {**grid, key: [v for v in values if v != drop]}
            shrink = _grid_len(grid) - _grid_len(candidate)
            if best is None or shrink > best[0]:
                best = (shrink, key, drop)
        if best is None:
            break
        _, key, drop = best
        grid[key] = [v for v in grid[key] if v != drop]

    logger.info(t"    Stage {stage.name}: configs reduced")
    stage.configs = grid
    return True


def _reduce_stage_seeds(stage: Stage, scale_factor: float) -> bool:
    """Clamp a stage's seeds toward the budget (never below its minimum)."""
    new_seeds = max(_min_seeds_for(stage), int(stage.seeds * scale_factor))
    if new_seeds >= stage.seeds:
        return False
    logger.info(t"    Stage {stage.name}: seeds {stage.seeds} -> {new_seeds}")
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
    1. Reduce epochs — only on evidence (baseline) stages, floored at
       ``_MIN_EVIDENCE_EPOCHS`` so parity accuracy stays analyzable. Gating
       (non-baseline) stages keep their epochs: a pass rule whose threshold
       becomes unreachable (e.g. xor ``acc >= 0.60`` in 1 epoch) rejects the
       whole model set and yields a degenerate single-survivor report.
    2. Reduce configs (all stages).
    3. Reduce seeds (respecting minimums: 1 for smoke, 10 for evidence).
    """
    import copy

    scaled = copy.deepcopy(campaign)

    current = _estimate_total_wall_time(scaled, epoch_times, models, producer)
    logger.info(
        t"  Current estimate: {current / 60:.1f}min, target: {target_seconds / 60:.1f}min"
    )
    if current <= target_seconds:
        return scaled  # already fits

    # Iteratively reduce until the estimate fits the budget or every resource
    # is at its floor. A single pass is not enough: reducing epochs changes the
    # cost linearly, and re-solving after each step converges on the floor
    # instead of overshooting the budget.
    for _ in range(64):  # bounded: each reduction strictly decreases cost
        current = _estimate_total_wall_time(scaled, epoch_times, models, producer)
        if current <= target_seconds:
            return scaled
        # Clamp per-pass reduction so the schedule converges gradually instead
        # of jumping straight to a floor and under-fitting a generous budget.
        # Each pass removes at most 30% of epochs/configs/seeds; the loop then
        # re-solves, so an evidence stage keeps as many epochs/configs as the
        # budget allows (a more thorough, more informative schedule).
        scale_factor = max(target_seconds / current, 0.7)
        _reduce_epochs_keeping_gates(scaled.stages, scale_factor)
        if (
            _estimate_total_wall_time(scaled, epoch_times, models, producer)
            <= target_seconds
        ):
            return scaled
        for stage in scaled.stages:
            _reduce_stage_configs(stage, scale_factor)
        if (
            _estimate_total_wall_time(scaled, epoch_times, models, producer)
            <= target_seconds
        ):
            return scaled
        for stage in scaled.stages:
            _reduce_stage_seeds(stage, scale_factor)

    current = _estimate_total_wall_time(scaled, epoch_times, models, producer)
    if current > target_seconds:
        logger.warning(
            t"  WARNING: cannot fit {target_seconds / 60:.1f}min with these minimums; estimate {current / 60:.1f}min"
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
    from computronium.experiment.param_estimator import estimate_param_count

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
    from computronium.hyperopt.eval_tiers import estimate_total_time

    deepest = max(stage.epochs for stage in campaign.stages)
    return estimate_total_time(
        _patience_for_epochs(deepest), n_models=len(_all_models(campaign))
    )


def _cmd_validate(config: str) -> int:
    campaign = load_campaign(config)
    arms = ", ".join(sorted(campaign.arms))
    stage_summary = ", ".join(f"{s.name}:{s.task}" for s in campaign.stages)
    logger.info(
        t"OK: campaign {campaign.meta.name!r} valid\n  arms: [{arms}]\n  stages: {stage_summary}"
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
        logger.info(t"Calibrating on {device} to fit {time_budget} budget...")
        epoch_times = _calibrate_epoch_times(
            campaign, device, models, producer, target_seconds
        )
        campaign = _auto_scale_campaign(
            campaign, target_seconds, epoch_times, models, producer
        )
        producer = _producer(campaign, producer_kind, candidates)
        logger.info("Scaled campaign:")

    total = 0
    logger.info(f"campaign: {campaign.meta.name!r}")
    for stage in campaign.stages:
        pairs = _in_budget_pairs(campaign, stage, models, producer)
        n_seeds = stage.seeds
        n_probes = len(pairs) * n_seeds
        total += n_probes
        logger.info(
            f"  stage {stage.name:<16}{len(pairs)} in-budget (model, config) pairs x {n_seeds} seeds = {n_probes} probes"
        )
    logger.info(f"total probes: {total}")

    if time_budget:
        # Report the device-calibrated estimate (matches run's actual cost),
        # not the heuristic tier table.
        est = _estimate_total_wall_time(campaign, epoch_times, models, producer)
        logger.info(
            f"calibrated total time: {est:.0f}s ({est / 60:.1f} min) vs {time_budget} target"
        )
    else:
        budget = _time_budget(campaign)
        logger.info(
            f"estimated total time: {budget['estimated_completion']} ({budget['minutes']:.0f} min across {budget['trials_per_model']} trials_per_model)"
        )
    return 0


def _cmd_run(  # run threads every operator knob; grouped by the parser, not the signature  # ruff: ignore[too-many-locals]
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
        logger.info(t"Calibrating on {resolved_device} to fit {time_budget} budget...")
        epoch_times = _calibrate_epoch_times(
            campaign, resolved_device, models, producer, target_seconds
        )
        campaign = _auto_scale_campaign(
            campaign, target_seconds, epoch_times, models, producer
        )
        logger.info(t"Scaled campaign will be executed.")

    report_path = _report_path(campaign, report_override)
    if report_path.exists() and report_path.stat().st_size:
        logger.info(t"resuming report {report_path} (finished probes are no-ops)")

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
    effective = _resolve_device(campaign, device)
    logger.info(t"running on {effective}")
    try:
        outcomes = runner.run()
    except KeyboardInterrupt:
        logger.info(  # ruff: ignore[logging-too-few-args]
            "#'--report' path is the resume contract; a partial run must be re-runnable; interrupted: %s holds %d finished probes; rerun to resume",
            f"{report_path} holds {len(report.finished_keys())} finished probes; rerun to resume",
        )
        return 130
    except Exception as exc:  # broad: a long overnight run must not lose the resume contract to a single probe/driver crash
        logger.error(
            "run aborted at %d finished probe(s): %s",
            len(report.finished_keys()),
            exc,
            exc_info=True,
        )
        logger.error(  # ruff: ignore[error-instead-of-exception]
            t"run aborted: {report_path} holds {len(report.finished_keys())} finished probes and is resumable (rerun to continue)"
        )
        return 1
    elapsed = time.time() - start

    for outcome in outcomes:
        verdict = outcome.verdict.value
        logger.info(t"  [{verdict:>6}] {outcome.model:<24} {outcome.reason}")
    n_pass = sum(1 for o in outcomes if o.verdict is Verdict.PASS)
    logger.info(t"report written: {report_path}")
    logger.info(t"{n_pass}/{len(outcomes)} PASS; completed in {elapsed:.1f}s")
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
    except yaml.YAMLError, ValueError, FileNotFoundError:
        logger.exception("experiment error")  # user-facing CLI: a traceback is noise
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
        logger.error("report error: %s", exc)  # user-facing CLI: a traceback is noise  # ruff: ignore[error-instead-of-exception]
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
