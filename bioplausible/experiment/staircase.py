"""The survivor-cascade staircase (architecture §6.3, §6.6).

This is the thin experiment layer. A model **PASSES** a stage iff, for every
:class:`MetricRule` in the stage's pass rule, the requested aggregate over its
seeds satisfies the rule **and** the ok-seed count is at least
``min_seed_ok``. Non-finite or errored seeds never satisfy ``>=``. Only models
that PASS a stage advance to the next (the survivor cascade).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bioplausible.experiment.probe import ProbeResult
    from bioplausible.experiment.report import Report
    from bioplausible.experiment.schema import MetricRule, Stage

__all__ = [
    "Outcome",
    "StageMetrics",
    "StaircaseRunner",
    "Verdict",
    "passes_stage",
]

ParamCounter = Callable[[str, dict[str, object], int, int], int]


@dataclass(frozen=True, slots=True)
class _StageContext:
    """Immutable, stage-scoped facts shared across a stage's surviving models.

    Bundled so the per-model scheduling helpers take one context instead of a
    growing list of loose parameters (configs, geometry, per-model budget).
    """

    configs: list[dict[str, object]]
    geom: tuple[int, int]
    budget_by_model: dict[str, int | None]


def _default_param_counter(
    model: str, config: dict[str, object], input_dim: int, output_dim: int
) -> int:
    """Count parameters by constructing the real model (the production path)."""
    from bioplausible.experiment.param_estimator import estimate_param_count

    return estimate_param_count(
        model, config, input_dim=input_dim, output_dim=output_dim
    )


_AGGREGATORS = {
    "median": statistics.median,
    "mean": statistics.fmean,
    "min": min,
}


class Verdict(StrEnum):
    """A model's result after a stage."""

    PASS = "PASS"  # ruff: ignore[hardcoded-password-string]  (verdict label, not a secret)
    REJECT = "REJECT"


class StageMetrics:
    """Aggregated per-model metrics over a stage's seeds."""

    def __init__(self, results: list[ProbeResult]) -> None:
        self.results = results
        self.ok = [r for r in results if r.status == "ok"]
        self.ok_seed_count = len(self.ok)
        self.accs = [r.final_acc for r in self.ok]
        self.epoch_times = [r.epoch_time_s for r in self.ok]

    def value(self, metric: str, aggregate: str) -> float:
        """Return the aggregated value for a metric name."""
        values = self._raw_values(metric)
        if not values:
            return float("nan")
        return float(_AGGREGATORS[aggregate](values))

    def _raw_values(self, metric: str) -> list[float]:
        match metric:
            case "acc":
                return self.accs
            case "epoch_time_s":
                return self.epoch_times
            case "loss":
                return [r.final_train_loss for r in self.ok]
            case "flops":
                return [float(r.forward_flops) for r in self.ok]
            case "memory":
                return [r.peak_memory_mb for r in self.ok]
            case _:
                return []


@dataclass(frozen=True, slots=True)
class Outcome:
    """A model's verdict for a stage, with its metrics."""

    model: str
    verdict: Verdict
    metrics: StageMetrics
    reason: str


def _isfinite(value: float) -> bool:
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _satisfies(rule: MetricRule, aggregated: float) -> bool:
    if not _isfinite(aggregated):
        return False
    match rule.op:
        case ">=":
            return aggregated >= rule.value
        case "<=":
            return aggregated <= rule.value
        case ">":
            return aggregated > rule.value
        case "<":
            return aggregated < rule.value
        case _:
            return False


def passes_stage(stage: Stage, results: list[ProbeResult]) -> tuple[bool, str]:
    """Evaluate one model's stage pass rule (survivor gate).

    Args:
        stage: The stage whose pass rule applies.
        results: The model's per-seed probes for this stage.

    Returns:
        ``(passed, reason)`` — passed iff every rule clears the aggregate AND
        ``min_seed_ok`` ok-seeds exist.
    """
    metrics = StageMetrics(results)
    if metrics.ok_seed_count < (stage.pass_rule.min_seed_ok if stage.pass_rule else 1):
        reason = (
            f"ok_seeds={metrics.ok_seed_count} < "
            f"{stage.pass_rule.min_seed_ok if stage.pass_rule else 1}"
        )
        return False, reason

    if stage.pass_rule is None:
        return True, "no pass rule"
    for rule in stage.pass_rule.rules:
        agg = metrics.value(rule.metric, rule.aggregate)
        if not _satisfies(rule, agg):
            return False, (
                f"{rule.metric}({rule.aggregate})={agg:.4f} fails "
                f"{rule.op} {rule.value}"
            )
    return True, "all rules satisfied"


class StaircaseRunner:
    """Runs a campaign's stages, advancing only survivors (architecture §6.6)."""

    def __init__(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]  (runner collects all collaborators; param_counter is the DI seam for tests/CLI)
        self,
        campaign,
        report: Report,
        driver,
        producer,
        compute=None,
        param_counter=None,
    ) -> None:
        self.campaign = campaign
        self.report = report
        self.driver = driver
        self.producer = producer
        self.compute = compute
        self.param_counter = param_counter or _default_param_counter

    def _initial_models(self) -> list[str]:
        models: list[str] = []
        for arm in self.campaign.arms.values():
            models.extend(arm.models)
        return models

    def _run_stage(self, stage: Stage, survivors: list[str]) -> list[Outcome]:
        finished = self.report.finished_keys()
        ctx = _StageContext(
            configs=self.producer.configs_for(stage),  # enumerate the grid once
            geom=self.campaign.geometry(stage.task),
            budget_by_model={
                model: self.campaign.max_params_for(model) for model in survivors
            },
        )
        outcomes: list[Outcome] = []
        for model in survivors:
            results = self._collect_probes(stage, model, ctx, finished)
            passed, reason = passes_stage(stage, results)
            if not results and not passed:
                reason = self._over_budget_reason(model, ctx)
            verdict = Verdict.PASS if passed else Verdict.REJECT
            outcomes.append(
                Outcome(
                    model=model,
                    verdict=verdict,
                    metrics=StageMetrics(results),
                    reason=reason,
                )
            )
        return outcomes

    def _over_budget_reason(self, model: str, ctx: _StageContext) -> str:
        """Explain an empty result set when it is caused by the param budget."""
        over = [f"{cfg}" for cfg in ctx.configs if self._over_budget(model, cfg, ctx)]
        budget = ctx.budget_by_model[model]
        return (
            f"ok_seeds=0: all configs exceed max_params={budget} (over-budget: {over})"
        )

    def _count_params(
        self, model: str, config: dict[str, object], ctx: _StageContext
    ) -> int:
        """Count a (model, config)'s params via the injected ``param_counter``."""
        return self.param_counter(model, config, ctx.geom[0], ctx.geom[1])

    def _over_budget(
        self,
        model: str,
        config: dict[str, object],
        ctx: _StageContext,
    ) -> bool:
        """Return whether a (model, config) exceeds its arm's ``max_params``.

        Used only to explain an all-over-budget rejection
        (:meth:`_over_budget_reason`); the scheduling path in
        :meth:`_collect_probes` computes the count once itself.
        """
        budget = ctx.budget_by_model[model]
        if budget is None:
            return False
        return self._count_params(model, config, ctx) > budget

    def _collect_probes(
        self,
        stage: Stage,
        model: str,
        ctx: _StageContext,
        finished: set[str],
    ) -> list[ProbeResult]:
        """Gather (or resume) all probes for one model under a stage, in budget.

        Finished probes are rehydrated from the Report so verdicts reflect the
        full seed set; missing probes are trained and appended. Budget and
        resume checks both happen **before** training, so over-budget configs
        (architecture §6.3) and completed probes are never re-trained. A config
        whose seeds are all already recorded is skipped entirely — re-running a
        finished campaign builds no models (a true no-op, not just a no-train).
        """
        from bioplausible.experiment.probe import ProbeResult
        from bioplausible.experiment.probe import config_key as _config_key
        from bioplausible.experiment.report import probe_index_key

        results: list[ProbeResult] = []
        seen: set[str] = set()
        for recorded in self.report.stage_results(stage.name):
            if recorded.model != model:
                continue
            key = probe_index_key(stage.name, recorded)
            if key in finished:
                results.append(recorded)
                seen.add(key)
        for config in ctx.configs:
            key = _config_key(config)
            pending = [
                seed
                for seed in range(stage.seeds)
                if f"{stage.name}:{model}:{key}:{seed}" not in seen
            ]
            if not pending:
                continue
            try:
                param_count = self._count_params(model, config, ctx)
            except (
                Exception
            ) as exc:  # broad: a broken static ctor must not abort the run
                probe = ProbeResult(
                    model=model,
                    task=stage.task,
                    config=config,
                    config_key=key,
                    seed=pending[0],
                    status="error",
                    error=f"param estimation failed: {exc}",
                )
                self.report.append(stage.name, probe)
                results.append(probe)
                continue
            budget = ctx.budget_by_model[model]
            if budget is not None and param_count > budget:
                continue
            for seed in pending:
                probe = self._run_probe(stage, model, config, seed, param_count)
                self.report.append(stage.name, probe)
                results.append(probe)
        return results

    def _run_probe(
        self,
        stage: Stage,
        model: str,
        config: dict[str, object],
        seed: int,
        param_count: int,
    ) -> ProbeResult:
        from bioplausible.experiment.probe import run_probe

        return run_probe(
            self.driver,
            model=model,
            task=stage.task,
            config=config,
            seed=seed,
            epochs=stage.epochs,
            device=self._resolve_device(),
            param_count=param_count,
        )

    def _resolve_device(self) -> str:
        if self.compute is None:
            return "cpu"
        device = self.compute.device
        if device == "auto":
            return "cpu"  # CI/overnight runs default to CPU unless overridden
        return device

    def run(self) -> list[Outcome]:
        """Run the full cascade; only survivors advance between stages.

        Returns:
            The ``(stage, model, verdict)`` outcomes for every stage.
        """
        survivors = self._initial_models()
        all_outcomes: list[Outcome] = []
        for stage in self.campaign.stages:
            outcomes = self._run_stage(stage, survivors)
            all_outcomes.extend(outcomes)
            survivors = [o.model for o in outcomes if o.verdict is Verdict.PASS]
        return all_outcomes
