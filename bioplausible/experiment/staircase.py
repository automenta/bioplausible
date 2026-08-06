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
    "aggregate_values",
    "metric_value",
    "passes_stage",
]

ParamCounter = Callable[[str, dict[str, object], int, int], int]


def _default_param_counter(
    model: str, config: dict[str, object], input_dim: int, output_dim: int
) -> int:
    """Count parameters by constructing the real model (the production path)."""
    from bioplausible.experiment.param_estimator import estimate_param_count

    return estimate_param_count(model, config, input_dim=input_dim, output_dim=output_dim)


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


def metric_value(metric: str, result: ProbeResult) -> float:
    """Extract one metric from a probe result (non-finite -> NaN)."""
    match metric:
        case "acc":
            return float(result.final_acc)
        case "epoch_time_s":
            return float(result.epoch_time_s)
        case "loss":
            return float(result.final_train_loss)
        case "flops":
            return float(result.forward_flops or 0)
        case "memory":
            return float(result.peak_memory_mb or 0)
        case _:
            return float("nan")


def aggregate_values(values: list[float], aggregate: str) -> float:
    """Aggregate a list across the requested strategy (``median``/``mean``/``min``)."""
    finite = [v for v in values if _isfinite(v)]
    if not finite:
        return float("nan")
    return float(_AGGREGATORS[aggregate](finite))


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
        outcomes: list[Outcome] = []
        for model in survivors:
            results = self._collect_probes(stage, model, finished)
            passed, reason = passes_stage(stage, results)
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

    def _collect_probes(
        self, stage: Stage, model: str, finished: set[str]
    ) -> list[ProbeResult]:
        """Gather (or resume) all probes for one model under a stage.

        Finished probes are rehydrated from the Report so verdicts reflect the
        full seed set; missing probes are trained and appended. The resume check
        happens **before** training, so a re-launch is a true no-op for
        completed probes.
        """
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
        for work in self.producer.schedule(stage, [model], finished=finished):
            key = _config_key(work.config)
            for seed in range(stage.seeds):
                rkey = f"{stage.name}:{model}:{key}:{seed}"
                if rkey in seen:
                    continue
                probe = self._run_probe(stage, work.model, work.config, seed)
                self.report.append(stage.name, probe)
                results.append(probe)
        return results

    def _run_probe(
        self, stage: Stage, model: str, config: dict[str, object], seed: int
    ) -> ProbeResult:
        geom = self.campaign.geometry(stage.task)
        param_count = self.param_counter(
            model, config, input_dim=geom[0], output_dim=geom[1]
        )
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
