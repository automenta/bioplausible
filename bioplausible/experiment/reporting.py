"""Reporter over an experiment Report (architecture §6.8, §10).

Consumes the append-only JSONL :class:`Report` and renders, per stage: a parity
table (mean +/- bootstrap CI for accuracy, param_count, epoch_time_s vs the
stage baseline, with Cohen's d and Cliff's delta effect sizes), the Pareto
frontier (maximise accuracy, minimise parameters), and the failure manifesto
(every ``status == "error"`` probe). Statistics come from
``validation/statistics`` and the staircase passes from ``experiment.staircase``.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING

from bioplausible.experiment.report import Report
from bioplausible.validation.statistics import bootstrap_ci, cliffs_delta, cohens_d

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bioplausible.experiment.probe import ProbeResult
    from bioplausible.experiment.schema import Stage

__all__ = [
    "failure_manifesto",
    "pareto_frontier",
    "parity_table",
    "render_report",
]

_DEFAULT_N_BOOT = 1_000


def _mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(xs, n_boot=_DEFAULT_N_BOOT)
    return fmean(xs), lo, hi


def parity_table(
    stage: Stage, results: Sequence[ProbeResult], baseline: str | None = None
) -> str:
    """Render a mean +/- CI parity table for one stage's probes."""
    rows: dict[str, list[ProbeResult]] = {}
    for r in results:
        rows.setdefault(r.model, []).append(r)

    header = (
        f"{'model':<24}{'n':>4}{'acc_mean':>10}{'acc_ci_lo':>10}"
        f"{'acc_ci_hi':>10}{'params':>9}{'epoch_s':>9}"
    )
    lines = [f"stage: {stage.name}  task: {stage.task}", header]
    baseline_accs: dict[str, list[float]] = {}
    for model, model_results in rows.items():
        accs = [r.final_acc for r in model_results]
        mean, lo, hi = _mean_ci(accs)
        params = fmean([r.param_count for r in model_results])
        eps = fmean([r.epoch_time_s for r in model_results])
        lines.append(
            f"{model:<24}{len(accs):>4}{mean:>10.4f}{lo:>10.4f}"
            f"{hi:>10.4f}{params:>9.0f}{eps:>9.2f}"
        )
        baseline_accs[model] = accs

    if baseline and baseline in baseline_accs:
        ref = baseline_accs[baseline]
        lines.append(f"effect sizes vs baseline {baseline}:")
        for model, accs in baseline_accs.items():
            if model == baseline:
                continue
            d = cohens_d(ref, accs)
            delta = cliffs_delta(ref, accs)
            lines.append(f"  {model:<22}cohen_d={d:.3f}  cliff_delta={delta:.3f}")
    return "\n".join(lines)


def pareto_frontier(results: Sequence[ProbeResult]) -> list[dict[str, str | float]]:
    """Return the Pareto frontier points (maximise acc, minimise params)."""
    best: dict[str, tuple[float, float]] = {}
    for r in results:
        if r.status != "ok" or not r.param_count:
            continue
        key = r.config_key
        current = best.get(key)
        if current is None or r.final_acc > current[0]:
            best[key] = (r.final_acc, r.param_count)
    dominated: set[str] = set()
    keys = list(best)
    for key_a in keys:
        acc_a, p_a = best[key_a]
        for key_b in keys:
            if key_a == key_b or key_b in dominated:
                continue
            acc_b, p_b = best[key_b]
            if acc_b >= acc_a and p_b <= p_a and (acc_b > acc_a or p_b < p_a):
                dominated.add(key_a)
                break
    return [
        {"config_key": key, "acc": acc, "param_count": p}
        for key, (acc, p) in best.items()
        if key not in dominated
    ]


def failure_manifesto(results: Sequence[ProbeResult]) -> list[str]:
    """List every failed probe in a stage (for the failure manifesto)."""
    return [
        f"{r.model}/{r.task} seed={r.seed}: {r.error}"
        for r in results
        if r.status == "error" and r.error
    ]


def render_report(path: str | Path, baseline: str | None = None) -> str:
    """Render the full human-readable report for an experiment Report.

    Args:
        path: Path to the Report JSONL.
        baseline: Optional baseline model name for effect sizes.

    Returns:
        A multi-line report string.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(  # descriptive message is the public API
            f"report {p} does not exist; run 'biopl-run run' first"
        )
    report = Report(p)
    ok: dict[str, list[ProbeResult]] = {}
    err: dict[str, list[ProbeResult]] = {}
    for stage in report.stage_names():
        for result in report.stage_results(stage):
            (err if result.status == "error" else ok).setdefault(stage, []).append(
                result
            )

    sections: list[str] = []
    for stage, results in sorted(ok.items()):
        sections.append(
            parity_table(_stage_spec(results[0].task, stage), results, baseline)
        )
        frontier = pareto_frontier(results)
        if frontier:
            sections.append(
                "Pareto frontier (maximise acc, minimise params):\n"
                + "\n".join(
                    f"  {point['config_key']}  acc={point['acc']:.4f} "
                    f"params={point['param_count']}"
                    for point in frontier
                )
            )
    for stage, results in sorted(err.items()):
        failures = failure_manifesto(results)
        sections.append(f"failure manifesto (stage: {stage}):")
        sections.extend(f"  - {f}" for f in failures)
    if not sections:
        return f"report {p}: no probe records"
    return "\n\n".join(sections)


def _stage_spec(task: str, name: str = "report") -> Stage:
    """Build a minimal stage spec for report headers when none is provided."""
    from bioplausible.experiment.schema import Stage

    return Stage(name=name, task=task if task else "xor", epochs=1, seeds=1)
