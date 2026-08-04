#!/usr/bin/env python
"""Round-robin experiment runner for Phase 0.4 / Phase 1 HPO.

A single script that orchestrates compute-matched HPO across multiple families
on multiple tasks, with:

* **Round-robin scheduling** — one trial per model study per cycle, so every
  family advances together and one slow model can't block the others.
* **Live dashboard** — prints a table every ``--interval`` seconds showing
  per-study trials done/total, best accuracy, parity gap vs baseline, and ETA.
* **Intermediate artifacts** — writes ``results/portfolio_interim.csv`` every
  ``--emit-every`` trials so the portfolio can be built at any time.
* **Crash-resilient** — Optuna studies persist in SQLite (``--db``);
  re-running the script continues from where it left off (idempotent).
* **Configurable** — budgets per family/task via a YAML config
  (``--config``), with CLI overrides.

Usage:
    uv run python run_experiment.py --config experiments/phase1.yaml --db compute.db
    uv run python run_experiment.py --family backprop,fa,forward_only,eqprop \\
        --task digits,cifar10 --budget 20 --budget-tier standard \\
        --db compute.db --interval 15 --emit-every 10

Stop with Ctrl-C; re-run to resume (studies are load_if_exists=True).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch

from bioplausible.cli.run import (
    FAMILY_MAP,
    _make_objective,
    _model_compatible,
    _resolve_family_models,
    _safe_sampler_name,
)
from bioplausible.hyperopt.eval_tiers import PatientLevel, get_evaluation_config
from bioplausible.hyperopt.optuna_bridge import create_study

# Active config (set in main from --config). Used by _build_objective to read
# multi-objective settings (objectives/directions) without threading config
# through the round-robin graph.
_ACTIVE_CONFIG: dict[str, Any] | None = None


def _cleanup_corrupt_trials(db_path: str) -> int:
    """Remove corrupt trials (NULL objective values) and stale RUNNING trials from the Optuna DB.

    Returns the number of deleted trials. This runs via raw SQL to bypass Optuna's
    fragile trial reconstruction which crashes on NULL objective values.
    """
    if not Path(db_path).exists():
        return 0
    deleted = 0
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()

        # Find trial IDs with any NULL objective value in trial_values
        cur.execute("""
            SELECT DISTINCT trial_id
            FROM trial_values
            WHERE value IS NULL
        """)
        corrupt_trial_ids = {row[0] for row in cur.fetchall()}

        # Also find stale RUNNING trials (state = 2 is RUNNING in Optuna)
        cur.execute("""
            SELECT trial_id FROM trials WHERE state = 2
        """)
        running_trial_ids = {row[0] for row in cur.fetchall()}

        # Combine: corrupt OR stale
        to_delete = corrupt_trial_ids | running_trial_ids

        if to_delete:
            for table in (
                "trial_values",
                "trial_params",
                "trial_user_attributes",
                "trial_system_attributes",
                "trial_heartbeats",
                "trials",
            ):
                col = "trial_id"
                conn.execute(
                    f"DELETE FROM {table} WHERE {col} IN ({','.join('?' * len(to_delete))})",
                    tuple(to_delete),
                )
            deleted = len(to_delete)
            conn.commit()
            logging.info(
                "[DB-CLEAN] Deleted %d corrupt/stale trials from %s", deleted, db_path
            )
    except Exception as e:
        logging.warning("[DB-CLEAN] Cleanup failed (continuing): %s", e)
    finally:
        conn.close()
    return deleted


from bioplausible.zoo import get_model_spec

logger = logging.getLogger("run_experiment")

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyTarget:
    """A single (family, model, task) study to advance."""

    reg_family: str
    cli_family: str
    model: str
    task: str
    budget: int
    study_name: str
    tier: str

    @property
    def storage_url(self) -> str:
        # Set globally by main() from --db
        return f"sqlite:///{StudyTargetStorage.db_path}"


class StudyTargetStorage:
    """Static class for the active DB path (avoids threading attrs)."""

    db_path: str = "compute.db"


def _load_config(path: str | None) -> dict[str, Any]:
    """Load a YAML experiment config, or return an empty dict if path is None."""
    if not path:
        return {}
    import yaml  # local import: PyYAML is a project dep

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_targets_from_config(
    config: dict[str, Any],
    cli_families: list[str],
    tasks: list[str],
    budget: int,
    tier: str,
) -> list[StudyTarget]:
    """Build the list of StudyTarget from the config or CLI args."""
    # CLI-driven mode (no config file): uniform budget across all targets
    targets: list[StudyTarget] = []

    # Determine per-family/task budgets from config, else fallback to --budget
    family_budgets: dict[str, int] = {}
    tier_budgets: dict[str, int] = {}
    if config:
        family_budgets = {
            k: int(v) for k, v in config.get("family_budgets", {}).items()
        }
        if "default_budget" in config:
            tier_budgets["__default__"] = int(config["default_budget"])

    elig_families = cli_families if cli_families else list(config.get("families", []))
    if not elig_families:
        elig_families = list(FAMILY_MAP.keys())

    for cli_family in elig_families:
        reg_family, models = _resolve_family_models(cli_family)
        if not models:
            logger.warning("No models registered for family '%s'; skipping", cli_family)
            continue
        # Per-model budget: config can override per-model
        cfg_budget = budget
        if cli_family in family_budgets:
            cfg_budget = family_budgets[cli_family]
        elif "__default__" in tier_budgets:
            cfg_budget = tier_budgets["__default__"]
        elif config and "budget" in config:
            cfg_budget = int(config["budget"])

        for task in tasks:
            compatible = [m for m in models if _model_compatible(m, task)]
            for model in compatible:
                # Config can override per-model budget
                per_model = cfg_budget
                key = f"{cli_family}.{model}"
                if config and key in config.get("model_budgets", {}):
                    per_model = int(config["model_budgets"][key])
                study_name = f"{reg_family}_{model}_{task}"
                targets.append(
                    StudyTarget(
                        reg_family=reg_family,
                        cli_family=cli_family,
                        model=model,
                        task=task,
                        budget=per_model,
                        study_name=study_name,
                        tier=tier,
                    )
                )
    return targets


# ---------------------------------------------------------------------------
# Study state polling
# ---------------------------------------------------------------------------


@dataclass
class StudyStats:
    """Snapshot of a study's progress at a moment in time."""

    study_name: str
    family: str
    model: str
    task: str
    budget: int
    complete: int
    total: int
    best_acc: float | None
    last_trial_time_s: float | None
    model_type: str = ""
    locality: str = ""


def _study_stats(target: StudyTarget, storage_url: str) -> StudyStats:
    """Poll a study for current progress (read-only, safe mid-run)."""
    try:
        spec = get_model_spec(target.model)
        model_type = spec.model_type or ""
        locality = spec.credit_locality or ""
    except ValueError:
        model_type = ""
        locality = ""
    empty = StudyStats(
        study_name=target.study_name,
        family=target.cli_family,
        model=target.model,
        task=target.task,
        budget=target.budget,
        complete=0,
        total=0,
        best_acc=None,
        last_trial_time_s=None,
        model_type=model_type,
        locality=locality,
    )
    try:
        study = optuna.load_study(study_name=target.study_name, storage=storage_url)
    except (KeyError, OSError):
        # Study doesn't exist yet
        return empty
    except Exception:
        return empty
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best_acc = None
    if complete_trials:
        # Multi-objective [acc, loss] : index 0 = accuracy
        best_acc = max((t.values[0] for t in complete_trials if t.values), default=None)
    # Avg iteration_time from the most recent N trials (user_attr)
    recent = complete_trials[-5:]
    avg_t: float | None = None
    times = [(t.user_attrs.get("iteration_time") or 0.0) for t in recent]
    times = [t for t in times if t > 0]
    if times:
        avg_t = sum(times) / len(times)
    return StudyStats(
        study_name=target.study_name,
        family=target.cli_family,
        model=target.model,
        task=target.task,
        budget=target.budget,
        complete=len(complete_trials),
        total=len(study.trials),
        best_acc=best_acc,
        last_trial_time_s=avg_t,
        model_type=model_type,
        locality=locality,
    )


def _baselines(stats: list[StudyStats]) -> dict[str, float]:
    """Return {task: baseline_acc} from the backprop family per task."""
    out: dict[str, float] = {}
    for s in stats:
        if s.family == "backprop" and s.best_acc is not None and s.complete > 0:
            if s.task not in out or s.best_acc > out[s.task]:
                out[s.task] = s.best_acc
    return out


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def _fmt_acc(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _progress_bar(fraction: float, width: int = 24) -> str:
    done = round(fraction * width)
    bar = "#" * done + "." * (width - done)
    return f"[{bar}]"


def _render_dashboard(
    stats: list[StudyStats],
    baselines: dict[str, float],
    cycle: int,
    total_done: int,
    elapsed_s: float,
) -> None:
    """Print a live progress table to stdout."""
    lines: list[str] = []
    total_budget = sum(s.budget for s in stats)
    total = max(total_budget, 1)
    pct = 100.0 * total_done / total
    lines.append("")
    lines.append(
        f"=== Cycle {cycle} | {total_done}/{total_budget} trials complete "
        f"({pct:.0f}%) | elapsed {elapsed_s / 60:.1f} min ==="
    )
    lines.append(f"    overall {_progress_bar(total_done / total)}")

    # Group by task for readability
    tasks = sorted({s.task for s in stats})
    for task in tasks:
        task_stats = [s for s in stats if s.task == task]
        bl = baselines.get(task)
        lines.append("")
        lines.append(
            f"--- task: {task}" + (f"  (backprop baseline acc={bl:.4f})" if bl else "")
        )
        header = (
            f"{'model':26s} {'type':22s} {'locality':12s} "
            f"{'done/budget':>11s} {'best_acc':>8s} {'gap_pp':>7s} {'avg_t':>7s}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for s in sorted(task_stats, key=lambda x: (x.family, x.model)):
            done = f"{s.complete:>4d}/{s.budget:<4d}"
            best = _fmt_acc(s.best_acc)
            gap = "—"
            if s.best_acc is not None and bl is not None:
                gap = f"{(bl - s.best_acc) * 100:.1f}"
            avg_t = f"{s.last_trial_time_s:.1f}s" if s.last_trial_time_s else "—"
            mtype = s.model_type or s.family
            lines.append(
                f"{s.model[:26]:26s} {mtype[:22]:22s} {s.locality[:12]:12s} "
                f"{done:>11s} {best:>8s} {gap:>7s} {avg_t:>7s}"
            )

    # ETA based on remaining trials * average trial time (sequential estimate)
    remaining = sum(max(0, s.budget - s.complete) for s in stats)
    avg_per_trial = None
    nonzero = [s.last_trial_time_s for s in stats if s.last_trial_time_s]
    if nonzero:
        avg_per_trial = sum(nonzero) / len(nonzero)
    eta_str = "?"
    if avg_per_trial and remaining:
        eta_s = remaining * avg_per_trial
        if eta_s < 3600:
            eta_str = f"{eta_s / 60:.0f} min"
        else:
            eta_str = f"{eta_s / 3600:.1f} h"
    lines.append("")
    if avg_per_trial:
        lines.append(
            f"ETA: ~{eta_str} ({remaining} trials remaining, "
            f"~{avg_per_trial:.1f}s/trial seq est)"
        )
    else:
        lines.append(
            f"ETA: ~{eta_str} ({remaining} trials remaining — no timing data yet)"
        )
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Intermediate portfolio emission
# ---------------------------------------------------------------------------


def _emit_portfolio(
    stats: list[StudyStats],
    baselines: dict[str, float],
    out_path: str,
) -> None:
    """Write an interim portfolio CSV of per-family best acc + gap per task.

    Aggregates across models within a family (best accuracy wins).
    """
    # Aggregate: {family: {task: best_acc, complete_sum}}
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for s in stats:
        if s.complete == 0:
            continue
        fam = s.family
        agg.setdefault(fam, {}).setdefault(s.task, {"best": 0.0, "n": 0})
        if s.best_acc is not None:
            agg[fam][s.task]["best"] = max(agg[fam][s.task]["best"], s.best_acc)
        agg[fam][s.task]["n"] += s.complete

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_seen = sorted({s.task for s in stats})
    header = (
        ["family", "n_trials"]
        + [f"acc_{t}" for t in tasks_seen]
        + [f"gap_{t}_pp" for t in tasks_seen]
    )
    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        # backprop baseline first
        fams = sorted(agg.keys(), key=lambda f: (f != "backprop", f))
        for fam in fams:
            row: list[str | float] = [fam]
            n_total = sum(agg[fam][t]["n"] for t in tasks_seen if t in agg[fam])
            row.append(n_total)
            for t in tasks_seen:
                row.append(
                    round(agg[fam].get(t, {}).get("best", 0.0), 4)
                    if fam in agg and t in agg[fam]
                    else ""
                )
            for t in tasks_seen:
                bl = baselines.get(t)
                a = agg[fam].get(t, {}).get("best")
                if bl is not None and a is not None and a > 0:
                    row.append(round((bl - a) * 100, 2))
                else:
                    row.append("")
            w.writerow(row)
    logger.info("[EMIT] Interim portfolio -> %s", out_path)


# ---------------------------------------------------------------------------
# Round-robin runner
# ---------------------------------------------------------------------------


def _build_objective(
    target: StudyTarget,
    eval_cfg,
    device: str,
    quick_mode: bool,
    config: dict[str, Any] | None = None,
):
    """Build the Optuna objective closure for one target."""
    try:
        family_for_tag = get_model_spec(target.model).family or target.reg_family
    except ValueError:
        family_for_tag = target.reg_family
    from bioplausible.cli.run import _TrialContext

    ctx = _TrialContext(
        model=target.model,
        family=family_for_tag,
        task=target.task,
        eval_cfg=eval_cfg,
        quick_mode=quick_mode,
        device=device,
        tier_name=target.tier,
    )

    # The objective closes over _DB_PATH (used for run_single_trial_task storage_path).
    # We need to ensure the module-level _DB_PATH matches our --db.
    import bioplausible.cli.run as runmod

    config = _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else {}
    objectives = config.get("objectives", ["accuracy", "loss"])
    directions = config.get("directions", ["maximize", "minimize"])
    return _make_objective(ctx, objectives, directions), runmod


def _ensure_studies(
    targets: list[StudyTarget],
    storage_url: str,
    eval_cfg,
    seed: int,
    device: str,
    config: dict[str, Any] | None = None,
) -> dict[str, optuna.Study]:
    """Create or load all studies; build objectives; return {study_name: study}.

    Each study gets a deterministic, distinct seed derived from the base ``seed``
    and its study name. Passing the same seed to every sampler would make every
    study's trial-0 (and warmup random draws) identical, collapsing the HPO
    exploration across models.
    """
    # Read multi-objective config (default: 2 objectives)
    n_objectives = 2
    if config:
        objectives = config.get("objectives", ["accuracy", "loss"])
        n_objectives = len(objectives)

    studies: dict[str, optuna.Study] = {}
    for i, t in enumerate(targets):
        n_startup = getattr(eval_cfg, "n_startup_trials", 10)
        # Force TPE unless the all-pruned path requires random fallback
        sampler_name = _safe_sampler_name(t.study_name, "tpe", n_startup, storage_url)
        sampler_name = sampler_name  # locked
        study_seed = _per_study_seed(seed, t.study_name, i)
        study = create_study(
            model_names=[t.model],
            n_objectives=n_objectives,
            storage=storage_url,
            study_name=t.study_name,
            use_pruning=eval_cfg.use_pruning,
            sampler_name=sampler_name,
            mode="pareto",
            seed=study_seed,
        )
        studies[t.study_name] = study
    return studies


def _per_study_seed(base_seed: int, study_name: str, index: int) -> int:
    """Deterministically derive a unique sampler seed for one study.

    ``base_seed`` (the --seed CLI value) anchors a stable master key so a re-run
    (with the same config) reproduces the same per-study seeds, while each study
    still explores a different region of the search space.
    """
    master = random.Random(base_seed)
    master.random()  # advance the master stream before deriving
    key = f"{base_seed}:{study_name}:{index}"
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return h % (2**31)  # Optuna/NumPy sampler seeds are 32-bit signed-safe


def _run_round_robin(
    targets: list[StudyTarget],
    studies: dict[str, optuna.Study],
    eval_cfg,
    seed: int,
    device: str,
    storage_url: str,
    interval_s: int,
    emit_every: int,
    out_interim_csv: str,
) -> None:
    """Round-robin: one trial per study per cycle, dashboard each cycle."""

    quick_mode = False  # standard tier
    objectives: dict[str, Any] = {}
    for t in targets:
        obj, _ = _build_objective(t, eval_cfg, device, quick_mode)
        objectives[t.study_name] = obj

    cycle = 0
    start_time = time.time()
    emit_counter = 0
    # Track per-study "no-progress" trial counts so models that always prune
    # (shape-bug / missing train_step) don't make the loop infinite.
    # A model is declared exhausted after `max_stall` trials with 0 COMPLETE.
    exhausted: set[str] = set()
    max_stall = 4

    # Show the empty table immediately so the user sees the full plan upfront.
    init_stats = [_study_stats(t, storage_url) for t in targets]
    _render_dashboard(
        init_stats,
        _baselines(init_stats),
        cycle,
        0,
        0.0,
    )
    try:
        while True:
            remaining = False
            cycle += 1
            cycle_start = time.time()
            for t in targets:
                if t.study_name in exhausted:
                    continue
                study = studies[t.study_name]
                try:
                    trials = study.trials
                except Exception as exc:
                    logger.warning(
                        "[SKIP] %s unreadable (%s: %s); skipping cycle",
                        t.study_name,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                completed = sum(
                    1 for x in trials if x.state == optuna.trial.TrialState.COMPLETE
                )
                total = len([
                    x
                    for x in trials
                    if x.state
                    in (
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                        optuna.trial.TrialState.FAIL,
                    )
                ])
                if completed >= t.budget:
                    continue
                # Stall: many attempts but zero complete trials -> model can't train.
                # Only mark exhausted if we've tried multiple times AND got zero complete.
                # Don't mark exhausted on total == 0 - we need to launch the first trial.
                if total >= max_stall and completed == 0:
                    logger.warning(
                        "[STALL] %s: %d attempts, 0 complete — marking exhausted",
                        t.study_name,
                        total,
                    )
                    exhausted.add(t.study_name)
                    continue
                remaining = True
                obj = objectives[t.study_name]
                trial_no = completed
                print(
                    f"  ▶ [{t.cli_family}/{t.task}] {t.model}  "
                    f"trial {trial_no + 1}/{t.budget}",
                    flush=True,
                )
                try:
                    study.optimize(obj, n_trials=1, show_progress_bar=False)
                except KeyboardInterrupt:
                    logger.warning("Interrupted mid-cycle; studies persisted.")
                    raise
                except (RuntimeError, ValueError, OSError, TypeError) as e:
                    logger.warning(
                        "[SKIP] %s trial failed: %s: %s",
                        t.study_name,
                        type(e).__name__,
                        e,
                    )
                emit_counter += 1
                if emit_counter % emit_every == 0:
                    stats = [_study_stats(t, storage_url) for t in targets]
                    bls = _baselines(stats)
                    _emit_portfolio(stats, bls, out_interim_csv)

            # Dashboard after each cycle
            stats = [_study_stats(t, storage_url) for t in targets]
            bls = _baselines(stats)
            total_done = sum(s.complete for s in stats)
            elapsed = time.time() - start_time
            _render_dashboard(stats, bls, cycle, total_done, elapsed)
            _emit_portfolio(stats, bls, out_interim_csv)

            # Periodic cleanup of stale RUNNING trials (can accumulate from crashes)
            if cycle % 5 == 0:
                _cleanup_corrupt_trials(StudyTargetStorage.db_path)

            if not remaining:
                logger.info("All studies reached their budgets or are exhausted. Done.")
                break
            # Sleep between dashboard refreshes if cycle was instant
            cycle_elapsed = time.time() - cycle_start
            if cycle_elapsed < interval_s:
                time.sleep(interval_s - cycle_elapsed)
    except KeyboardInterrupt:
        logger.warning("Stopped by user; studies persisted in %s", storage_url)
        stats = [_study_stats(t, storage_url) for t in targets]
        bls = _baselines(stats)
        _emit_portfolio(stats, bls, out_interim_csv)
        _render_dashboard(
            stats,
            bls,
            cycle,
            sum(s.complete for s in stats),
            time.time() - start_time,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin experiment runner for Phase 1 HPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", help="YAML config file (optional)")
    parser.add_argument(
        "--family",
        help="Comma-separated CLI family labels (overrides config)",
    )
    parser.add_argument(
        "--task",
        default="digits",
        help="Comma-separated task names (default: digits)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Per-model Optuna trials (0 = config/tier default)",
    )
    parser.add_argument(
        "--budget-tier",
        dest="tier",
        default="standard",
        choices=["smoke", "shallow", "standard", "deep"],
        help="Compute-matching tier",
    )
    parser.add_argument(
        "--db",
        default="compute.db",
        help="SQLite DB file (default: compute.db)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for samplers",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Dashboard refresh interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--emit-every",
        type=int,
        default=10,
        help="Emit interim portfolio every N trials (default: 10)",
    )
    parser.add_argument(
        "--out-interim",
        default="results/portfolio_interim.csv",
        help="Interim portfolio CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # quiet per-trial noise
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Our runner logs at INFO; optuna at WARNING
    logger.setLevel(logging.INFO)

    # Set the module-level storage in cli.run so objectives write to --db
    config = _load_config(args.config)
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    cli_families = (
        [f.strip() for f in args.family.split(",") if f.strip()] if args.family else []
    )
    budget = args.budget if args.budget > 0 else int(config.get("budget", 20))

    targets = _resolve_targets_from_config(
        config, cli_families, tasks, budget, args.tier
    )
    if not targets:
        parser.error("No study targets resolved. Check --family/--task/config.")

    # Set module-level storage in cli.run (used by objective -> run_single_trial_task)
    import bioplausible.cli.run as runmod

    db_path = args.db
    if not str(db_path).endswith(".db"):
        db_path = f"{db_path}.db"
    runmod._DB_PATH = db_path
    runmod._STORAGE_URL = f"sqlite:///{db_path}"
    storage_url = runmod._STORAGE_URL
    StudyTargetStorage.db_path = db_path

    # Clean corrupt/stale trials BEFORE any Optuna study access
    _cleanup_corrupt_trials(db_path)

    device = args.device
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[DEVICE] auto -> %s", resolved)
        print(f"[DEVICE] auto -> {resolved}", flush=True)

    tier = PatientLevel(args.tier)
    eval_cfg = get_evaluation_config(tier)
    print(
        f"[TIER] {tier.value}: epochs={eval_cfg.epochs} "
        f"batch={eval_cfg.batch_size} n_startup={eval_cfg.n_startup_trials} "
        f"pruning={eval_cfg.use_pruning}",
        flush=True,
    )

    print(
        f"[TARGETS] {len(targets)} studies across "
        f"{len({t.cli_family for t in targets})} families x "
        f"{len(tasks)} tasks, budget={budget}/model",
        flush=True,
    )
    by_family: dict[str, list[StudyTarget]] = {}
    for t in targets:
        by_family.setdefault(t.cli_family, []).append(t)
    for cli_family, fam_targets in by_family.items():
        models = sorted({t.model for t in fam_targets})
        tasks_here = sorted({t.task for t in fam_targets})
        b = fam_targets[0].budget
        print(
            f"  [{cli_family}] {len(models)} models x {len(tasks_here)} tasks, "
            f"budget {b}/model",
            flush=True,
        )
        for m in models:
            try:
                spec = get_model_spec(m)
                tinfo = f"{spec.model_type}"
            except ValueError:
                tinfo = ""
            print(f"      - {m:28s} {tinfo}", flush=True)

    studies = _ensure_studies(targets, storage_url, eval_cfg, args.seed, device, config)

    print(
        f"\nStarting round-robin. Ctrl-C to stop; re-run to resume. "
        f"Dashboard every `{args.interval}s`, emit every `{args.emit_every}` trials.",
        flush=True,
    )
    _run_round_robin(
        targets=targets,
        studies=studies,
        eval_cfg=eval_cfg,
        seed=args.seed,
        device=device,
        storage_url=storage_url,
        interval_s=args.interval,
        emit_every=args.emit_every,
        out_interim_csv=args.out_interim,
    )

    # Final portfolio
    final_csv: str = args.out_interim.replace("_interim", "_final")
    stats = [_study_stats(t, storage_url) for t in targets]
    bls = _baselines(stats)
    _emit_portfolio(stats, bls, final_csv)
    print(f"\n[DONE] Final portfolio -> {final_csv}", flush=True)


if __name__ == "__main__":
    main()
