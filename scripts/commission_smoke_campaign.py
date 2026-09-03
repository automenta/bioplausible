"""Commission a campaign: per-seed runs → kill/resume lifecycle → records.

Implements TODO8 R5.1a/b/c. Spawns ``comp campaign run`` as a real
subprocess per seed (each seed in its own ``seed_<s>/`` store), SIGKILLs the
first seed mid-flight once fault-tolerance checkpoints exist, resumes it
through the CLI ``--resume`` path, then renders the golden ``manifest.json``,
``report.md``, and ``episodes.json`` from the records merged across seeds.

Usage:
    # R5.1a CPU smoke (single seed, kill→resume lifecycle)
    uv run scripts/commission_smoke_campaign.py --fresh

    # R5.1c commissioned replication campaign (grid, 5 seeds, 2 families)
    uv run scripts/commission_smoke_campaign.py --fresh --device cuda \
        --output-dir autoscientist_campaigns/r51c --campaign-id r51c \
        --space joint_grid --layout grid --seeds 0,1,2,3,4 \
        --tasks synthetic,parity --experiments-per-iter 8 --iterations-resume 18
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIRNAME = "records"
# SIGKILL discards buffered stdout — the pre-kill event trail would be lost.
_SUBPROC_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commission a campaign (TODO8 R5.1a/b/c lifecycle runs)"
    )
    parser.add_argument("--output-dir", default="autoscientist_campaigns/smoke_cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--space", default="joint_smoke")
    parser.add_argument("--objective", default="lifecycle_smoke")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--campaign-id", default="smoke_r51a")
    parser.add_argument(
        "--layout",
        choices=["random", "grid"],
        default="random",
        help="Coordinate proposal (grid = deterministic round-robin over the "
        "full space; shared by every seed for replication)",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seed list; one campaign store per seed",
    )
    parser.add_argument(
        "--tasks",
        default="synthetic",
        help="Comma-separated task families (batch families: synthetic, parity)",
    )
    parser.add_argument("--iterations-first", type=int, default=4)
    parser.add_argument(
        "--iterations-resume",
        type=int,
        default=6,
        help="Iterations after resume on the kill seed; also the total "
        "iteration budget for clean (non-kill) seeds",
    )
    parser.add_argument("--experiments-per-iter", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--kill-after-episodes",
        type=int,
        default=1,
        help="SIGKILL the first seed once this many episodes are durably recorded",
    )
    parser.add_argument(
        "--kill-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for checkpoints before killing anyway",
    )
    parser.add_argument(
        "--no-kill",
        action="store_true",
        help="Skip the kill/resume lifecycle (clean runs for every seed)",
    )
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-families", type=int, default=2)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the output directory before commissioning",
    )
    return parser.parse_args()


def _seed_dir(args: argparse.Namespace, seed: int) -> Path:
    return Path(args.output_dir) / f"seed_{seed}"


def _campaign_id(args: argparse.Namespace, seed: int) -> str:
    return f"{args.campaign_id}_s{seed}"


def _tasks(args: argparse.Namespace) -> list[str]:
    return [t.strip() for t in args.tasks.split(",") if t.strip()]


def _seeds(args: argparse.Namespace) -> list[int]:
    return [int(s) for s in args.seeds.split(",") if s.strip()]


def _seed_cli(
    args: argparse.Namespace, *, seed: int, resume: bool, iterations: int
) -> list[str]:
    return [
        "uv",
        "run",
        "comp",
        "campaign",
        "run",
        "--space",
        args.space,
        "--objective",
        args.objective,
        "--branch",
        args.branch,
        "--layout",
        args.layout,
        "--campaign-id",
        _campaign_id(args, seed),
        "--iterations",
        str(iterations),
        "--experiments-per-iter",
        str(args.experiments_per_iter),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--output-dir",
        str(_seed_dir(args, seed)),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--tasks",
        args.tasks,
        *(["--resume"] if resume else []),
    ]


def _open_store(seed: int, args: argparse.Namespace):
    from computronium.core.campaign.campaign_store import CampaignStore

    d = _seed_dir(args, seed)
    return CampaignStore(d / "campaign.db", d / "checkpoints")


def _episode_count(seed: int, args: argparse.Namespace, campaign_id: str) -> int:
    """Episode count via a live read; tolerates transient write locks."""
    import sqlite3

    try:
        return len(_open_store(seed, args).get_episodes(campaign_id))
    except sqlite3.OperationalError:
        return 0


def _checkpoints(seed: int, args: argparse.Namespace) -> list[Path]:
    d = _seed_dir(args, seed)
    cid = _campaign_id(args, seed)
    return sorted((d / "checkpoints").glob(f"checkpoint_{cid}_*.pkl"))


def _db_state(seed: int, args: argparse.Namespace) -> tuple[int, int, str]:
    """(iteration, episode count, campaign_id) for the seed's latest campaign."""
    store = _open_store(seed, args)
    state = store.get_latest_on_branch(args.branch)
    if state is None:
        return 0, 0, ""
    return (
        state.iteration,
        len(store.get_episodes(state.campaign_id)),
        state.campaign_id,
    )


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the whole session — uv alone would orphan the worker python."""
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()


def _stage_kill(args: argparse.Namespace, seed: int, log_path: Path) -> dict:
    """Run the first seed, SIGKILL mid-flight once episodes are durable."""
    started = time.monotonic()
    killed = False
    cid = _campaign_id(args, seed)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            _seed_cli(args, seed=seed, resume=False, iterations=args.iterations_first),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_SUBPROC_ENV,
        )
        while time.monotonic() - started < args.kill_timeout:
            if proc.poll() is not None:
                break
            if _episode_count(seed, args, cid) >= args.kill_after_episodes:
                killed = True
                _kill_tree(proc)
                break
            time.sleep(0.02)
        else:
            if proc.poll() is None:
                killed = True
                _kill_tree(proc)
        proc.wait()
    iteration, episodes, _ = _db_state(seed, args)
    return {
        "trigger": f"episodes >= {args.kill_after_episodes}",
        "killed": killed,
        "checkpoints_observed": len(_checkpoints(seed, args)),
        "db_iteration_at_kill": iteration,
        "episodes_at_kill": episodes,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


def _stage_run(
    args: argparse.Namespace,
    seed: int,
    iterations: int,
    log_path: Path,
    *,
    resume: bool,
) -> float:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            _seed_cli(args, seed=seed, resume=resume, iterations=iterations),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=_SUBPROC_ENV,
        )
    if proc.returncode != 0:
        sys.exit(f"seed {seed} run failed (exit {proc.returncode}); see {log_path}")
    return round(time.monotonic() - started, 2)


def _git_commit() -> str:
    git = shutil.which("git") or sys.exit("git not found")
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merged_records(args: argparse.Namespace, seeds: list[int]):
    from computronium.core.campaign.frontier_record import FrontierRecord

    records = []
    for seed in seeds:
        store = _open_store(seed, args)
        records.extend(
            FrontierRecord.from_dict(ep.frontier_record)
            for ep in store.get_episodes(_campaign_id(args, seed))
        )
    return records


def _assert_unique_episodes(args: argparse.Namespace, seeds: list[int]) -> None:
    """Resume must never duplicate (iteration, coordinate, task) rows."""
    for seed in seeds:
        store = _open_store(seed, args)
        keys = [
            (ep.iteration, ep.coordinate, ep.task_name)
            for ep in store.get_episodes(_campaign_id(args, seed))
        ]
        if len(keys) != len(set(keys)):
            sys.exit(
                f"seed {seed}: duplicate episode keys after resume "
                f"({len(keys)} rows, {len(set(keys))} unique)"
            )


def _copy_yaml_checkpoints(args: argparse.Namespace, seeds: list[int]) -> None:
    from computronium.core.campaign.campaign_store import CampaignStore

    for seed in seeds:
        d = _seed_dir(args, seed)
        store = CampaignStore(d / "campaign.db", d / "checkpoints")
        checkpoints = store.list_checkpoints(args.branch)
        if checkpoints:
            dest = (
                Path(args.output_dir)
                / RECORDS_DIRNAME
                / (f"checkpoint_s{seed}{checkpoints[-1].suffix}")
            )
            shutil.copy(checkpoints[-1], dest)
            print(f"yaml checkpoint -> {dest}")


def _write_manifest(
    args: argparse.Namespace,
    seeds: list[int],
    seed_details: dict[str, dict],
    replication_summary: dict,
) -> None:
    import torch

    grid_size = None
    if args.layout == "grid":
        from computronium.cli.campaign import _get_search_space
        from computronium.core.campaign import space_grid

        grid_size = len(space_grid(_get_search_space(args.space)))

    config = {
        "space": args.space,
        "objective": args.objective,
        "layout": args.layout,
        "grid_size": grid_size,
        "seeds": seeds,
        "tasks": _tasks(args),
        "branch": args.branch,
        "budget": {
            "iterations_first": args.iterations_first,
            "iterations_resume": args.iterations_resume,
            "experiments_per_iter": args.experiments_per_iter,
            "checkpoint_interval": args.checkpoint_interval,
        },
        "min_seeds": args.min_seeds,
        "min_families": args.min_families,
    }
    manifest = {
        "campaign_id": args.campaign_id,
        **config,
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "determinism": {
            "construction_seeding": True,
            "replay_mode": "tolerance",
        },
        "git_commit": _git_commit(),
        "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "seeds_detail": seed_details,
        "replication_summary": replication_summary,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path = Path(args.output_dir) / RECORDS_DIRNAME / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {path}")


def _write_report(
    args: argparse.Namespace,
    seeds: list[int],
    seed_details: dict[str, dict],
    records: list,
) -> None:
    from computronium.analysis.counterfactual import attribute_axis_effects
    from computronium.core.campaign import pareto_frontier, replication_manifest
    from computronium.core.campaign.stack import (
        DEFAULT_PARETO_MAXIMIZE,
        DEFAULT_PARETO_OBJECTIVES,
    )

    frontier = pareto_frontier(
        records, DEFAULT_PARETO_OBJECTIVES, DEFAULT_PARETO_MAXIMIZE
    )
    attributions = attribute_axis_effects(records, metric="task_accuracy")
    replication = replication_manifest(
        records, min_seeds=args.min_seeds, min_families=args.min_families
    )
    n_replicated = sum(r.replicated for r in replication.values())

    by_coord: dict[str, list] = defaultdict(list)
    for r in records:
        by_coord[r.coordinate].append(r)
    aggregate = sorted(
        (
            (
                coordinate,
                len(rows),
                len({row.seed for row in rows}),
                len({row.task_name for row in rows}),
                sum(row.task_loss for row in rows) / len(rows),
                sum(row.task_accuracy for row in rows) / len(rows),
                sum(row.rho_jacobian for row in rows) / len(rows),
            )
            for coordinate, rows in by_coord.items()
        ),
        key=lambda row: -row[5],
    )

    seed_lines = [
        f"| {s} | `{d['campaign_id']}` | {d['iteration']} | {d['episodes']} "
        f"| {d.get('resume_elapsed_s', d.get('elapsed_s'))} "
        f"| {'kill-resume' if d.get('kill') else 'clean'} |"
        for s, d in seed_details.items()
    ]

    lines = [
        f"# Commissioned campaign report — `{args.campaign_id}`",
        "",
        f"- Layout `{args.layout}` · space `{args.space}` · seeds `{seeds}` "
        f"· tasks `{_tasks(args)}` · device `{args.device}`",
        f"- Episodes persisted: {len(records)} across {len(by_coord)} coordinates",
        "",
        "## Seeds",
        "",
        "| seed | campaign | iteration | episodes | run | lifecycle |",
        "|---|---|---|---|---|---|",
        *seed_lines,
        "",
        "## Coordinates (mean accuracy, best first)",
        "",
        "| coordinate | n | seeds | families | mean loss | mean acc | mean growth |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| `{c}` | {n} | {s} | {f} | {ml:.4f} | {ma:.4f} | {mg:.3f} |"
            for c, n, s, f, ml, ma, mg in aggregate
        ],
        "",
        "## Pareto frontier (task_loss, stability_score, energy)",
        "",
        f"Hypervolume: {frontier.hypervolume:.6g} · "
        f"{len(frontier.frontier)} on frontier / {len(frontier.dominated)} dominated",
        "",
        *[
            f"- `{r.coordinate}` ({r.task_name}, seed {r.seed}, loss={r.task_loss:.4f})"
            for r in frontier.frontier
        ],
        "",
        "## Counterfactual attribution",
        "",
    ]
    if attributions:
        lines += [
            "| axis | from → to | mean Δ | pairs |",
            "|---|---|---|---|",
            *[
                f"| {a.axis} | {a.from_value} → {a.to_value} "
                f"| {a.mean_delta:+.4f} | {a.n_pairs} |"
                for a in attributions
            ],
        ]
    else:
        lines += ["No minimal pairs in this campaign."]

    lines += [
        "",
        "## Replication gate "
        f"(>={args.min_seeds} seeds, >={args.min_families} families)",
        "",
        f"**{n_replicated}/{len(replication)} coordinates replicated.**",
        "",
        "| coordinate | seeds | families | replicated |",
        "|---|---|---|---|",
        *[
            f"| `{c.coordinate}` | {len(c.seeds)} | {len(c.task_families)} "
            f"| {'✅' if c.replicated else '—'} |"
            for c in replication.values()
        ],
        "",
    ]

    path = Path(args.output_dir) / RECORDS_DIRNAME / "report.md"
    path.write_text("\n".join(lines))
    print(f"report -> {path}")


def _write_episodes(args: argparse.Namespace, seeds: list[int], records: list) -> None:
    path = Path(args.output_dir) / RECORDS_DIRNAME / "episodes.json"
    path.write_text(
        json.dumps([r.to_dict() for r in records], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"episodes -> {path} ({len(records)} from seeds {seeds})")


def main() -> None:
    args = _parse_args()
    seeds = _seeds(args)
    out_dir = Path(args.output_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    existing = [s for s in seeds if (_seed_dir(args, s) / "campaign.db").exists()]
    if existing:
        sys.exit(
            f"campaign DB exists for seeds {existing}; pass --fresh to re-commission"
        )
    records_dir = out_dir / RECORDS_DIRNAME
    records_dir.mkdir(parents=True, exist_ok=True)

    seed_details: dict[str, dict] = {}
    for i, seed in enumerate(seeds):
        seed_records = _seed_dir(args, seed) / RECORDS_DIRNAME
        seed_records.mkdir(parents=True, exist_ok=True)
        detail: dict[str, object] = {"campaign_id": _campaign_id(args, seed)}
        if i == 0 and not args.no_kill:
            print(
                f"seed {seed} stage 1: run + mid-flight kill "
                f"(at {args.kill_after_episodes} episode(s))"
            )
            kill = _stage_kill(args, seed, seed_records / "run_first.txt")
            print(f"  kill: {json.dumps(kill)}")
            print(f"seed {seed} stage 2: resume via CLI")
            detail["resume_elapsed_s"] = _stage_run(
                args,
                seed,
                args.iterations_resume,
                seed_records / "run_resume.txt",
                resume=True,
            )
            detail["kill"] = kill
        else:
            print(f"seed {seed}: clean run ({args.iterations_resume} iterations)")
            detail["elapsed_s"] = _stage_run(
                args,
                seed,
                args.iterations_resume,
                seed_records / "run.txt",
                resume=False,
            )
        iteration, episodes, cid = _db_state(seed, args)
        if cid != _campaign_id(args, seed) or episodes == 0:
            sys.exit(
                f"seed {seed} campaign incomplete: iteration={iteration}, "
                f"episodes={episodes}"
            )
        detail["iteration"] = iteration
        detail["episodes"] = episodes
        seed_details[str(seed)] = detail
        print(f"  seed {seed}: {episodes} episodes, iteration {iteration}")

    _assert_unique_episodes(args, seeds)
    records = _merged_records(args, seeds)

    replication = _replication_summary(args, records)
    _copy_yaml_checkpoints(args, seeds)
    _write_manifest(args, seeds, seed_details, replication)
    _write_report(args, seeds, seed_details, records)
    _write_episodes(args, seeds, records)

    per_task = Counter(r.task_name for r in records)
    print(
        f"commissioned: {len(records)} episodes over {len(seeds)} seed(s) "
        f"({dict(per_task)})"
    )


def _replication_summary(args: argparse.Namespace, records: list) -> dict:
    from computronium.core.campaign import replication_manifest

    replication = replication_manifest(
        records, min_seeds=args.min_seeds, min_families=args.min_families
    )
    return {
        "min_seeds": args.min_seeds,
        "min_families": args.min_families,
        "total_coordinates": len(replication),
        "replicated": sum(r.replicated for r in replication.values()),
    }


if __name__ == "__main__":
    main()
