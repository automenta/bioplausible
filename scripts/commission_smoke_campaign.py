"""Commission a smoke campaign: start → checkpoint → kill → resume → records.

Implements TODO8 R5.1a. Spawns ``comp campaign run`` as a real subprocess,
SIGKILLs it once fault-tolerance checkpoints exist, resumes through the CLI
``--resume`` path, then renders ``manifest.json`` + ``report.md`` from the
persisted episode records. Re-run with ``--device cuda`` for the R5.1b quick
campaign; the artifact layout is the template for R5.1c commissioned runs.

Usage:
    uv run scripts/commission_smoke_campaign.py
    uv run scripts/commission_smoke_campaign.py --fresh --device cuda \
        --output-dir autoscientist_campaigns/quick_gpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import] - fixed argv, no shell
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computronium.core.campaign.campaign_store import CampaignStore

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIRNAME = "records"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commission a smoke campaign (TODO8 R5.1a lifecycle run)"
    )
    parser.add_argument("--output-dir", default="autoscientist_campaigns/smoke_cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--space", default="joint_smoke")
    parser.add_argument("--objective", default="lifecycle_smoke")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--campaign-id", default="smoke_r51a")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="synthetic")
    parser.add_argument("--iterations-first", type=int, default=4)
    parser.add_argument(
        "--iterations-resume",
        type=int,
        default=6,
        help="Iterations after resume; > iterations-first so the campaign "
        "finishes the plan and progresses beyond the kill point",
    )
    parser.add_argument("--experiments-per-iter", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--kill-after-episodes",
        type=int,
        default=1,
        help="SIGKILL once this many episodes are durably recorded",
    )
    parser.add_argument(
        "--kill-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for checkpoints before killing anyway",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the output directory before commissioning",
    )
    return parser.parse_args()


def _cli_command(
    args: argparse.Namespace, *, resume: bool, iterations: int
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
        "--campaign-id",
        args.campaign_id,
        "--iterations",
        str(iterations),
        "--experiments-per-iter",
        str(args.experiments_per_iter),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--tasks",
        args.tasks,
        *(["--resume"] if resume else []),
    ]


def _checkpoints(out_dir: Path, campaign_id: str) -> list[Path]:
    return sorted((out_dir / "checkpoints").glob(f"checkpoint_{campaign_id}_*.pkl"))


def _db_state(out_dir: Path, branch: str) -> tuple[int, int, str]:
    """(iteration, episode count, campaign_id) for the latest campaign on branch."""
    from computronium.core.campaign.campaign_store import CampaignStore

    store = CampaignStore(out_dir / "campaign.db", out_dir / "checkpoints")
    state = store.get_latest_on_branch(branch)
    if state is None:
        return 0, 0, ""
    return (
        state.iteration,
        len(store.get_episodes(state.campaign_id)),
        state.campaign_id,
    )


def _open_store(out_dir: Path):
    from computronium.core.campaign.campaign_store import CampaignStore

    return CampaignStore(out_dir / "campaign.db", out_dir / "checkpoints")


def _episode_count(store: CampaignStore, campaign_id: str) -> int:
    """Episode count via a live read; tolerates transient write locks."""
    import sqlite3

    try:
        return len(store.get_episodes(campaign_id))
    except sqlite3.OperationalError:
        return 0


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the whole session — uv alone would orphan the worker python."""
    import os

    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait()


def _stage_one(args: argparse.Namespace, log_path: Path) -> dict[str, object]:
    """Run, SIGKILL mid-flight once episodes are durable, snapshot the DB."""
    out_dir = Path(args.output_dir)
    started = time.monotonic()
    killed = False
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv, no shell
            _cli_command(args, resume=False, iterations=args.iterations_first),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        store = _open_store(out_dir)
        while time.monotonic() - started < args.kill_timeout:
            if proc.poll() is not None:
                break
            if _episode_count(store, args.campaign_id) >= args.kill_after_episodes:
                killed = True
                _kill_tree(proc)
                break
            time.sleep(0.02)
        else:
            if proc.poll() is None:
                killed = True
                _kill_tree(proc)
        proc.wait()
    iteration, episodes, _ = _db_state(out_dir, args.branch)
    return {
        "trigger": f"episodes >= {args.kill_after_episodes}",
        "killed": killed,
        "checkpoints_observed": len(_checkpoints(out_dir, args.campaign_id)),
        "db_iteration_at_kill": iteration,
        "episodes_at_kill": episodes,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


def _stage_two(args: argparse.Namespace, log_path: Path) -> float:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv, no shell
            _cli_command(args, resume=True, iterations=args.iterations_resume),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        sys.exit(f"resume failed (exit {proc.returncode}); see {log_path}")
    return round(time.monotonic() - started, 2)


def _git_commit() -> str:
    git = shutil.which("git") or sys.exit("git not found")
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv, no shell
        [git, "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    campaign_id: str,
    kill: dict[str, object],
    resume_elapsed: float,
) -> None:
    import torch

    iteration, episodes, _ = _db_state(out_dir, args.branch)
    manifest = {
        "campaign_id": campaign_id,
        "branch": args.branch,
        "space": args.space,
        "objective": args.objective,
        "seed": args.seed,
        "tasks": args.tasks.split(","),
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "budget": {
            "iterations_first": args.iterations_first,
            "iterations_resume": args.iterations_resume,
            "experiments_per_iter": args.experiments_per_iter,
            "checkpoint_interval": args.checkpoint_interval,
        },
        "kill": kill,
        "resume_elapsed_s": resume_elapsed,
        "final": {"iteration": iteration, "episodes": episodes},
    }
    path = out_dir / RECORDS_DIRNAME / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {path}")


def _write_report(
    args: argparse.Namespace, out_dir: Path, campaign_id: str, kill: dict[str, object]
) -> None:
    from computronium.core.campaign.stack import CampaignStack

    stack = CampaignStack(out_dir, branch=args.branch, device=None)
    records = stack.frontier_records(campaign_id)
    frontier = stack.pareto(campaign_id=campaign_id)
    attributions = stack.counterfactuals(campaign_id=campaign_id)
    replication = stack.replication(campaign_id=campaign_id)

    per_iteration = Counter(r.episode_index for r in records)
    overlaps = {
        it: n
        for it, n in sorted(per_iteration.items())
        if n > args.experiments_per_iter
    }

    lines = [
        f"# Smoke campaign report — `{campaign_id}`",
        "",
        f"- Device requested: `{args.device}` · seed `{args.seed}`"
        f" · space `{args.space}`",
        f"- Kill stage: {json.dumps(kill)}",
        f"- Episodes persisted: {len(records)} across iterations "
        f"{min(per_iteration)}-{max(per_iteration)}",
    ]
    if overlaps:
        lines += [
            "",
            f"**Crash-semantics note:** iterations {sorted(overlaps)} carry extra "
            "episodes — an episode recorded before the kill is durable, and the "
            "interrupted iteration re-runs from the deterministic coordinate/"
            "batch stream, so its coordinates appear more than once. Metrics for "
            "repeated coordinates differ slightly because parameter init draws "
            "ride the ambient RNG stream rather than the checkpointed θ.",
        ]

    lines += [
        "",
        "## Episodes",
        "",
        "| iter | coordinate | task | loss | acc | growth | latency (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r.episode_index} | `{r.coordinate}` | {r.task_name} "
            f"| {r.task_loss:.4f} | {r.task_accuracy:.4f} "
            f"| {r.rho_jacobian:.3f} | {r.resources.latency:.4f} |"
        )

    lines += [
        "",
        "## Pareto frontier (task_loss, stability_score, energy)",
        "",
        f"Hypervolume: {frontier.hypervolume:.6g} · "
        f"{len(frontier.frontier)} on frontier / {len(frontier.dominated)} dominated",
        "",
    ]
    lines += [f"- `{r.coordinate}` (loss={r.task_loss:.4f})" for r in frontier.frontier]

    lines += ["", "## Counterfactual attribution", ""]
    if attributions:
        lines += [
            "| axis | from → to | mean Δ | pairs |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {a.axis} | {a.from_value} → {a.to_value} "
            f"| {a.mean_delta:+.4f} | {a.n_pairs} |"
            for a in attributions
        ]
    else:
        lines += ["No minimal pairs in this campaign."]

    lines += [
        "",
        "## Replication gate",
        "",
        "| coordinate | seeds | families | replicated |",
        "|---|---|---|---|",
        *[
            f"| `{c.coordinate}` | {len(c.seeds)} | {len(c.task_families)} "
            f"| {'✅' if c.replicated else '—'} |"
            for c in replication.values()
        ],
        "",
        "Smoke-scale expectations: one seed/one task family cannot satisfy the "
        "gate; replication requires R5.1b/c budgets (≥5 seeds, ≥2 families).",
        "",
    ]

    path = out_dir / RECORDS_DIRNAME / "report.md"
    path.write_text("\n".join(lines))
    print(f"report -> {path}")


def _copy_yaml_checkpoint(out_dir: Path, branch: str) -> None:
    from computronium.core.campaign.campaign_store import CampaignStore

    store = CampaignStore(out_dir / "campaign.db", out_dir / "checkpoints")
    checkpoints = store.list_checkpoints(branch)
    if not checkpoints:
        return
    dest = out_dir / RECORDS_DIRNAME / checkpoints[-1].name
    shutil.copy(checkpoints[-1], dest)
    print(f"yaml checkpoint -> {dest}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    if (out_dir / "campaign.db").exists():
        sys.exit(f"{out_dir / 'campaign.db'} exists; pass --fresh to re-commission")
    records_dir = out_dir / RECORDS_DIRNAME
    records_dir.mkdir(parents=True, exist_ok=True)

    print(f"stage 1: run + mid-flight kill (at {args.kill_after_episodes} episode(s))")
    kill = _stage_one(args, records_dir / "run_first.txt")
    print(f"  kill: {json.dumps(kill)}")

    print("stage 2: resume via CLI")
    resume_elapsed = _stage_two(args, records_dir / "run_resume.txt")
    print(f"  resumed in {resume_elapsed}s")

    iteration, episodes, campaign_id = _db_state(out_dir, args.branch)
    if campaign_id != args.campaign_id or episodes == 0:
        sys.exit(
            f"campaign {args.campaign_id} incomplete: "
            f"iteration={iteration}, episodes={episodes}"
        )

    _copy_yaml_checkpoint(out_dir, args.branch)
    _write_manifest(args, out_dir, campaign_id, kill, resume_elapsed)
    _write_report(args, out_dir, campaign_id, kill)
    print(f"commissioned: {episodes} episodes, iteration {iteration}")


if __name__ == "__main__":
    main()
