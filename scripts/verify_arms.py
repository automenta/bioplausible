"""Phase 3.5 — Arm Implementation Verification.

Probe every continual-learning arm's ability to learn a single binary Split-MNIST
task (Phase 3.5.1 gate: backprop baseline >=95%; all arms should learn, not sit at
chance). Run once per arm on task 0 (digits 0/1) and task 1 (digits 2/3).

This reproduces the Phase 2 null-result suspicion that non-replay arms were not
correctly calibrated (chance-level per-task accuracy in
`benchmark_results/continual_learning_full_rerun_v2/`).

Usage:
    uv run python scripts/verify_arms.py [--arms fast_weights,backprop,...] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch

from computronium.core.system_trainer import (
    CLConfig,
    create_backprop_arm,
    create_ewc_arm,
    create_fast_weight_arm,
    create_lwf_arm,
    create_replay_arm,
    create_si_arm,
)
from computronium.domains.base import TaskSplit
from computronium.domains.vision import SplitMNIST

ARM_FACTORIES = {
    "fast_weights": lambda c, d: (
        create_fast_weight_arm(c.input_dim, c.hidden_dim, c.output_dim, d),
        {},
    ),
    "ewc": lambda c, d: tuple(
        create_ewc_arm(c.input_dim, c.hidden_dim, c.output_dim, d, c.ewc_lambda)
    ),
    "backprop": lambda c, d: (
        create_backprop_arm(c.input_dim, c.hidden_dim, c.output_dim, d),
        {},
    ),
    "replay": lambda c, d: tuple(
        create_replay_arm(c.input_dim, c.hidden_dim, c.output_dim, d, c.replay_capacity)
    ),
    "lwf": lambda c, d: tuple(
        create_lwf_arm(c.input_dim, c.hidden_dim, c.output_dim, d)
    ),
    "si": lambda c, d: tuple(create_si_arm(c.input_dim, c.hidden_dim, c.output_dim, d)),
}


def _loader(task_id: int, config: CLConfig, device: str, split: TaskSplit):
    task = SplitMNIST(
        task_id=task_id, batch_size=config.batch_size, device=device, num_workers=0
    )
    task.setup()
    return task.get_dataloader(split)


def single_task_accuracy(
    arm_name: str, task_id: int, config: CLConfig, device: str, seed: int
) -> float:
    """Train one arm on a single binary task and return test accuracy."""
    random.seed(seed)
    torch.manual_seed(seed)

    model, extra = ARM_FACTORIES[arm_name](config, device)
    train_loader = _loader(task_id, config, device, TaskSplit.TRAIN)
    test_loader = _loader(task_id, config, device, TaskSplit.TEST)

    # LwF needs a frozen previous model so distillation is active even for
    # single-task probes on task 1.
    if arm_name == "lwf":
        import copy

        extra.set_prev_model(copy.deepcopy(model))

    model.set_task(task_id)
    model.train()
    for epoch in range(config.epochs_per_task):
        for x, y in train_loader:
            x = x.view(x.shape[0], -1).to(device)  # ruff: ignore[redefined-loop-name]
            y = y.to(device)  # ruff: ignore[redefined-loop-name]
            if arm_name == "lwf":
                from computronium.core.system_trainer import _lwf_train_step

                _lwf_train_step(model, x, y, task_id, extra)
            elif arm_name == "si":
                from computronium.core.system_trainer import _si_train_step

                _si_train_step(model, x, y, task_id, extra)
            else:
                model.train_step(x, y, task_id=task_id)

    # Evaluate on test set (task-masked)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.shape[0], -1).to(device)  # ruff: ignore[redefined-loop-name]
            y = y.to(device)  # ruff: ignore[redefined-loop-name]
            logits = model(x, task_id=task_id)
            task_logits = logits[:, task_id * 2 : task_id * 2 + 2]
            pred = task_logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total if total else 0.0


def verify_arms(arms: list[str], seeds: int, epochs: int, device: str) -> dict:
    config = CLConfig(epochs_per_task=epochs, batch_size=64, seed=0, device=device)
    results = {}
    for arm in arms:
        results[arm] = {}
        for task_id in (0, 1):
            accs = []
            for seed in range(seeds):
                t0 = time.perf_counter()
                accs.append(single_task_accuracy(arm, task_id, config, device, seed))
                print(
                    f"  {arm} task{task_id} seed{seed}: {accs[-1]:.3f} ({time.perf_counter() - t0:.1f}s)"
                )
            results[arm][task_id] = {"accs": accs, "mean": sum(accs) / len(accs)}
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="fast_weights,ewc,backprop,replay,lwf,si")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="benchmark_results/arm_verification/single_task.json"
    )
    args = parser.parse_args(argv)

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(
        f"Verifying arms {arms} on {device}, seeds={args.seeds}, epochs={args.epochs}"
    )
    results = verify_arms(arms, args.seeds, args.epochs, device)

    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    for arm, tasks in results.items():
        for task_id, r in tasks.items():
            flag = (
                "OK"
                if r["mean"] >= 0.95
                else ("<chance?" if r["mean"] < 0.60 else "LOW")
            )
            print(f"  {arm:12s} task{task_id}: mean={r['mean']:.3f} {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
