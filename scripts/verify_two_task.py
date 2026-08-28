"""Phase 3.5.2 — Two-Task Catastrophic Forgetting Probe.

Train task 0 (digits 0/1), then task 1 (digits 2/3); measure forgetting on task 0
after training task 1. Expected per TODO5 §3.5.2: backprop ~0.15, EWC ~0.05,
replay ~0.01, fast_weights target <=0.1. Any arm deviating >2x from its expected
range flags a wiring bug.

This is the probe that validates LwF/SI actually differ from backprop (the
single-task probe cannot, since SI omega is empty and LwF has no prior model).

Usage:
    uv run python scripts/verify_two_task.py [--arms ...] [--seeds 3] [--epochs 5] [--hidden_dim 32]
"""

from __future__ import annotations

import argparse
import copy
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


def _eval_task(model, loader, task_id, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            logits = model(x, task_id=task_id)
            task_logits = logits[:, task_id * 2 : task_id * 2 + 2]
            correct += (task_logits.argmax(dim=1) == y).sum().item()
            total += y.shape[0]
    return correct / total if total else 0.0


def two_task_forgetting(
    arm_name: str, config: CLConfig, device: str, seed: int
) -> tuple[float, float, float]:
    """Train task0 then task1; return forgetting on task 0."""
    random.seed(seed)
    torch.manual_seed(seed)

    model, extra = ARM_FACTORIES[arm_name](config, device)
    loaders = {t: _loader(t, config, device, TaskSplit.TRAIN) for t in (0, 1)}
    test_loaders = {t: _loader(t, config, device, TaskSplit.TEST) for t in (0, 1)}

    for task_id in (0, 1):
        model.set_task(task_id)
        # Arm-specific boundary setup
        if arm_name == "fast_weights":
            model.reset_plastic_state()
        elif arm_name in ("ewc", "si"):
            extra.start_task()
        elif arm_name == "lwf":
            extra.set_prev_model(copy.deepcopy(model))

        model.train()
        for epoch in range(config.epochs_per_task):
            for x, y in loaders[task_id]:
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)
                if arm_name == "lwf":
                    from computronium.core.system_trainer import _lwf_train_step

                    _lwf_train_step(model, x, y, task_id, extra)
                elif arm_name == "si":
                    from computronium.core.system_trainer import _si_train_step

                    _si_train_step(model, x, y, task_id, extra)
                elif arm_name == "replay":
                    model.train_step(x, y, task_id=task_id)
                    if len(extra) >= config.batch_size:
                        rx, ry, rt = extra.sample(config.batch_size)
                        model.train_step(rx, ry, task_id=rt[0].item())
                else:
                    model.train_step(x, y, task_id=task_id)
            if arm_name in ("ewc", "si"):
                extra.update_importance()

    # Forgetting on task 0: acc_after_t0 - acc_after_t1
    acc_t0_after_t0 = _eval_task(model, test_loaders[0], 0, device)
    acc_t0_after_t1 = _eval_task(model, test_loaders[0], 0, device)
    # Evaluate task1 too for context
    acc_t1 = _eval_task(model, test_loaders[1], 1, device)
    return acc_t0_after_t0, acc_t0_after_t1, acc_t1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="fast_weights,ewc,backprop,replay,lwf,si")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension (32-64 for capacity-limited probe)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="benchmark_results/arm_verification/two_task.json"
    )
    args = parser.parse_args(argv)

    from pathlib import Path

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    config = CLConfig(
        epochs_per_task=args.epochs,
        batch_size=64,
        seed=0,
        device=device,
        hidden_dim=args.hidden_dim,
    )

    results = {}
    for arm in arms:
        rows = []
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            after_t0, after_t1, acc_t1 = two_task_forgetting(arm, config, device, seed)
            forgetting = after_t0 - after_t1
            rows.append({
                "acc_t0_after_t0": after_t0,
                "acc_t0_after_t1": after_t1,
                "acc_t1": acc_t1,
                "forgetting": forgetting,
            })
            print(
                f"  {arm} seed{seed}: t0_after_t0={after_t0:.3f} t0_after_t1={after_t1:.3f} forgetting={forgetting:+.3f} t1={acc_t1:.3f} ({time.perf_counter() - t0:.0f}s)"
            )
        mean_f = sum(r["forgetting"] for r in rows) / len(rows)
        results[arm] = {"seeds": rows, "mean_forgetting": mean_f}
        print(f"  {arm}: MEAN forgetting = {mean_f:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
