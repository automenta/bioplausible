"""Continual learning benchmark runner."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from computronium.core.continual.arms import (
    create_backprop_arm,
    create_ewc_arm,
    create_fast_weight_arm,
    create_lwf_arm,
    create_replay_arm,
    create_si_arm,
)
from computronium.core.continual.constants import CL_NUM_TASKS
from computronium.core.continual.metrics import CLConfig, CLMetrics, compute_cl_metrics
from computronium.core.continual.stability import (
    check_stability,
    create_stability_guard,
    make_transition_fn,
)
from computronium.core.continual.training import _lwf_train_step, _si_train_step
from computronium.domains.base import TaskSplit
from computronium.domains.vision import SplitMNIST

if TYPE_CHECKING:
    from computronium.core.continual.system import ContinualJointSystem


def run_continual_learning(
    arm_name: str,
    config: CLConfig,
    protocol: str = "task_incremental",
) -> CLMetrics:
    """Run continual learning for one arm."""
    device_str = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device = torch.device(device_str)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Create task loaders
    task_loaders = []
    for task_id in range(CL_NUM_TASKS):
        task = SplitMNIST(task_id=task_id, batch_size=config.batch_size, device=device_str, num_workers=config.num_workers)
        task.setup()
        task_loaders.append(task.get_dataloader(TaskSplit.TRAIN))

    test_loaders = []
    for task_id in range(CL_NUM_TASKS):
        task = SplitMNIST(task_id=task_id, batch_size=config.batch_size, device=device_str, num_workers=config.num_workers)
        task.setup()
        test_loaders.append(task.get_dataloader(TaskSplit.TEST))

    # Create arm
    model: ContinualJointSystem
    extra: dict[str, object] = {}
    if arm_name == "fast_weights":
        model = create_fast_weight_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
    elif arm_name == "ewc":
        model, update = create_ewc_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str, config.ewc_lambda)
        extra["update"] = update
    elif arm_name == "backprop":
        model = create_backprop_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
    elif arm_name == "replay":
        model, buffer = create_replay_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str, config.replay_capacity)
        extra["buffer"] = buffer
    elif arm_name == "lwf":
        model, lwf_loss = create_lwf_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra["lwf_loss"] = lwf_loss
    elif arm_name == "si":
        model, si = create_si_arm(config.input_dim, config.hidden_dim, config.output_dim, device_str)
        extra["si"] = si
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    # Stability guard
    guard = create_stability_guard(
        threshold=config.stability_threshold,
        statistic="fast_proxy",
        window=config.stability_window,
    )
    transition_fn = make_transition_fn(model)
    # Get context from joint system for stability guard
    guard_context = model.context

    # Training
    accuracy_matrix = [[0.0 for _ in range(CL_NUM_TASKS)] for _ in range(CL_NUM_TASKS)]
    stability_verdicts: list = []
    start_time = time.perf_counter()

    if protocol == "task_incremental":
        # Task boundaries are signaled
        for task_id in range(CL_NUM_TASKS):
            model.set_task(task_id)

            # Arm-specific setup at task boundary
            if arm_name == "fast_weights":
                # Reset plastic state at task boundary (new episode)
                model.reset_plastic_state()
            elif arm_name == "ewc":
                update = extra["update"]
                update.consolidate(model.geometry.params)
            elif arm_name == "lwf":
                lwf_loss = extra["lwf_loss"]
                # Save current model as previous for distillation
                prev_model = copy.deepcopy(model)
                lwf_loss.set_prev_model(prev_model)
            elif arm_name == "si":
                si = extra["si"]
                si.start_task()

            loader = task_loaders[task_id]

            for epoch in range(config.epochs_per_task):
                for batch_idx, (x, y) in enumerate(loader):
                    x = x.view(x.shape[0], -1).to(device)
                    y = y.to(device)

                    # Arm-specific training step
                    if arm_name == "lwf":
                        lwf_loss_fn = extra["lwf_loss"]
                        metrics = _lwf_train_step(model, x, y, task_id, lwf_loss_fn)
                    elif arm_name == "si":
                        si_tracker = extra["si"]
                        metrics = _si_train_step(model, x, y, task_id, si_tracker)
                    else:
                        # Use joint system's train_step with task-masked loss
                        metrics = model.train_step(x, y, task_id=task_id)

                    # Stability check
                    verdict = check_stability(guard, transition_fn, x, step=epoch * len(loader) + batch_idx, context=guard_context)
                    stability_verdicts.append(verdict)

                    # Replay buffer update
                    if arm_name == "replay":
                        buffer = extra["buffer"]
                        buffer.add(x, y, task_id)

                    # Replay training
                    if arm_name == "replay" and len(extra["buffer"]) > 0:
                        buffer = extra["buffer"]
                        # Sample up to batch_size (handles case where buffer < batch_size)
                        sample_size = min(config.batch_size, len(buffer))
                        rx, ry, rt = buffer.sample(sample_size)
                        # For replay, we need to train on the replay task
                        # Use the replay sample's task_id
                        replay_task_id = rt[0].item()
                        model.train_step(rx, ry, task_id=replay_task_id)

                # End of task: update importance for EWC/SI
                if arm_name == "ewc":
                    update = extra["update"]
                    update.consolidate(model.geometry.params)
                elif arm_name == "si":
                    si = extra["si"]
                    si.update_importance()

            # Evaluate on all tasks so far
            for eval_task_id in range(task_id + 1):
                model.set_task(eval_task_id)
                correct = 0
                total = 0
                model.eval()
                with torch.no_grad():
                    for x, y in test_loaders[eval_task_id]:
                        x = x.view(x.shape[0], -1).to(device)
                        y = y.to(device)
                        logits = model(x, task_id=eval_task_id)
                        task_start = eval_task_id * 2  # CL_CLASSES_PER_TASK
                        task_end = task_start + 2
                        task_logits = logits[:, task_start:task_end]
                        pred = task_logits.argmax(dim=1)
                        local_y = y % 2
                        correct += (pred == local_y).sum().item()
                        total += y.shape[0]
                accuracy_matrix[eval_task_id][task_id] = correct / total if total > 0 else 0.0

    elif protocol == "task_free":
        # No task boundaries - gradual shift (simulate by mixing tasks)
        all_loaders = [iter(task_loaders[i]) for i in range(CL_NUM_TASKS)]
        total_batches = config.epochs_per_task * max(len(l) for l in task_loaders)

        for batch_idx in range(total_batches):
            task_id = batch_idx % CL_NUM_TASKS
            model.set_task(task_id)

            try:
                x, y = next(all_loaders[task_id])
            except StopIteration:
                all_loaders[task_id] = iter(task_loaders[task_id])
                x, y = next(all_loaders[task_id])

            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)

            model.train_step(x, y, task_id=task_id)

            verdict = check_stability(guard, transition_fn, x, step=batch_idx, context=guard_context)
            stability_verdicts.append(verdict)

            if arm_name == "replay":
                buffer = extra["buffer"]
                buffer.add(x, y, task_id)

            # Replay training (task-free protocol)
            if arm_name == "replay" and len(extra["buffer"]) > 0:
                buffer = extra["buffer"]
                sample_size = min(config.batch_size, len(buffer))
                rx, ry, rt = buffer.sample(sample_size)
                replay_task_id = rt[0].item()
                model.train_step(rx, ry, task_id=replay_task_id)

            # Periodic evaluation
            if batch_idx % (total_batches // CL_NUM_TASKS) == 0:
                eval_task = batch_idx // (total_batches // CL_NUM_TASKS)
                if eval_task < CL_NUM_TASKS:
                    for eval_task_id in range(eval_task + 1):
                        model.set_task(eval_task_id)
                        correct = 0
                        total = 0
                        model.eval()
                        with torch.no_grad():
                            for ex, ey in test_loaders[eval_task_id]:
                                ex = ex.view(ex.shape[0], -1).to(device)
                                ey = ey.to(device)
                                elogits = model(ex, task_id=eval_task_id)
                                task_start = eval_task_id * 2
                                task_end = task_start + 2
                                task_logits = elogits[:, task_start:task_end]
                                epred = task_logits.argmax(dim=1)
                                local_ey = ey % 2
                                correct += (epred == local_ey).sum().item()
                                total += ey.shape[0]
                        accuracy_matrix[eval_task_id][eval_task] = correct / total if total > 0 else 0.0

    total_time = time.perf_counter() - start_time

    # Final evaluation on all tasks
    final_metrics = compute_cl_metrics(model, test_loaders, CL_NUM_TASKS - 1, accuracy_matrix)
    final_metrics.total_time_s = total_time
    final_metrics.stability_verdicts = stability_verdicts
    final_metrics.max_spectral_radius = max(v.statistic for v in stability_verdicts) if stability_verdicts else 0.0

    # Memory footprint
    if hasattr(model.plasticity, "fast_weight_dim"):
        final_metrics.plastic_state_bytes = model.plasticity.fast_weight_dim * 4 * config.batch_size
    if arm_name == "replay" and "buffer" in extra:
        final_metrics.replay_buffer_bytes = extra["buffer"].memory_bytes()

    return final_metrics


def run_continual_learning_suite(
    arms: list[str],
    protocols: list[str],
    output_dir: str | Path,
    config: CLConfig | None = None,
    seeds: int = 3,
) -> dict[str, dict[str, dict[str, object]]]:
    """Run continual learning benchmark suite."""
    config = config or CLConfig()
    output_dir = Path(output_dir)

    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    config.device = device

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for arm in arms:
        all_results[arm] = {}
        for protocol in protocols:
            print(f"\n=== {arm} / {protocol} ===")
            arm_results: dict[str, list[dict[str, float | int]] | dict[str, float]] = {"seeds": []}

            for seed in range(seeds):
                print(f"  Seed {seed}...")
                config.seed = seed
                metrics = run_continual_learning(arm, config, protocol)
                arm_results["seeds"].append({
                    "final_accuracies": metrics.final_accuracies,
                    "accuracy_matrix": metrics.accuracy_matrix,
                    "backward_transfer": metrics.backward_transfer,
                    "forward_transfer": metrics.forward_transfer,
                    "forgetting": metrics.forgetting,
                    "avg_forgetting": metrics.avg_forgetting,
                    "peak_memory_mb": metrics.peak_memory_mb,
                    "plastic_state_bytes": metrics.plastic_state_bytes,
                    "replay_buffer_bytes": metrics.replay_buffer_bytes,
                    "max_spectral_radius": metrics.max_spectral_radius,
                    "stability_kills": sum(1 for v in metrics.stability_verdicts if getattr(v, "kill", False)),
                    "total_time_s": metrics.total_time_s,
                })
                print(f"    Avg forgetting: {metrics.avg_forgetting:.4f}, BWT: {metrics.backward_transfer:.4f}")

            # Aggregate across seeds
            seeds_list = arm_results["seeds"]
            if seeds_list:
                for key in ["avg_forgetting", "backward_transfer", "forward_transfer", "max_spectral_radius", "total_time_s"]:
                    vals = [float(s[key]) for s in seeds_list]
                    mean_val = sum(vals) / len(vals)
                    arm_results[f"mean_{key}"] = mean_val
                    arm_results[f"std_{key}"] = (sum((v - mean_val)**2 for v in vals) / len(vals))**0.5 if len(vals) > 1 else 0.0

            all_results[arm][protocol] = arm_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "continual_learning_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    return all_results


__all__ = [
    "run_continual_learning",
    "run_continual_learning_suite",
]
