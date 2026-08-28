"""Continual learning metrics and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from computronium.core.continual.constants import CL_CLASSES_PER_TASK

if TYPE_CHECKING:
    from computronium.core.continual.system import ContinualJointSystem


@dataclass
class CLConfig:
    """Configuration for continual learning experiment."""

    # Model
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10  # CL_TOTAL_CLASSES

    # Training
    epochs_per_task: int = 5
    batch_size: int = 64
    lr: float = 0.001

    # Replay
    replay_capacity: int = 41  # Matched to fast weight plastic state: 512-dim * 64 batch * 4 bytes = 128 KB ≈ 41 samples * 785 * 4 bytes

    # LwF
    lwf_temperature: float = 2.0
    lwf_lambda: float = 1.0

    # SI
    si_xi: float = 0.1

    # EWC
    ewc_lambda: float = 1000.0

    # Stability
    stability_threshold: float = 1.029
    stability_window: int = 10

    # Experiment
    device: str = "auto"
    seed: int = 42
    protocol: str = "task_incremental"  # or "task_free"
    num_workers: int = 0  # 0 to avoid multiprocessing resource leaks


@dataclass
class CLMetrics:
    """Continual learning metrics."""

    # Per-task final accuracies (after all training)
    final_accuracies: list[float] = field(default_factory=list)

    # Accuracy matrix: accuracy_matrix[i][j] = accuracy on task i after training task j
    accuracy_matrix: list[list[float]] = field(default_factory=list)

    # Backward transfer: BWT = mean(acc_i_after_all - acc_i_after_task_i)
    backward_transfer: float = 0.0

    # Forward transfer: FWT = mean(acc_i_after_task_{i-1} - random_init_acc)
    forward_transfer: float = 0.0

    # Forgetting: F_i = max_{j<i} acc_i_after_task_j - acc_i_after_all
    forgetting: list[float] = field(default_factory=list)
    avg_forgetting: float = 0.0

    # Memory footprint
    peak_memory_mb: float = 0.0
    plastic_state_bytes: float = 0.0
    replay_buffer_bytes: float = 0.0

    # Stability rider
    stability_verdicts: list = field(default_factory=list)
    max_spectral_radius: float = 0.0

    # Training time
    total_time_s: float = 0.0


def compute_cl_metrics(
    model: ContinualJointSystem,
    task_loaders: list,
    current_task: int,
    accuracy_matrix: list[list[float]] | None = None,
) -> CLMetrics:
    """Compute comprehensive CL metrics."""
    metrics = CLMetrics()
    metrics.accuracy_matrix = accuracy_matrix or []

    # Evaluate on all tasks up to current_task
    final_accs = []
    device = next(model.parameters()).device
    for i, loader in enumerate(task_loaders):
        if i > current_task:
            final_accs.append(0.0)
            continue
        model.set_task(i)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.view(x.shape[0], -1).to(device)
                y = y.to(device)
                logits = model(x, task_id=i)
                # Mask to task-relevant classes
                task_start = i * CL_CLASSES_PER_TASK
                task_end = task_start + CL_CLASSES_PER_TASK
                task_logits = logits[:, task_start:task_end]
                pred = task_logits.argmax(dim=1)
                # Map global labels to local (0/1)
                local_y = y % CL_CLASSES_PER_TASK
                correct += (pred == local_y).sum().item()
                total += y.shape[0]
        acc = correct / total if total > 0 else 0.0
        final_accs.append(acc)

    metrics.final_accuracies = final_accs

    # Compute backward transfer (only if we have history)
    if len(metrics.accuracy_matrix) > 0 and current_task > 0:
        bwt_sum = 0.0
        for i in range(current_task):
            if i < len(metrics.accuracy_matrix) and current_task < len(
                metrics.accuracy_matrix[i]
            ):
                acc_after_i = metrics.accuracy_matrix[i][i]
                acc_after_all = metrics.accuracy_matrix[i][current_task]
                bwt_sum += acc_after_all - acc_after_i
        metrics.backward_transfer = bwt_sum / current_task if current_task > 0 else 0.0

    # Compute forgetting
    if len(metrics.accuracy_matrix) > 0:
        forgetting = []
        for i in range(current_task + 1):
            if i < len(metrics.accuracy_matrix):
                row = metrics.accuracy_matrix[i]
                if len(row) > current_task:
                    max_acc = max(row[: current_task + 1])
                    final_acc = row[current_task]
                    forgetting.append(max_acc - final_acc)
        metrics.forgetting = forgetting
        metrics.avg_forgetting = (
            sum(forgetting) / len(forgetting) if forgetting else 0.0
        )

    return metrics


__all__ = ["CLConfig", "CLMetrics", "compute_cl_metrics"]
