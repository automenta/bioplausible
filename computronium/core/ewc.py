"""
Elastic Weight Consolidation (EWC) utilities.

Canonical EWC utilities moved from zoo/optimizers/ewc.py to core/ per REFACTOR4.
"""

import torch


def update_fisher(
    model: torch.nn.Module,
    dataloader,
    task_id: int,
    loss_fn=None,
) -> None:
    """Compute Fisher information for a task and store in model's EWC buffers.

    Args:
        model: The model being trained (must have EWC buffers registered).
        dataloader: DataLoader for the task data.
        task_id: Unique identifier for this task.
        loss_fn: Loss function (default: cross_entropy).
    """
    import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]

    model.train()
    fisher = {}
    for p in model.parameters():
        fisher[id(p)] = torch.zeros_like(p)

    for x, y in dataloader:
        model.zero_grad()
        output = model(x)
        loss_f = loss_fn or F.cross_entropy
        loss = loss_f(output, y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                fisher[id(p)] += p.grad.pow(2) / len(dataloader)

    # Store in model's EWC buffers (assumed to exist via register_ewc)
    if not hasattr(model, "_ewc_fisher"):
        model._ewc_fisher = {}
    if not hasattr(model, "_ewc_optimal_params"):
        model._ewc_optimal_params = {}

    model._ewc_fisher[task_id] = fisher
    model._ewc_optimal_params[task_id] = {
        id(p): p.data.clone() for p in model.parameters()
    }


def register_ewc(model: torch.nn.Module) -> None:
    """Register EWC buffers on a model."""
    model._ewc_fisher = {}
    model._ewc_optimal_params = {}


def compute_ewc_loss(
    model: torch.nn.Module, task_id: int, ewc_lambda: float = 0.1
) -> torch.Tensor:
    """Compute EWC regularization loss for a task."""
    if not hasattr(model, "_ewc_fisher") or task_id not in model._ewc_fisher:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    fisher = model._ewc_fisher[task_id]
    optimal = model._ewc_optimal_params[task_id]
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for p in model.parameters():
        pid = id(p)
        if pid in fisher and pid in optimal:
            loss += ewc_lambda * (fisher[pid] * (p - optimal[pid]).pow(2)).sum()

    return loss
