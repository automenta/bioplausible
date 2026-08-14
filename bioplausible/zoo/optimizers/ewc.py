"""
EWC (Elastic Weight Consolidation) optimizer.

Provides a standard optimizer interface for EWC-based learning.
Uses core EWC utilities for Fisher computation.
"""

import torch

from bioplausible.core.ewc import update_fisher
from bioplausible.core.registry import register_optimizer

__all__ = [
    "EWC",
]


@register_optimizer("ewc", family="optimizer")
class EWC:
    """Elastic Weight Consolidation.

    Applies EWC regularization to prevent catastrophic forgetting.
    Tracks importance of parameters via Fisher information.
    """

    def __init__(self, params, lr=0.01, ewc_lambda=0.1):
        self.params = list(params)
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self._fisher: dict[int, torch.Tensor] = {}
        self._optimal_params: dict[int, torch.Tensor] = {}

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                ewc_grad = p.grad.clone()
                for tid, fisher in self._fisher.items():
                    opt = self._optimal_params[tid]
                    ewc_grad = ewc_grad + self.ewc_lambda * fisher * (p - opt)
                p.data.add_(ewc_grad, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def update_fisher(self, model, dataloader, task_id: int, loss_fn=None):
        """Compute Fisher information for a task using core EWC utility.

        Args:
            model: The model being trained.
            dataloader: DataLoader for the task.
            task_id: Unique ID for this task.
            loss_fn: Loss function (default: cross_entropy).
        """
        # Use core utility but adapt storage to this class's internal dicts
        import torch.nn.functional as F

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

        self._fisher[task_id] = {}
        self._optimal_params[task_id] = {}
        for p in model.parameters():
            pid = id(p)
            self._fisher[task_id][pid] = fisher[pid]
            self._optimal_params[task_id][pid] = p.data.clone()
