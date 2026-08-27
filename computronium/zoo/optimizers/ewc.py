"""
EWC (Elastic Weight Consolidation) optimizer.

Provides a standard optimizer interface for EWC-based learning.
Uses core EWC utilities for Fisher computation.
"""

import torch

from computronium.core.ewc import update_fisher as _core_update_fisher
from computronium.core.registry import register_param_update

__all__ = [
    "EWC",
]


@register_param_update("ewc", family="optimizer")
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
        """Compute Fisher information for a task via the core EWC utility.

        Args:
            model: The model being trained.
            dataloader: DataLoader for the task.
            task_id: Unique ID for this task.
            loss_fn: Loss function (default: cross_entropy).
        """
        # core/ewc.update_fisher stores per-parameter Fisher/optimal snapshots
        # in the model's EWC buffers; adopt them into this optimizer's dicts.
        _core_update_fisher(model, dataloader, task_id, loss_fn)
        self._fisher[task_id] = model._ewc_fisher[task_id]
        self._optimal_params[task_id] = model._ewc_optimal_params[task_id]
