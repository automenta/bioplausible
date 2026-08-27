"""Continual learning losses: LwF and Synaptic Intelligence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from computronium.core.continual.constants import CL_CLASSES_PER_TASK


class LwFLoss(nn.Module):
    """LwF loss: distillation from previous model + current task CE."""

    def __init__(self, temperature: float = 2.0, lambda_lwf: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_lwf = lambda_lwf
        self.prev_model: nn.Module | None = None

    def set_prev_model(self, model: nn.Module) -> None:
        """Set the previous model for distillation."""
        self.prev_model = model
        for p in self.prev_model.parameters():
            p.requires_grad_(False)
        self.prev_model.eval()

    def forward(self, logits: Tensor, targets: Tensor, task_id: int, prev_logits: Tensor | None = None) -> Tensor:
        """Compute loss: CE on current task classes + distillation from previous model.

        Args:
            logits: Full 10-class logits [batch, 10]
            targets: Target labels (0 or 1 for current task)
            task_id: Current task ID
            prev_logits: Previous model's full 10-class logits [batch, 10]
        """
        # Current task classes
        task_start = task_id * CL_CLASSES_PER_TASK
        task_end = task_start + CL_CLASSES_PER_TASK
        task_logits = logits[:, task_start:task_end]
        # Map targets 0/1 to 0/1 for the task's 2 classes
        task_targets = targets % CL_CLASSES_PER_TASK
        ce_loss = F.cross_entropy(task_logits, task_targets)

        if self.prev_model is None or task_id == 0 or prev_logits is None:
            return ce_loss

        # Distillation loss on old task logits
        distill = self.distill_only(logits, task_id, prev_logits)
        if distill is None:
            return ce_loss
        return ce_loss + distill

    def distill_only(
        self, logits: Tensor, task_id: int, prev_logits: Tensor
    ) -> Tensor | None:
        """Return only the LwF distillation term (CE is handled by the caller).

        Returns None when there is no prior model / no old classes to distill.
        """
        if self.prev_model is None or task_id == 0 or prev_logits is None:
            return None
        num_old_classes = task_id * CL_CLASSES_PER_TASK
        if num_old_classes <= 0:
            return None
        soft_targets = F.softmax(
            prev_logits[:, :num_old_classes] / self.temperature, dim=1
        )
        soft_logits = F.log_softmax(
            logits[:, :num_old_classes] / self.temperature, dim=1
        )
        distill = (
            F.kl_div(soft_logits, soft_targets, reduction="batchmean")
            * (self.temperature**2)
        )
        return self.lambda_lwf * distill


class SynapticIntelligence:
    """Synaptic Intelligence: importance-weighted parameter regularization.

    Computes per-parameter importance (omega) online during training,
    then regularizes changes to important parameters.
    """

    def __init__(self, model: nn.Module, xi: float = 0.1, epsilon: float = 1e-3):
        self.model = model
        self.xi = xi
        self.epsilon = epsilon
        self.omega: dict[int, Tensor] = {}  # Parameter importance
        self.prev_params: dict[int, Tensor] = {}  # Parameters at task boundary
        self.W: dict[int, Tensor] = {}  # Accumulated parameter-specific contribution

    def start_task(self) -> None:
        """Call at the start of each new task."""
        # Store current parameters as reference for this task
        for name, param in self.model.named_parameters():
            pid = id(param)
            self.prev_params[pid] = param.data.clone()
            if pid not in self.W:
                self.W[pid] = torch.zeros_like(param.data)

    def update_importance(self) -> None:
        """Update parameter importance (omega) at task boundary."""
        for name, param in self.model.named_parameters():
            pid = id(param)
            if pid in self.prev_params:
                # Delta from task start
                delta = param.data - self.prev_params[pid]
                # Accumulate contribution: path integral of gradients * delta
                if param.grad is not None:
                    self.W[pid] += -param.grad * delta
                # Update omega (importance)
                self.omega[pid] = self.W[pid] / (delta**2 + self.epsilon)
                # Reset W for next task
                self.W[pid].zero_()

    def regularization_loss(self) -> Tensor:
        """Compute SI regularization loss."""
        if not self.omega:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            pid = id(param)
            if pid in self.omega and pid in self.prev_params:
                loss += (self.omega[pid] * (param - self.prev_params[pid]) ** 2).sum()
        return self.xi * loss


__all__ = ["LwFLoss", "SynapticIntelligence"]
