"""
Unified Energy-Based Model Framework (Phase A.1)

Defines the ``EnergyModel`` protocol and ``EBMTrainer`` that together
unify Predictive Coding, Equilibrium Propagation, and Contrastive
Hebbian Learning under a single energy-based abstraction.

Per Millidge et al. (2022, arXiv:2206.02629), all three families are
instances of a single framework — energy-based models at the
infinitesimal inference limit — where backpropagation emerges as the
linearized gradient of the energy at free-phase equilibrium.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    "EBMTrainer",
    "EnergyModel",
    "logger",
]
logger = logging.getLogger(__name__)


@runtime_checkable
class EnergyModel(Protocol):
    """Protocol for energy-based learning algorithms.

    All of PC, EP, and CHL satisfy this protocol. The trainer
    selects the nudging protocol and energy function; the model
    provides settle dynamics and energy computation.

    This is a structural protocol (``@runtime_checkable``) — models
    implement it by defining the required methods, not by subclassing.
    """

    def energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Total free energy of the model given input ``x`` and optional target ``y``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor | None
            Optional target tensor (for supervised energy terms).

        Returns
        -------
        torch.Tensor
            Scalar energy value (lower is better / more stable).
        """
        ...

    def settle(
        self,
        x: torch.Tensor,
        steps: int,
        beta: float = 0.0,
        y: torch.Tensor | None = None,
    ) -> object:
        """Iterate internal states toward equilibrium.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        steps : int
            Number of settling iterations.
        beta : float
            Nudging strength (0.0 = free phase, >0.0 = nudged toward target).
        y : torch.Tensor | None
            Optional target tensor for nudged phase.

        Returns
        -------
        object
            The equilibrium state (type varies by model — typically a
            ``torch.Tensor`` single-state or ``list[torch.Tensor]`` for
            multi-layer models).
        """
        ...

    def contrastive_update(
        self,
        free_state: object,
        nudged_state: object,
        beta: float,
        lr: float = 1.0,
    ) -> None:
        """Apply weight update from free/nudged state difference.

        Parameters
        ----------
        free_state : object
            Equilibrium state from free phase (beta=0).
        nudged_state : object
            Equilibrium state from nudged phase (beta>0).
        beta : float
            Nudging strength used during the nudged phase.
        lr : float
            Learning rate scaling factor.
        """
        ...


class EBMTrainer:
    """Training loop for any model satisfying the ``EnergyModel`` protocol.

    Handles the two-phase (free/nudged) training common to all EBM
    families. The model provides ``settle()`` and ``contrastive_update()``;
    the trainer orchestrates the loop.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.01,
        free_steps: int = 30,
        nudged_steps: int | None = None,
        beta: float = 0.1,
        clip_grad_norm: float | None = None,
    ):
        self.model = model
        self.lr = lr
        self.free_steps = free_steps
        self.nudged_steps = nudged_steps or max(free_steps // 2, 1)
        self.beta = beta
        self.clip_grad_norm = clip_grad_norm

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Single training step: free phase → nudged phase → contrastive update.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        y : torch.Tensor
            Target batch.

        Returns
        -------
        dict[str, float]
            Metrics dict with at least ``"loss"`` and ``"accuracy"``.
        """
        if not isinstance(self.model, EnergyModel):
            return self._fallback_bptt(x, y)

        # Free phase: settle to equilibrium without nudging
        free_state = self.model.settle(x, steps=self.free_steps, beta=0.0)

        # Nudged phase: settle with weak target nudging
        nudged_state = self.model.settle(
            x, steps=self.nudged_steps, beta=self.beta, y=y
        )

        # Contrastive weight update
        self.model.contrastive_update(
            free_state, nudged_state, beta=self.beta, lr=self.lr
        )

        # Metrics from free-phase output
        return self._compute_metrics(x, y, free_state)

    def _fallback_bptt(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Standard backprop-through-time fallback."""
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        return {
            "loss": loss.item(),
            "accuracy": (logits.argmax(dim=1) == y).float().mean().item(),
        }

    def _compute_metrics(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        free_state: object,
    ) -> dict[str, float]:
        """Compute loss/accuracy from the free-phase equilibrium state.

        Subclasses can override for model-specific metric extraction.
        """
        # Default: try to forward the model on x for metrics
        try:
            logits = self.model(x)  # type: ignore[misc]
            loss = nn.functional.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            return {"loss": loss.item(), "accuracy": acc}
        except Exception:
            logger.warning(
                "Could not compute metrics from model forward. "
                "Override _compute_metrics for custom extraction."
            )
            return {"loss": float("nan"), "accuracy": 0.0}
