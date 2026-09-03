"""
Shared Energy Function Library (Phase A.3)

A small library of energy functions reusable across all EBM families
(PC, EP, CHL). Each function returns a scalar ``torch.Tensor`` suitable
for use in settling loops or contrastive updates.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "contrastive_energy",
    "hybrid_energy",
    "mse_energy",
    "node_energy",
    "prediction_error_energy",
    "supervised_energy",
]


def prediction_error_energy(
    activities: list[torch.Tensor],
    predictions: list[torch.Tensor],
    weights: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Prediction-error energy: sum of squared errors between layers.

    Computes ``Σ_i ||activities[i+1] - predictions[i]||²``, optionally
    weighted by ``weights[i]``.

    Parameters
    ----------
    activities : list[torch.Tensor]
        Layer activities (including input as first element).
    predictions : list[torch.Tensor]
        Top-down predictions for each layer.
    weights : list[torch.Tensor] | None
        Optional per-layer scalar weights.

    Returns
    -------
    torch.Tensor
        Scalar energy (sum of squared errors).
    """
    total = torch.tensor(0.0, device=activities[0].device)
    n = min(len(activities) - 1, len(predictions))
    for i in range(n):
        err = activities[i + 1] - predictions[i]
        sq = (err * err).sum()
        if weights is not None and i < len(weights):
            sq = sq * weights[i]  # ruff: ignore[non-augmented-assignment]
        total = total + sq  # ruff: ignore[non-augmented-assignment]
    return total


def supervised_energy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor] = F.cross_entropy,
) -> torch.Tensor:
    """Supervised energy from logits vs targets.

    Parameters
    ----------
    logits : torch.Tensor
        Output logits.
    targets : torch.Tensor
        Target labels.
    loss_fn : Callable
        Loss function (default: cross-entropy).

    Returns
    -------
    torch.Tensor
        Scalar supervised energy term.
    """
    return loss_fn(logits, targets)


def hybrid_energy(
    activities: list[torch.Tensor],
    predictions: list[torch.Tensor],
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervised_weight: float = 1.0,
) -> torch.Tensor:
    """Hybrid energy combining prediction error + supervised term.

    ``E = prediction_error_energy(activities, predictions) + supervised_weight * supervised_energy(logits, targets)``

    Parameters
    ----------
    activities : list[torch.Tensor]
        Layer activities.
    predictions : list[torch.Tensor]
        Top-down predictions.
    logits : torch.Tensor
        Output logits.
    targets : torch.Tensor
        Target labels.
    supervised_weight : float
        Weight for the supervised term.

    Returns
    -------
    torch.Tensor
        Scalar hybrid energy.
    """
    pred = prediction_error_energy(activities, predictions)
    sup = supervised_energy(logits, targets)
    return pred + supervised_weight * sup


def contrastive_energy(
    free_energy: torch.Tensor,
    nudged_energy: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Contrastive loss: difference between free and nudged energies.

    ``L = (E_nudged - E_free) / beta``

    This is the core contrastive learning objective used by EqProp.

    Parameters
    ----------
    free_energy : torch.Tensor
        Energy at free-phase equilibrium.
    nudged_energy : torch.Tensor
        Energy at nudged-phase equilibrium.
    beta : float
        Nudging strength.

    Returns
    -------
    torch.Tensor
        Scalar contrastive loss.
    """
    return (nudged_energy - free_energy) / max(beta, 1e-12)


def mse_energy(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error energy.

    Parameters
    ----------
    target : torch.Tensor
        Target values.
    prediction : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Scalar MSE.
    """
    return F.mse_loss(prediction, target)


def node_energy(
    activity: torch.Tensor,
    bias: torch.Tensor | None = None,
    reg_weight: float = 0.0,
) -> torch.Tensor:
    """Single-node energy (activation penalty).

    ``E = reg_weight * Σ activity²`` (optional bias term).

    Parameters
    ----------
    activity : torch.Tensor
        Node activity.
    bias : torch.Tensor | None
        Optional bias term.
    reg_weight : float
        Regularization strength.

    Returns
    -------
    torch.Tensor
        Scalar node energy.
    """
    e = reg_weight * (activity * activity).sum()
    if bias is not None:
        e = e + (bias * activity).sum()  # ruff: ignore[non-augmented-assignment]
    return e
