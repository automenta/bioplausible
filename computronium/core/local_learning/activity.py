"""
Built-in activity update function implementations.

Provides standard activity settling dynamics for the tile algorithm:
- EP activity update: Equilibrium Propagation style relaxation
- Hebbian activity update: Single-pass activity (no relaxation)
- Spiking activity update: Threshold-and-reset spiking dynamics
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.tile.kernels import compute_activity_update

if TYPE_CHECKING:
    from computronium.core.local_learning.protocols import TileState


def ep_activity_update(  # dynamics contract
    tile: TileState,
    *,
    feedback: list[Tensor],
    importance: float,
    step_size: float,
    lambda_error: float,
    clamp_min: float,
    clamp_max: float,
    clamp: bool,
) -> Tensor:
    """Equilibrium activity settling: activity -= step * (error + lambda*act + feedback)."""
    if tile.activity is None or tile.error is None:
        raise ValueError("_ep_activity_update requires settled activity and error")
    return compute_activity_update(
        activity=tile.activity,
        error=tile.error,
        fwd_feedback=feedback,
        importance=importance,
        step_size=step_size,
        lambda_error=lambda_error,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        clamp=clamp,
    )


def hebbian_activity_update(  # dynamics contract signature
    tile: TileState,
    *,
    feedback: list[Tensor],
    importance: float,
    step_size: float,
    lambda_error: float,
    clamp_min: float,
    clamp_max: float,
    clamp: bool,
) -> Tensor:
    """Single-pass activity: settle straight to the prediction (no relaxation)."""
    if tile.prediction is None:
        raise ValueError("_hebbian_activity_update requires a precomputed prediction")
    return tile.prediction


def spiking_activity_update(  # dynamics contract
    tile: TileState,
    *,
    feedback: list[Tensor],
    importance: float,
    step_size: float,
    lambda_error: float,
    clamp_min: float,
    clamp_max: float,
    clamp: bool,
) -> Tensor:
    """Spiking activity: integrate input, fire above threshold, reset.

    Neuron model: ``activity = ReLU(activity - threshold) * (1 - fired)``
    then add the EP-style relaxation on the sub-threshold component.
    """
    if tile.activity is None or tile.error is None:
        raise ValueError("_spiking_activity_update requires settled activity and error")
    relaxed = compute_activity_update(
        activity=tile.activity,
        error=tile.error,
        fwd_feedback=feedback,
        importance=importance,
        step_size=step_size,
        lambda_error=lambda_error,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        clamp=clamp,
    )
    spike_thresh = clamp_max if clamp else 1.0
    fired = relaxed > spike_thresh
    return torch.where(fired, torch.zeros_like(relaxed), relaxed)


__all__ = [
    "ep_activity_update",
    "hebbian_activity_update",
    "spiking_activity_update",
]
