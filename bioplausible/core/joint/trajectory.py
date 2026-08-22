"""Joint Trajectory: Immutable recorded trajectory for credit assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from bioplausible.core.joint.state import CompositeState

__all__ = ["JointTrajectory"]


@dataclass(frozen=True, slots=True)
class JointTrajectory:
    """Immutable joint trajectory recording.

    Contains the full joint state trajectory z_0, z_1, ..., z_T
    for credit assignment and stability analysis.

    Attributes:
        activity: List of activity snapshots [x_0, x_1, ..., x_T]
        plastic: List of plastic state snapshots [ψ_0, ψ_1, ..., ψ_T]
        substrate: List of substrate state snapshots [σ_0, σ_1, ..., σ_T]
        checkpoint_indices: Indices where gradient checkpoints were placed
    """

    activity: list[dict[str, Tensor]]
    plastic: list[dict[str, Tensor]]
    substrate: list[dict[str, Tensor]]
    checkpoint_indices: list[int]

    def __len__(self) -> int:
        return len(self.activity)

    def get_step(self, t: int) -> "CompositeState":
        """Reconstruct CompositeState at step t."""
        from bioplausible.core.joint.state import CompositeState

        return CompositeState(
            activity=self.activity[t],
            plastic=self.plastic[t] if t < len(self.plastic) else {},
            substrate=self.substrate[t] if t < len(self.substrate) else {},
        )

    def get_activity(self, name: str) -> list[Tensor]:
        """Get trajectory of a specific activity variable."""
        return [step.get(name, torch.empty(0)) for step in self.activity]

    def get_plastic(self, name: str) -> list[Tensor]:
        """Get trajectory of a specific plastic variable."""
        return [step.get(name, torch.empty(0)) for step in self.plastic]

    def get_substrate(self, name: str) -> list[Tensor]:
        """Get trajectory of a specific substrate variable."""
        return [step.get(name, torch.empty(0)) for step in self.substrate]