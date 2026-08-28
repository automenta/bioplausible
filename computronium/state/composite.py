"""CompositeState: Joint intra-episode state z_t = (activity, plastic, substrate)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=False, slots=True)
class CompositeState:
    """Joint intra-episode state: z_t = (activity, plastic, substrate).

    Attributes:
        activity: x_t — neural activations at time t (includes persistent θ refs)
        plastic: ψ_t — fast plastic variables (e.g., eligibility traces, fast weights)
        substrate: σ_t — substrate-owned state (e.g., memristor conductance, analog noise)
    """

    activity: Mapping[str, Tensor]
    plastic: Mapping[str, Tensor]
    substrate: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        # Ensure mappings are mutable dicts for in-place updates during stepping
        if not isinstance(self.activity, dict):
            object.__setattr__(self, "activity", dict(self.activity))
        if not isinstance(self.plastic, dict):
            object.__setattr__(self, "plastic", dict(self.plastic))
        if not isinstance(self.substrate, dict):
            object.__setattr__(self, "substrate", dict(self.substrate))

    @classmethod
    def empty(cls) -> CompositeState:
        """Create an empty joint state."""
        return cls(activity={}, plastic={}, substrate={})

    def clone(self) -> CompositeState:
        """Create a deep copy with cloned tensors (detached from graph)."""
        return CompositeState(
            activity={k: v.detach().clone() for k, v in self.activity.items()},
            plastic={k: v.detach().clone() for k, v in self.plastic.items()},
            substrate={k: v.detach().clone() for k, v in self.substrate.items()},
        )

    def detach_(self) -> CompositeState:
        """Detach all tensors in-place (for stopping gradient flow)."""
        for v in self.activity.values():
            v.detach_()
        for v in self.plastic.values():
            v.detach_()
        for v in self.substrate.values():
            v.detach_()
        return self

    def to(self, device: torch.device | str) -> CompositeState:
        """Move all tensors to device."""
        return CompositeState(
            activity={k: v.to(device) for k, v in self.activity.items()},
            plastic={k: v.to(device) for k, v in self.plastic.items()},
            substrate={k: v.to(device) for k, v in self.substrate.items()},
        )