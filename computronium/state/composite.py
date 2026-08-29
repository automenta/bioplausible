"""CompositeState: Joint intra-episode state z_t = (activity, plastic, substrate)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

type ActivityValue = Tensor | list[Tensor] | float | dict[str, float]


@dataclass(frozen=False, slots=True)
class CompositeState:
    """Joint intra-episode state: z_t = (activity, plastic, substrate).

    Attributes:
        activity: x_t — neural activations at time t (includes persistent θ refs)
        plastic: ψ_t — fast plastic variables (e.g., eligibility traces, fast weights)
        substrate: σ_t — substrate-owned state (e.g., memristor conductance, analog noise)
    """

    activity: dict[str, ActivityValue]
    plastic: dict[str, Tensor]
    substrate: dict[str, Tensor]

    # For compatibility with StateDynamics.settle which expects state.x
    @property
    def x(self) -> Tensor | None:
        val = self.activity.get("x")
        return val if isinstance(val, Tensor) else None

    @x.setter
    def x(self, value: Tensor | None) -> None:
        if value is None:
            self.activity.pop("x", None)
        else:
            self.activity["x"] = value

    @property
    def y(self) -> Tensor | None:
        val = self.activity.get("y")
        return val if isinstance(val, Tensor) else None

    @y.setter
    def y(self, value: Tensor | None) -> None:
        if value is None:
            self.activity.pop("y", None)
        else:
            self.activity["y"] = value

    @property
    def activations(self) -> list[Tensor] | Tensor | None:
        """Get all layer activations (for backward compat)."""
        val = self.activity.get("activations")
        if isinstance(val, (list, Tensor)):
            return val
        return None

    @activations.setter
    def activations(self, value: list[Tensor] | Tensor | None) -> None:
        if value is None:
            self.activity.pop("activations", None)
        else:
            self.activity["activations"] = value

    @property
    def free_state(self) -> list[Tensor] | Tensor | None:
        val = self.activity.get("free_state")
        if isinstance(val, (list, Tensor)):
            return val
        return None

    @free_state.setter
    def free_state(self, value: list[Tensor] | Tensor | None) -> None:
        if value is None:
            self.activity.pop("free_state", None)
        else:
            self.activity["free_state"] = value

    @property
    def nudged_state(self) -> list[Tensor] | Tensor | None:
        val = self.activity.get("nudged_state")
        if isinstance(val, (list, Tensor)):
            return val
        return None

    @nudged_state.setter
    def nudged_state(self, value: list[Tensor] | Tensor | None) -> None:
        if value is None:
            self.activity.pop("nudged_state", None)
        else:
            self.activity["nudged_state"] = value

    @property
    def loss(self) -> Tensor | float | None:
        val = self.activity.get("loss")
        if isinstance(val, (Tensor, float, int)):
            return val
        return None

    @loss.setter
    def loss(self, value: Tensor | float | None) -> None:
        if value is None:
            self.activity.pop("loss", None)
        else:
            self.activity["loss"] = value

    @property
    def metrics(self) -> dict[str, float] | None:
        val = self.activity.get("metrics")
        if isinstance(val, dict):
            return val
        return None

    @metrics.setter
    def metrics(self, value: dict[str, float] | None) -> None:
        if value is None:
            self.activity.pop("metrics", None)
        else:
            self.activity["metrics"] = value

    def __post_init__(self) -> None:
        # Ensure mappings are mutable dicts for in-place updates during stepping
        if not isinstance(self.activity, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "activity", dict(self.activity))
        if not isinstance(self.plastic, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "plastic", dict(self.plastic))
        if not isinstance(self.substrate, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "substrate", dict(self.substrate))

    @classmethod
    def empty(cls) -> CompositeState:
        """Create an empty joint state."""
        return cls(activity={}, plastic={}, substrate={})

    def clone(self) -> CompositeState:
        """Create a deep copy with cloned tensors (detached from graph)."""
        new_activity: dict[str, ActivityValue] = {}
        for k, v in self.activity.items():
            if isinstance(v, Tensor):
                new_activity[k] = v.detach().clone()
            elif isinstance(v, list):
                new_activity[k] = [t.detach().clone() for t in v]
            else:
                new_activity[k] = v
        return CompositeState(
            activity=new_activity,
            plastic={k: v.detach().clone() for k, v in self.plastic.items()},
            substrate={k: v.detach().clone() for k, v in self.substrate.items()},
        )

    def detach_(self) -> CompositeState:
        """Detach all tensors in-place (for stopping gradient flow)."""
        for v in self.activity.values():
            if isinstance(v, Tensor):
                v.detach_()
            elif isinstance(v, list):
                for t in v:
                    t.detach_()
        for v in self.plastic.values():
            v.detach_()
        for v in self.substrate.values():
            v.detach_()
        return self

    def to(self, device: torch.device | str) -> CompositeState:
        """Move all tensors to device."""
        new_activity: dict[str, ActivityValue] = {}
        for k, v in self.activity.items():
            if isinstance(v, Tensor):
                new_activity[k] = v.to(device)
            elif isinstance(v, list):
                new_activity[k] = [t.to(device) for t in v]
            else:
                new_activity[k] = v
        return CompositeState(
            activity=new_activity,
            plastic={k: v.to(device) for k, v in self.plastic.items()},
            substrate={k: v.to(device) for k, v in self.substrate.items()},
        )
