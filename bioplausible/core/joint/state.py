"""Joint State Architecture: CompositeState, StateVariable, StateRegistry.

Provides the 6-D joint state representation and lifecycle management.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from bioplausible.core.joint.trajectory import JointTrajectory


__all__ = [
    "CompositeState",
    "JointTrajectoryRecorder",
    "StateRegistry",
    "StateVariable",
]


# ============================================================
# StateVariable: Metadata for a single state tensor
# ============================================================


@dataclass(frozen=True, slots=True)
class StateVariable:
    """Metadata describing a state variable's lifecycle properties.

    Attributes:
        name: Unique identifier for this state variable.
        persistent: Survives episode boundaries (traditionally θ, weights).
        fast_plastic: Evolves via intra-episode plasticity law (traditionally ψ).
        substrate_owned: Subject to physical device constraints (traditionally σ).
        consolidatable: Can be promoted to persistent state at episode end.
    """

    name: str
    persistent: bool = False
    fast_plastic: bool = False
    substrate_owned: bool = False
    consolidatable: bool = False

    def __post_init__(self) -> None:
        # A variable must have at least one lifecycle role
        if not any((
            self.persistent,
            self.fast_plastic,
            self.substrate_owned,
            self.consolidatable,
        )):
            raise ValueError(
                f"StateVariable {self.name!r} must have at least one lifecycle role "
                "(persistent, fast_plastic, substrate_owned, or consolidatable)"
            )
        # consolidatable implies fast_plastic
        if self.consolidatable and not self.fast_plastic:
            raise ValueError(
                f"StateVariable {self.name!r}: consolidatable=True requires fast_plastic=True"
            )


# ============================================================
# StateRegistry: Manages all state variables with lifecycle validation
# ============================================================


@runtime_checkable
class StateRegistryProtocol(Protocol):
    """Protocol for StateRegistry to enable dependency inversion."""

    def register(self, var: StateVariable) -> None: ...
    def validate(self, z: CompositeState) -> None: ...
    def lifecycle_groups(self) -> dict[str, list[str]]: ...


class StateRegistry:
    """Registry managing all state variables with lifecycle validation.

    The registry enforces the lifecycle contract for the joint system:
    - Persistent (θ) variables are immutable intra-episode
    - Fast plastic (ψ) variables evolve via plasticity projection
    - Substrate-owned (σ) variables respect physical constraints
    - Consolidatable variables are promoted at episode boundaries
    """

    def __init__(self) -> None:
        self._variables: dict[str, StateVariable] = {}

    def register(self, var: StateVariable) -> None:
        """Register a state variable.

        Args:
            var: StateVariable to register.

        Raises:
            ValueError: If a variable with the same name already exists.
        """
        if var.name in self._variables:
            raise ValueError(
                f"StateVariable {var.name!r} already registered. "
                f"Existing: {self._variables[var.name]}"
            )
        self._variables[var.name] = var

    def get(self, name: str) -> StateVariable | None:
        """Get a state variable by name."""
        return self._variables.get(name)

    def validate(self, z: CompositeState) -> None:
        """Validate that CompositeState contains all registered variables.

        Args:
            z: Joint state to validate.

        Raises:
            ValueError: If required variables are missing or have wrong types.
        """
        # Check persistent variables (θ)
        for name, var in self._variables.items():
            if var.persistent:
                if name not in z.activity:
                    raise ValueError(
                        f"Persistent variable {name!r} missing from activity"
                    )
                if not isinstance(z.activity[name], Tensor):
                    raise ValueError(f"Persistent variable {name!r} must be Tensor")

        # Check fast plastic variables (ψ)
        for name, var in self._variables.items():
            if var.fast_plastic:
                if name not in z.plastic:
                    raise ValueError(
                        f"Fast plastic variable {name!r} missing from plastic"
                    )
                if not isinstance(z.plastic[name], Tensor):
                    raise ValueError(f"Fast plastic variable {name!r} must be Tensor")

        # Check substrate-owned variables (σ)
        for name, var in self._variables.items():
            if var.substrate_owned:
                if name not in z.substrate:
                    raise ValueError(
                        f"Substrate-owned variable {name!r} missing from substrate"
                    )
                if not isinstance(z.substrate[name], Tensor):
                    raise ValueError(
                        f"Substrate-owned variable {name!r} must be Tensor"
                    )

    def lifecycle_groups(self) -> dict[str, list[str]]:
        """Group variable names by lifecycle role.

        Returns:
            Dict mapping lifecycle role -> list of variable names.
        """
        groups: dict[str, list[str]] = {
            "persistent": [],
            "fast_plastic": [],
            "substrate_owned": [],
            "consolidatable": [],
        }
        for name, var in self._variables.items():
            if var.persistent:
                groups["persistent"].append(name)
            if var.fast_plastic:
                groups["fast_plastic"].append(name)
            if var.substrate_owned:
                groups["substrate_owned"].append(name)
            if var.consolidatable:
                groups["consolidatable"].append(name)
        return groups

    def __contains__(self, name: str) -> bool:
        return name in self._variables

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self):
        return iter(self._variables.values())


# ============================================================
# CompositeState: Joint intra-episode state z_t = (x_t, ψ_t, σ_t)
# ============================================================


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


# ============================================================
# JointTrajectoryRecorder: Checkpointed trajectory recording
# ============================================================


@dataclass(slots=True)
class JointTrajectoryRecorder:
    """Records joint trajectories with gradient checkpointing support.

    Addresses autograd graph fragmentation from long settling trajectories
    by recording only what's needed for credit assignment, with optional
    checkpointing for memory efficiency.

    Attributes:
        max_steps: Maximum trajectory length to record.
        checkpoint_interval: Steps between gradient checkpoints (0 = no checkpointing).
        record_plastic: Whether to record ψ trajectory.
        record_substrate: Whether to record σ trajectory.
    """

    max_steps: int = 1000
    checkpoint_interval: int = 0
    record_plastic: bool = True
    record_substrate: bool = True

    # Internal trajectory buffers
    _activity_traj: list[dict[str, Tensor]] = field(default_factory=list, init=False)
    _plastic_traj: list[dict[str, Tensor]] = field(default_factory=list, init=False)
    _substrate_traj: list[dict[str, Tensor]] = field(default_factory=list, init=False)
    _checkpoint_indices: list[int] = field(default_factory=list, init=False)

    def record(self, z: CompositeState) -> None:
        """Record a joint state snapshot.

        Args:
            z: Current joint state to record.
        """
        if len(self._activity_traj) >= self.max_steps:
            return

        # Clone activity (always needed for credit assignment)
        self._activity_traj.append({
            k: v.detach().clone() for k, v in z.activity.items()
        })

        # Optionally record plastic state
        if self.record_plastic:
            self._plastic_traj.append({
                k: v.detach().clone() for k, v in z.plastic.items()
            })

        # Optionally record substrate state
        if self.record_substrate:
            self._substrate_traj.append({
                k: v.detach().clone() for k, v in z.substrate.items()
            })

        # Mark checkpoint if interval set
        if (
            self.checkpoint_interval > 0
            and len(self._activity_traj) % self.checkpoint_interval == 0
        ):
            self._checkpoint_indices.append(len(self._activity_traj) - 1)

    def get_trajectory(self) -> JointTrajectory:
        """Return recorded trajectory as an immutable JointTrajectory."""
        # Import locally to avoid circular dependency
        import bioplausible.core.joint.trajectory as trajectory_module

        return trajectory_module.JointTrajectory(
            activity=self._activity_traj,
            plastic=self._plastic_traj if self.record_plastic else [],
            substrate=self._substrate_traj if self.record_substrate else [],
            checkpoint_indices=self._checkpoint_indices,
        )

    def clear(self) -> None:
        """Clear recorded trajectory."""
        self._activity_traj.clear()
        self._plastic_traj.clear()
        self._substrate_traj.clear()
        self._checkpoint_indices.clear()

    def __len__(self) -> int:
        return len(self._activity_traj)
