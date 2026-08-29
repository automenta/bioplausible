"""StateVariable and StateRegistry: Metadata and lifecycle management for state variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    from computronium.state.composite import CompositeState


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
