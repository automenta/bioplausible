"""Shared state accessor utilities for StateDynamics implementations.

Provides a unified protocol and accessor functions for working with
composite state objects across different dynamics implementations.
"""

from typing import Protocol

from torch import Tensor


class StateProtocol(Protocol):
    """Protocol for state objects used in dynamics.

    State objects can be either simple tensors or composite objects
    with activity, substrate, plastic, and metrics attributes.
    """

    pass  # Structural protocol - any object with the expected attributes works


def _is_composite_state(state: object) -> bool:
    """Check if state is a composite object (not a plain tensor)."""
    return not isinstance(state, Tensor)


def _get_state_x(state: object) -> Tensor | None:
    """Get the primary state tensor (x) from a state object."""
    if isinstance(state, Tensor):
        return state
    return getattr(state, "activity", None)


def _get_state_activations(state: object) -> list[Tensor] | Tensor | None:
    """Get layer activations from a state object."""
    if isinstance(state, Tensor):
        return state
    # Try different attribute names
    for attr in ("activations", "activity", "x"):
        val = getattr(state, attr, None)
        if val is not None:
            return val
    return None


def _get_state_free_state(state: object) -> list[Tensor] | Tensor | None:
    """Get free-phase state from a state object."""
    if isinstance(state, Tensor):
        return state
    return getattr(state, "free_state", None)


def _get_state_nudged_state(state: object) -> list[Tensor] | Tensor | None:
    """Get nudged-phase state from a state object."""
    if isinstance(state, Tensor):
        return state
    return getattr(state, "nudged_state", None)


def _get_state_loss(state: object) -> Tensor | float | None:
    """Get loss value from a state object."""
    if isinstance(state, Tensor):
        return None
    return getattr(state, "loss", None)


def _get_state_metrics(state: object) -> dict[str, float] | None:
    """Get metrics dictionary from a state object."""
    if isinstance(state, Tensor):
        return None
    return getattr(state, "metrics", None)


def _get_state_activity(state: object) -> dict | None:
    """Get activity dictionary from a state object."""
    if isinstance(state, Tensor):
        return None
    return getattr(state, "activity", None)


def _set_state_x(state: object, value: Tensor | None) -> None:
    """Set the primary state tensor (x) on a state object."""
    if isinstance(state, Tensor):
        raise TypeError("Cannot set x on plain tensor state")
    if hasattr(state, "activity"):
        state.activity = value


def _set_state_activations(state: object, value: list[Tensor] | Tensor | None) -> None:
    """Set layer activations on a state object."""
    if isinstance(state, Tensor):
        raise TypeError("Cannot set activations on plain tensor state")
    if hasattr(state, "activations"):
        state.activations = value
    elif hasattr(state, "activity"):
        state.activity = value


def _set_state_free_state(state: object, value: list[Tensor] | Tensor | None) -> None:
    """Set free-phase state on a state object."""
    if isinstance(state, Tensor):
        raise TypeError("Cannot set free_state on plain tensor state")
    if hasattr(state, "free_state"):
        state.free_state = value


def _set_state_nudged_state(state: object, value: list[Tensor] | Tensor | None) -> None:
    """Set nudged-phase state on a state object."""
    if isinstance(state, Tensor):
        raise TypeError("Cannot set nudged_state on plain tensor state")
    if hasattr(state, "nudged_state"):
        state.nudged_state = value


def _set_state_loss(state: object, value: Tensor | float | None) -> None:
    """Set loss value on a state object."""
    if isinstance(state, Tensor):
        raise TypeError("Cannot set loss on plain tensor state")
    if hasattr(state, "loss"):
        state.loss = value


def _set_state_metrics(state: object, value: dict[str, float] | None) -> None:
    """Set metrics dictionary on a state object."""
    if isinstance(state, Tensor):
        raise TypeError("Cannot set metrics on plain tensor state")
    if hasattr(state, "metrics"):
        state.metrics = value
