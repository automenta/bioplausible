"""Joint Transition Protocol: CoupledTransition + NullPlasticity.

The linchpin of the 6-D joint architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from computronium.state import (
    CompositeState,
    CoupledTransition,
    NullPlasticity,
    PlasticityConfig,
    PlasticityPrimitive,
    SystemContext,
)

if TYPE_CHECKING:
    from computronium.ontology import System
    from computronium.state.composite import ActivityValue

__all__ = [
    "CoupledTransition",
    "LegacyDynamicsAsCoupledTransition",
    "NullPlasticity",
    "PlasticityConfig",
    "PlasticityPrimitive",
]


# ============================================================
# Legacy Wrapper: 5-D System as Joint Transition with M=Null
# ============================================================


class LegacyDynamicsAsCoupledTransition:
    """Wraps existing 5-D System as a joint transition with ψ={}, σ={}, M=Null.

    Enables zero-cost compatibility: all existing 5-D compositions remain
    valid as NullPlasticity slices of the joint system.
    """

    def __init__(self, system: System) -> None:

        self.system = system
        self._null_plasticity = NullPlasticity()

    def step(
        self,
        z: CompositeState,
        context: SystemContext,
    ) -> CompositeState:
        """Execute one step using the wrapped 5-D system."""
        # Extract input from activity
        x = z.activity.get("x")
        if not isinstance(x, Tensor):
            raise TypeError(
                "CompositeState.activity must contain a Tensor 'x' key for input"
            )

        # Run 5-D system train_step (includes free + nudged phase)
        # Note: This is a simplification; full integration would interleave
        # plasticity steps within the settling loop.
        new_activity: dict[str, ActivityValue]
        y = z.activity.get("y")
        if not isinstance(y, Tensor):
            # Inference mode
            out = self.system.forward(x)
            new_activity = {"x": x, "output": out}
            new_plastic = {}
            new_substrate = {}
        else:
            # Training mode - use train_step
            metrics = self.system.train_step(x, y)
            # Get updated geometry params as theta
            new_activity = {"x": x, "output": metrics}
            new_plastic = {}
            new_substrate = {}

        return CompositeState(
            activity=new_activity,
            plastic=new_plastic,
            substrate=new_substrate,
        )
