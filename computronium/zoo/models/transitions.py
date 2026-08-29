"""
TransitionGraph: the single model-declared contract for state transitions.

The model is the sole authority on its single-step state transition modules.
Propagators (EqProp, CHL, MEP) consume ``model.transition_modules()`` instead
of doing their own (hardcoded, divergent) layer discovery.

See ``REFACTOR3.md`` for the full design rationale.

.. deprecated:: 1.0
   The legacy TransitionGraph protocol is deprecated. Use the Geometry axis instead:
   - ``computronium.ontology.geometry.Geometry`` protocol with ``transition_modules()``
   - ``computronium.ontology.geometry.FeedforwardGeometry``, ``RecurrentGeometry``, ``TileGeometry``
   - ``computronium.ontology.config.GeometryConfig`` for configuration
"""

from __future__ import annotations

import warnings

warnings.warn(
    "computronium.zoo.models.transitions is deprecated. "
    "Use computronium.ontology.geometry.Geometry protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Protocol, runtime_checkable

from torch import Tensor, nn

__all__ = ["TransitionGraph", "TransitionGraphMixin"]


@runtime_checkable
class TransitionGraph(Protocol):
    """Model declares its single-step state transition modules.

    Each module in ``transition_modules()`` is invoked in order during ONE
    forward step: ``module(state, external_input?) -> next_state``. The
    propagator iterates these modules to perform:

    - Free phase (EqProp/CHL): ``x -> h_1 -> h_2 -> ... -> h_n``
    - Nudged phase (EqProp): same, with target nudging
    - Energy settling (MEP): same, with energy gradient
    """

    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during ONE forward step.

        Each module: ``forward(state, external_input?) -> next_state``.
        The propagator iterates these modules to perform:

        - Free phase (EqProp/CHL): ``x -> h_1 -> h_2 -> ... -> h_n``
        - Nudged phase (EqProp): same, with target nudging
        - Energy settling (MEP): same, with energy gradient
        """
        ...

    def initial_state(self, x: Tensor) -> Tensor:
        """Initial state from input. Default: x."""
        ...

    def readout(self, final_state: Tensor) -> Tensor:
        """Convert final state to model output. Default: identity."""
        ...

    def num_settling_steps(self) -> int:
        """Iterations of ``transition_modules()`` for free/nudged phases.

        Default: 1 (feedforward). RNNs/EqProp override to >1.
        """
        ...


class TransitionGraphMixin:
    """Auto-discovers ``transition_modules()`` for models with standard
    structure.

    Standard models (``self.layers: nn.ModuleList``, ``self.forward_layers``,
    or a plain stack of Linear/Conv submodules) get ``TransitionGraph`` for
    free. Custom models override ``transition_modules()``.
    """

    # Declarative capability — automatically read by Registry._infer_metadata.
    provides: list[str] = ["transition_graph", "standard_autograd"]

    def transition_modules(self) -> list[nn.Module]:
        # 1. Explicit ModuleList (most common: StandardEqProp, MomentumEquilibrium, ...)
        layers = getattr(self, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return list(layers)
        # 2. Forward layers (DirectedEP).
        forward_layers = getattr(self, "forward_layers", None)
        if isinstance(forward_layers, nn.ModuleList):
            return list(forward_layers)

        raise NotImplementedError(
            f"{type(self).__name__} has no transition_modules(). "
            "Define `self.layers: nn.ModuleList[nn.Module]` or implement "
            "transition_modules()."
        )

    def initial_state(self, x: Tensor) -> Tensor:
        return x

    def readout(self, final_state: Tensor) -> Tensor:
        return final_state

    def num_settling_steps(self) -> int:
        return 1
