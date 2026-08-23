"""Substrate-Coupled Plasticity: Reuse substrate physics as plasticity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioplausible.core.joint.transition import PlasticityConfig

if TYPE_CHECKING:
    from torch import Tensor

if TYPE_CHECKING:
    from bioplausible.core.joint.context import SystemContext
    from bioplausible.core.joint.state import CompositeState


class SubstrateCoupledPlasticity:
    """Substrate-coupled plasticity: reuse substrate adapters as physical plasticity.

    In this formulation, the plastic state ψ IS the substrate state σ
    (or a subset thereof). The plasticity dynamics are exactly the
    substrate's physical evolution laws (memristive drift, analog noise,
    quantum decoherence, etc.).

    This is a no-op at the plasticity protocol level — the substrate
    itself handles state evolution through its forward/update operators.

    ψ ≡ σ  (plastic state is substrate state)
    """

    config: PlasticityConfig

    def __init__(self, **kwargs) -> None:
        """Initialize substrate-coupled plasticity.

        Args:
            **kwargs: Ignored (kept for config compatibility).
        """
        self.config = PlasticityConfig.substrate_coupled()

    def initial_psi(
        self, context: SystemContext | None, batch_size: int = 1
    ) -> dict[str, Tensor]:
        """Create initial plastic state.

        Substrate-coupled plasticity has no separate plastic state —
        plasticity IS the substrate state evolution.

        Args:
            context: System context (unused).
            batch_size: Batch size (unused, kept for protocol compliance).

        Returns:
            Empty dict (ψ is empty, substrate state in σ).
        """
        return {}

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        """No-op at plasticity level.

        Substrate state evolution is handled by the substrate's
        weight_update operator within the joint transition.

        Args:
            psi: Current plastic state (empty).
            z: Full joint state (substrate state in z.substrate).
            context: Immutable system context.

        Returns:
            Unchanged plastic state (empty).
        """
        return psi


def create_substrate_coupled_plasticity(
    config: PlasticityConfig,
) -> SubstrateCoupledPlasticity:
    """Factory to create SubstrateCoupledPlasticity from PlasticityConfig.

    Args:
        config: PlasticityConfig with plasticity_type="substrate_coupled".

    Returns:
        SubstrateCoupledPlasticity instance.

    Raises:
        ValueError: If config is not substrate_coupled type.
    """
    if config.plasticity_type != "substrate_coupled":
        raise ValueError(
            f"Expected substrate_coupled config, got {config.plasticity_type}"
        )

    return SubstrateCoupledPlasticity(**(config.consolidation_config or {}))
