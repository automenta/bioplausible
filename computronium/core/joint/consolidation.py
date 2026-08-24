"""Episode-boundary consolidation: ψ → θ promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computronium.core.joint.context import SystemContext
    from computronium.core.joint.state import CompositeState

__all__ = ["ConsolidationConfig", "consolidate"]


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    """Configuration for episode-boundary consolidation.

    Attributes:
        promote_all: If True, promote all consolidatable ψ to θ.
        promotion_scale: Scaling factor for promoted values.
        reset_plastic: If True, reset promoted ψ to zero after consolidation.
    """

    promote_all: bool = True
    promotion_scale: float = 1.0
    reset_plastic: bool = True


def consolidate(
    z_final: CompositeState,
    context: SystemContext,
    config: ConsolidationConfig | None = None,
) -> SystemContext:
    """Promote consolidatable ψ → θ at episode boundaries only.

    This is the ONLY place where θ (persistent parameters) can be modified.
    Intra-episode, θ is immutable (enforced by SystemContext frozen dataclass
    and trainer's torch.no_grad() wrapping).

    Args:
        z_final: Final joint state at episode boundary.
        context: Current system context (contains θ, registry).
        config: Consolidation configuration.

    Returns:
        New SystemContext with updated θ.
    """
    if config is None:
        config = ConsolidationConfig()

    registry = context.registry
    lifecycle_groups = registry.lifecycle_groups()
    consolidatable_names = lifecycle_groups["consolidatable"]

    # Build new theta by promoting consolidatable plastic variables
    new_theta = dict(context.theta)

    for name in consolidatable_names:
        if name not in z_final.plastic:
            continue

        psi_tensor = z_final.plastic[name]
        if psi_tensor is None:
            continue

        # Promotion: ψ → θ with optional scaling
        promoted = (psi_tensor.detach() * config.promotion_scale).requires_grad_(True)

        if name in new_theta:
            # Add to existing parameter (e.g., fast weights added to base weights)
            new_theta[name] = (new_theta[name] + promoted).requires_grad_(True)
        else:
            # New parameter (should be registered as persistent)
            new_theta[name] = promoted

        # Optionally reset plastic state
        if config.reset_plastic:
            # Use .data.zero_() to avoid in-place grad issues
            z_final.plastic[name].data.zero_()

    # Return new context with updated theta
    return context.with_updated_theta(new_theta)
