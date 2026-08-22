"""Fast Weight Plasticity: Episode-local associative memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from bioplausible.core.joint.transition import PlasticityConfig

if TYPE_CHECKING:
    from bioplausible.core.joint.context import SystemContext
    from bioplausible.core.joint.state import CompositeState


@dataclass(frozen=True, slots=True)
class FastWeightPlasticityConfig:
    """Configuration for fast weight plasticity dynamics.

    Attributes:
        fast_weight_dim: Dimension of fast weight matrix (flattened).
        decay: Decay factor for fast weights between steps.
        learning_rate: Learning rate for Hebbian update.
        outer_product_scale: Scaling for outer product.
    """

    fast_weight_dim: int = 512
    decay: float = 0.9
    learning_rate: float = 0.1
    outer_product_scale: float = 1.0


class FastWeightPlasticity:
    """Fast weight plasticity: episode-local associative memory.

    Maintains fast weights that accumulate Hebbian associations
    within an episode. Decays between episodes, consolidated
    at episode boundaries.

    ψ = fast_weights updated as:
        A_{t+1} = decay * A_t + lr * outer(pre_t, post_t)

    The fast weights can modulate activity dynamics or be
    consolidated into persistent weights at episode boundaries.
    """

    config: PlasticityConfig

    def __init__(
        self,
        fast_weight_dim: int = 512,
        decay: float = 0.9,
        learning_rate: float = 0.1,
        outer_product_scale: float = 1.0,
    ) -> None:
        """Initialize fast weight plasticity.

        Args:
            fast_weight_dim: Dimension of fast weight vector.
            decay: Decay factor per step.
            learning_rate: Hebbian update learning rate.
            outer_product_scale: Scale for outer product.
        """
        self._config = FastWeightPlasticityConfig(
            fast_weight_dim=fast_weight_dim,
            decay=decay,
            learning_rate=learning_rate,
            outer_product_scale=outer_product_scale,
        )
        self.config = PlasticityConfig.fast_weights(fast_weight_dim=fast_weight_dim)

    @property
    def fast_weight_dim(self) -> int:
        return self._config.fast_weight_dim

    def initial_psi(self, context: "SystemContext | None") -> dict[str, Tensor]:
        """Create initial plastic state.

        Args:
            context: System context (unused, kept for protocol compliance).

        Returns:
            Dict with fast_weights initialized to zero.
        """
        batch_size = 4  # Default, will be overridden by first step
        return {"fast_weights": torch.zeros(batch_size, self.fast_weight_dim)}

    def step(
        self,
        psi: dict[str, Tensor],
        z: "CompositeState",
        context: "SystemContext",
    ) -> dict[str, Tensor]:
        """Compute next plastic state via Hebbian update.

        Args:
            psi: Current plastic state with fast_weights.
            z: Full joint state (activity, plastic, substrate).
            context: Immutable system context.

        Returns:
            Updated plastic state with evolved fast weights.
        """
        fast_weights = psi["fast_weights"]
        batch_size = fast_weights.shape[0]

        # Decay existing fast weights
        new_fast_weights = self._config.decay * fast_weights

        # Hebbian update if pre and post activity available
        if "x" in z.activity and "y" in z.activity:
            pre = z.activity["x"]  # [batch, input_dim]
            post = z.activity["y"]  # [batch, output_dim] or [batch]

            # Handle different post shapes
            if post.dim() == 1:
                post = post.unsqueeze(-1)  # [batch, 1]
            elif post.dim() > 2:
                post = post.flatten(1)  # [batch, ...]

            # Compute outer product per batch element
            # pre: [batch, input_dim], post: [batch, output_dim]
            # outer: [batch, input_dim * output_dim]
            for b in range(batch_size):
                pre_b = pre[b].flatten()
                post_b = post[b].flatten()
                outer = torch.outer(pre_b, post_b).flatten()

                # Truncate or pad to fast_weight_dim
                if outer.shape[0] > self.fast_weight_dim:
                    outer = outer[: self.fast_weight_dim]
                elif outer.shape[0] < self.fast_weight_dim:
                    padding = torch.zeros(self.fast_weight_dim - outer.shape[0], device=outer.device, dtype=outer.dtype)
                    outer = torch.cat([outer, padding])

                new_fast_weights[b] = new_fast_weights[b] + self._config.learning_rate * self._config.outer_product_scale * outer

        return {"fast_weights": new_fast_weights}


def create_fast_weight_plasticity(config: PlasticityConfig) -> FastWeightPlasticity:
    """Factory to create FastWeightPlasticity from PlasticityConfig.

    Args:
        config: PlasticityConfig with plasticity_type="fast_weights".

    Returns:
        Configured FastWeightPlasticity instance.

    Raises:
        ValueError: If config is not fast_weights type.
    """
    if config.plasticity_type != "fast_weights":
        raise ValueError(f"Expected fast_weights config, got {config.plasticity_type}")

    fast_weight_dim = config.plastic_state_dims.get("fast_weights", 512) if config.plastic_state_dims else 512
    consolidation = config.consolidation_config or {}

    return FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=consolidation.get("decay", 0.9),
        learning_rate=consolidation.get("learning_rate", 0.1),
        outer_product_scale=consolidation.get("outer_product_scale", 1.0),
    )