"""Routing Plasticity: State-dependent pathway gating and sparse routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.joint.transition import PlasticityConfig

if TYPE_CHECKING:
    from computronium.state import CompositeState, SystemContext


@dataclass(frozen=True, slots=True)
class RoutingPlasticityConfig:
    """Configuration for routing plasticity dynamics.

    Attributes:
        gate_dim: Dimension of gate logits and active routes.
        temperature: Gumbel-Softmax temperature for differentiable routing.
        top_k: Number of routes to activate (None = all above threshold).
        decay: Decay factor for gate logits between steps.
        learning_rate: Learning rate for gate logit updates.
    """

    gate_dim: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    decay: float = 0.99
    learning_rate: float = 0.01


class RoutingPlasticity:
    """Routing plasticity: state-dependent pathway gating.

    Maintains gate logits that control which pathways are active.
    Uses differentiable routing (Gumbel-Softmax) during training,
    hard selection at evaluation.

    ψ = (gate_logits, active_routes) where:
    - gate_logits: learnable gating parameters [batch, gate_dim]
    - active_routes: binary/soft mask of active pathways [batch, gate_dim]

    The plasticity law:
        gate_logits_{t+1} = decay * gate_logits_t + lr * f(activity_t)
        active_routes = GumbelSoftmax(gate_logits, temperature)  # training
                    = top_k(gate_logits)                         # eval
    """

    config: PlasticityConfig

    def __init__(
        self,
        gate_dim: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        decay: float = 0.99,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize routing plasticity.

        Args:
            gate_dim: Number of routing gates.
            temperature: Gumbel-Softmax temperature.
            top_k: Number of routes to keep active (None = threshold-based).
            decay: Gate logit decay per step.
            learning_rate: Gate logit update learning rate.
        """
        self._config = RoutingPlasticityConfig(
            gate_dim=gate_dim,
            temperature=temperature,
            top_k=top_k,
            decay=decay,
            learning_rate=learning_rate,
        )
        self.config = PlasticityConfig.routing(gate_dim=gate_dim)

    @property
    def gate_dim(self) -> int:
        return self._config.gate_dim

    def initial_psi(
        self, context: SystemContext | None, batch_size: int = 1
    ) -> dict[str, Tensor]:
        """Create initial plastic state.

        Args:
            context: System context supplying the target device (unused otherwise).
            batch_size: Batch size for the plastic state tensors.

        Returns:
            Dict with gate_logits and active_routes initialized to zero.
        """
        device = context.device if context is not None else None
        return {
            "gate_logits": torch.zeros(batch_size, self.gate_dim, device=device),
            "active_routes": torch.zeros(batch_size, self.gate_dim, device=device),
        }

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        """Compute next plastic state.

        Args:
            psi: Current plastic state with gate_logits and active_routes.
            z: Full joint state (activity, plastic, substrate).
            context: Immutable system context.

        Returns:
            Updated plastic state.
        """
        gate_logits = psi["gate_logits"]
        batch_size = gate_logits.shape[0]

        # Decay gate logits
        new_gate_logits = self._config.decay * gate_logits

        # Update based on activity if available
        if "x" in z.activity:
            x = z.activity["x"]  # [batch, input_dim]
            # Handle batch size mismatch
            if x.shape[0] != batch_size:
                # Expand or truncate gate logits to match x
                if x.shape[0] > batch_size:
                    new_gate_logits = new_gate_logits.expand(x.shape[0], -1)
                else:
                    new_gate_logits = new_gate_logits[: x.shape[0]]
                batch_size = x.shape[0]

            # Compute gate update from input statistics
            gate_drive = x.abs().mean(dim=1, keepdim=True)  # [batch, 1]
            gate_drive = gate_drive.expand(-1, self.gate_dim)
            new_gate_logits = new_gate_logits + self._config.learning_rate * gate_drive

        # Compute active routes
        # Use training mode if context has theta with requires_grad
        is_training = False
        if context is not None:
            is_training = any(p.requires_grad for p in context.theta.values())
        if is_training:
            # Training: differentiable Gumbel-Softmax
            active_routes = self._gumbel_softmax(new_gate_logits)
        else:
            # Evaluation: hard top-k or threshold
            active_routes = self._hard_select(new_gate_logits)

        return {
            "gate_logits": new_gate_logits,
            "active_routes": active_routes,
        }

    def _gumbel_softmax(self, logits: Tensor) -> Tensor:
        """Differentiable routing via Gumbel-Softmax."""
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self._config.temperature
        return torch.softmax(gumbels, dim=-1)

    def _hard_select(self, logits: Tensor) -> Tensor:
        """Hard route selection for evaluation."""
        if self._config.top_k is not None:
            # Top-k selection
            _, indices = torch.topk(logits, k=self._config.top_k, dim=-1)
            routes = torch.zeros_like(logits)
            routes.scatter_(-1, indices, 1.0)
            return routes
        else:
            # Threshold-based (sigmoid > 0.5)
            return (torch.sigmoid(logits) > 0.5).float()

    def modulate(
        self, activations: list[Tensor] | Tensor, psi: dict[str, Tensor]
    ) -> list[Tensor] | Tensor:
        """Apply routing gates to activations.

        Per-sample scalar gate strength applied to hidden/output layers.
        """
        gate_logits = psi.get("gate_logits")
        if gate_logits is None:
            return activations

        # Per-sample gate strength: mean sigmoid across gates
        gate_strength = torch.sigmoid(gate_logits).mean(
            dim=-1, keepdim=True
        )  # [batch, 1]

        acts = activations if isinstance(activations, list) else [activations]
        modulated = []
        for a in acts:
            # Apply gate strength to batch dimension
            if a.shape[0] == gate_strength.shape[0]:
                modulated.append(a * gate_strength)
            else:
                modulated.append(a)
        return modulated if isinstance(activations, list) else modulated[0]


def create_routing_plasticity(config: PlasticityConfig) -> RoutingPlasticity:
    """Factory to create RoutingPlasticity from PlasticityConfig.

    Args:
        config: PlasticityConfig with plasticity_type="routing".

    Returns:
        Configured RoutingPlasticity instance.

    Raises:
        ValueError: If config is not routing type.
    """
    if config.plasticity_type != "routing":
        raise ValueError(f"Expected routing config, got {config.plasticity_type}")

    gate_dim = (
        config.plastic_state_dims.get("gate_logits", 64)
        if config.plastic_state_dims
        else 64
    )
    consolidation = config.consolidation_config or {}

    return RoutingPlasticity(
        gate_dim=gate_dim,
        temperature=consolidation.get("temperature", 1.0),
        top_k=consolidation.get("top_k"),
        decay=consolidation.get("decay", 0.99),
        learning_rate=consolidation.get("learning_rate", 0.01),
    )
