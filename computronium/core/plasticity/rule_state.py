"""Rule State Plasticity (Z3): Frozen-θ algorithm switching via ψ.

Implements the Z3 benchmark: fixed weights (θ frozen), changing algorithm
via ψ-mediated operator selection. The controller learns to select operators
from a library, while θ learns the operator embeddings.

Parameter invariance must be exact: ||θ_after - θ_before|| == 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.joint.transition import PlasticityConfig

if TYPE_CHECKING:
    from computronium.core.joint.context import SystemContext
    from computronium.core.joint.state import CompositeState


@dataclass(frozen=True, slots=True)
class RuleStatePlasticityConfig:
    """Configuration for rule state plasticity dynamics.

    Attributes:
        num_operators: Number of operators in the library.
        operator_dim: Dimension of each operator embedding.
        controller_hidden: Hidden dimension of the controller network.
        temperature: Gumbel-Softmax temperature for operator selection.
        learning_rate: Learning rate for controller and operator embeddings.
        decay: Decay factor for operator logits between steps.
    """

    num_operators: int = 8
    operator_dim: int = 64
    controller_hidden: int = 128
    temperature: float = 1.0
    learning_rate: float = 0.01
    decay: float = 0.99


class RuleStatePlasticity:
    """Rule State Plasticity (Z3): Operator selection via ψ for frozen-θ task switching.

    Maintains a library of operator embeddings and a controller that selects
    operators based on the current state. The key invariant:

    - θ (operator embeddings) is FROZEN during evaluation
    - ψ (controller state, operator logits) ADAPTS during evaluation
    - This enables algorithm migration without weight updates

    Operator library (Z3 minimal set):
        T_0 = Identity
        T_1 = Threshold
        T_2 = Accumulate
        T_3 = LastSymbol
        T_4 = Parity
        T_5 = SparseTopKRoute
        T_6 = SignFlip
        T_7 = Delay

    ψ = (operator_logits, controller_state) where:
        - operator_logits: [batch, num_operators] selection weights
        - controller_state: hidden state of controller RNN

    The plasticity law:
        operator_logits_{t+1} = decay * operator_logits_t + controller(ψ_t, x_t)
        active_operator = softmax(operator_logits)  # training (mixture)
                      = argmax(operator_logits)      # eval (hard selection)
    """

    config: PlasticityConfig

    def __init__(
        self,
        num_operators: int = 8,
        operator_dim: int = 64,
        controller_hidden: int = 128,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
        decay: float = 0.99,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize rule state plasticity.

        Args:
            num_operators: Size of operator library.
            operator_dim: Dimension of each operator embedding.
            controller_hidden: Hidden dimension of controller network.
            temperature: Gumbel-Softmax temperature.
            learning_rate: Learning rate for controller/embeddings.
            decay: Operator logit decay per step.
            device: Device for parameters.
        """
        self._config = RuleStatePlasticityConfig(
            num_operators=num_operators,
            operator_dim=operator_dim,
            controller_hidden=controller_hidden,
            temperature=temperature,
            learning_rate=learning_rate,
            decay=decay,
        )
        self.config = PlasticityConfig.rule_state(num_operators=num_operators)
        self._device = torch.device(device)

        # Operator embeddings (frozen during Z3 eval phase)
        # These are part of θ and should not change during evaluation
        self._operator_embeddings = torch.nn.Parameter(
            torch.randn(num_operators, operator_dim, device=self._device) * 0.02,
            requires_grad=True,  # Trainable during meta-training
        )

        # Controller network: (ψ_t, x_t) -> operator_logits
        # Simple MLP controller for operator selection
        self._controller = torch.nn.Sequential(
            torch.nn.Linear(controller_hidden + operator_dim, controller_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(controller_hidden, num_operators),
        ).to(self._device)

        # Controller hidden state (part of ψ, evolves during episodes)
        self._controller_hidden_dim = controller_hidden

    @property
    def num_operators(self) -> int:
        return self._config.num_operators

    @property
    def operator_dim(self) -> int:
        return self._config.operator_dim

    @property
    def operator_embeddings(self) -> torch.nn.Parameter:
        """Get operator embeddings (part of θ, frozen during eval)."""
        return self._operator_embeddings

    @property
    def controller(self) -> torch.nn.Module:
        """Get controller network."""
        return self._controller

    def initial_psi(
        self, context: SystemContext | None, batch_size: int = 1
    ) -> dict[str, Tensor]:
        """Create initial plastic state.

        Args:
            context: System context (unused, kept for protocol compliance).
            batch_size: Batch size for the plastic state tensors.

        Returns:
            Dict with operator_logits and controller_state initialized to zero.
        """
        return {
            "operator_logits": torch.zeros(
                batch_size, self.num_operators, device=self._device
            ),
            "controller_state": torch.zeros(
                batch_size, self._controller_hidden_dim, device=self._device
            ),
        }

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        """Compute next plastic state.

        Args:
            psi: Current plastic state with operator_logits and controller_state.
            z: Full joint state (activity, plastic, substrate).
            context: Immutable system context.

        Returns:
            Updated plastic state with evolved operator_logits and controller_state.
        """
        operator_logits = psi["operator_logits"]
        controller_state = psi["controller_state"]
        batch_size = operator_logits.shape[0]

        # Decay operator logits
        new_operator_logits = self._config.decay * operator_logits

        # Controller update if input activity available
        if "x" in z.activity:
            x = z.activity["x"]  # [batch, input_dim]

            # Flatten input if needed
            if x.dim() > 2:
                x = x.flatten(1)

            # Controller input: concatenate controller_state with input
            controller_input = torch.cat([controller_state, x], dim=-1)

            # Controller produces operator logits update
            logits_update = self._controller(controller_input)
            new_operator_logits = (
                new_operator_logits + self._config.learning_rate * logits_update
            )

            # Update controller hidden state (simple RNN-like update)
            new_controller_state = torch.tanh(
                controller_state
                + 0.1
                * logits_update.mean(dim=-1, keepdim=True).expand(
                    -1, self._controller_hidden_dim
                )
            )
        else:
            new_controller_state = controller_state * self._config.decay

        return {
            "operator_logits": new_operator_logits,
            "controller_state": new_controller_state,
        }

    def get_active_operator(
        self,
        operator_logits: Tensor,
        is_training: bool = True,
    ) -> Tensor:
        """Get active operator from logits.

        Args:
            operator_logits: [batch, num_operators]
            is_training: If True, use Gumbel-Softmax (differentiable).
                        If False, use hard argmax selection.

        Returns:
            Operator weights [batch, num_operators] (soft or one-hot).
        """
        if is_training:
            # Differentiable: Gumbel-Softmax
            gumbels = -torch.empty_like(operator_logits).exponential_().log()
            gumbels = (operator_logits + gumbels) / self._config.temperature
            return torch.softmax(gumbels, dim=-1)
        else:
            # Evaluation: hard selection
            _, indices = torch.topk(operator_logits, k=1, dim=-1)
            routes = torch.zeros_like(operator_logits)
            routes.scatter_(-1, indices, 1.0)
            return routes

    def apply_operator(
        self,
        x: Tensor,
        operator_weights: Tensor,
    ) -> Tensor:
        """Apply selected operator(s) to input.

        This is a simplified operator application. In practice,
        operators would be more complex transformations.

        Args:
            x: Input tensor [batch, operator_dim]
            operator_weights: Operator selection weights [batch, num_operators]

        Returns:
            Transformed output [batch, operator_dim]
        """
        # Weighted combination of operator embeddings
        # output = sum_k weight_k * (x @ operator_embedding_k.T)
        # This is a simplified linear operator application
        operator_out = torch.einsum(
            "bk,kd->bd", operator_weights, self._operator_embeddings
        )
        return x @ operator_out.T if x.shape[-1] == self.operator_dim else operator_out

    def freeze_theta(self) -> None:
        """Freeze operator embeddings (θ) for Z3 evaluation phase.

        This enforces the exact parameter invariance: ||θ_after - θ_before|| == 0
        """
        self._operator_embeddings.requires_grad_(False)
        for param in self._controller.parameters():
            param.requires_grad_(False)

    def unfreeze_theta(self) -> None:
        """Unfreeze operator embeddings for meta-training phase."""
        self._operator_embeddings.requires_grad_(True)
        for param in self._controller.parameters():
            param.requires_grad_(True)

    def verify_theta_frozen(self) -> bool:
        """Verify that θ (operator embeddings) is frozen.

        Returns:
            True if all theta parameters have requires_grad=False.
        """
        return not self._operator_embeddings.requires_grad and all(
            not p.requires_grad for p in self._controller.parameters()
        )


def create_rule_state_plasticity(config: PlasticityConfig) -> RuleStatePlasticity:
    """Factory to create RuleStatePlasticity from PlasticityConfig.

    Args:
        config: PlasticityConfig with plasticity_type="rule_state".

    Returns:
        Configured RuleStatePlasticity instance.

    Raises:
        ValueError: If config is not rule_state type.
    """
    if config.plasticity_type != "rule_state":
        raise ValueError(f"Expected rule_state config, got {config.plasticity_type}")

    num_operators = (
        config.plastic_state_dims.get("operator_logits", 8)
        if config.plastic_state_dims
        else 8
    )
    consolidation = config.consolidation_config or {}

    return RuleStatePlasticity(
        num_operators=num_operators,
        operator_dim=consolidation.get("operator_dim", 64),
        controller_hidden=consolidation.get("controller_hidden", 128),
        temperature=consolidation.get("temperature", 1.0),
        learning_rate=consolidation.get("learning_rate", 0.01),
        decay=consolidation.get("decay", 0.99),
    )
