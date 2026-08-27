"""ContinualJointSystem: Joint system adapted for continual learning."""

from __future__ import annotations

from torch import Tensor, nn

from computronium.core.joint.context import SystemContext
from computronium.core.joint.transition import PlasticityPrimitive
from computronium.core.ontology import (
    CreditAssignment,
    Geometry,
    ParameterUpdate,
    StateDynamics,
    Substrate,
)


class ContinualJointSystem(nn.Module):
    """Joint system adapted for continual learning with 10-class output.

    Uses task masking instead of task-specific heads. The joint system
    outputs 10-class logits (matching MNIST 10 digits), and we mask
    the loss to the current task's 2 classes.

    Maintains plastic state (ψ) across training steps for ψ/θ decoupling.
    """

    def __init__(
        self,
        substrate: Substrate | object = None,
        geometry: Geometry | object = None,
        dynamics: StateDynamics | object = None,
        credit: CreditAssignment | object = None,
        update: ParameterUpdate | object = None,
        plasticity: PlasticityPrimitive | object = None,
    ):
        super().__init__()
        # Backward compat: if first arg is a joint_system object, extract components
        if geometry is None and hasattr(substrate, "substrate"):
            joint_system = substrate
            self.substrate = joint_system.substrate
            self.geometry = joint_system.geometry
            self.dynamics = joint_system.dynamics
            self.credit = joint_system.credit
            self.update = joint_system.update
            self.plasticity = joint_system.plasticity
        else:
            self.substrate = substrate
            self.geometry = geometry
            self.dynamics = dynamics
            self.credit = credit
            self.update = update
            self.plasticity = plasticity

        # Register geometry as submodule so .to(device) works
        self.register_module("geometry", geometry)

        # Current task
        self.current_task = 0

        # Plastic state (ψ) - maintained across steps for fast weights
        self._psi: dict[str, Tensor] | None = None

    @classmethod
    def from_joint_system(cls, joint_system) -> ContinualJointSystem:
        """Create from a composed JointSystem."""
        return cls(
            substrate=joint_system.substrate,
            geometry=joint_system.geometry,
            dynamics=joint_system.dynamics,
            credit=joint_system.credit,
            update=joint_system.update,
            plasticity=joint_system.plasticity,
        )

    @property
    def joint_system(self):
        """Backward compatibility: return a joint-system-like object."""
        from types import SimpleNamespace
        return SimpleNamespace(
            substrate=self.substrate,
            geometry=self.geometry,
            dynamics=self.dynamics,
            credit=self.credit,
            update=self.update,
            plasticity=self.plasticity,
            context=self.context,
        )

    def copy(self):
        """Create a copy of this ContinualJointSystem (for LwF previous model)."""
        # Create a new instance with the same components
        new_model = self.__class__(
            substrate=self.substrate,
            geometry=self.geometry,
            dynamics=self.dynamics,
            credit=self.credit,
            update=self.update,
            plasticity=self.plasticity,
        )
        # Copy parameters and state
        new_model.load_state_dict(self.state_dict())
        new_model.current_task = self.current_task
        if self._psi is not None:
            new_model._psi = {k: v.clone() for k, v in self._psi.items()}
        return new_model

    def __deepcopy__(self, memo):
        """Support copy.deepcopy."""
        return self.copy()

    def to(self, *args, **kwargs):
        """Override to ensure joint system components are moved to device."""
        self = super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device")
        if device is not None:
            if hasattr(self.substrate, "to"):
                self.substrate.to(device)
            if hasattr(self.plasticity, "to"):
                self.plasticity.to(device)
            if hasattr(self.credit, "to"):
                self.credit.to(device)
            if hasattr(self.update, "to"):
                self.update.to(device)
            if hasattr(self.dynamics, "to"):
                self.dynamics.to(device)
        return self

    @property
    def context(self) -> SystemContext:
        """SystemContext bound to the current θ and component configs."""
        return self._make_context()

    def _make_context(self) -> SystemContext:
        """Create SystemContext from this joint system."""
        from computronium.core.joint.context import SystemContext
        from computronium.core.joint.state import StateRegistry, StateVariable
        from computronium.core.joint.transition import PlasticityConfig

        # Build registry from all components
        registry = StateRegistry()

        # Register persistent parameters (theta)
        for name, param in self.geometry.params.items():
            registry.register(
                StateVariable(
                    name=name,
                    persistent=True,
                    fast_plastic=False,
                    consolidatable=False,
                )
            )

        # Register plastic variables if any
        if hasattr(self.plasticity, "config") and self.plasticity.config.plastic_state_dims:
            for name, dim in self.plasticity.config.plastic_state_dims.items():
                registry.register(
                    StateVariable(
                        name=name,
                        persistent=False,
                        fast_plastic=True,
                        consolidatable=True,
                    )
                )

        # Build configs from components
        substrate_config = self.substrate.config
        geometry_config = self.geometry.config
        dynamics_config = self.dynamics.config
        credit_config = self.credit.config
        update_config = self.update.config
        plasticity_config = getattr(
            self.plasticity, "config", PlasticityConfig.null()
        )

        return SystemContext(
            theta=self.geometry.params,
            geometry=self.geometry,
            substrate=self.substrate,
            substrate_config=substrate_config,
            geometry_config=geometry_config,
            dynamics_config=dynamics_config,
            credit_config=credit_config,
            update_config=update_config,
            plasticity_config=plasticity_config,
            registry=registry,
        )

    def forward(self, x: Tensor, task_id: int | None = None) -> Tensor:
        """Forward pass through joint system with plastic state modulation.

        For FastWeightPlasticity, modulates the last hidden layer with fast weights.
        Returns full 10-class logits.
        """
        # Check if we have plastic state and fast weight plasticity
        plasticity = self.plasticity
        has_fast_weights = (
            self._psi is not None
            and "fast_weights" in self._psi
            and hasattr(plasticity, "fast_weight_dim")
        )

        if not has_fast_weights:
            return self._joint_forward(x)

        # Get intermediate activations from geometry
        substrate = self.substrate
        geometry = self.geometry
        acts = geometry.forward_with_intermediates(x, substrate)
        # acts: [input, hidden1, hidden2, ..., output]

        # Modulate last hidden layer with fast weights
        # Last hidden is acts[-2] (before output layer)
        if len(acts) >= 2:
            last_hidden = acts[-2]  # [batch, hidden_dim]
            fast_weights = self._psi["fast_weights"]  # [batch, fast_weight_dim]

            # Handle batch size mismatch - resize to current batch
            batch_size = x.shape[0]
            if fast_weights.shape[0] != batch_size:
                if fast_weights.shape[0] == 1:
                    fast_weights = fast_weights.expand(batch_size, -1)
                elif fast_weights.shape[0] > batch_size:
                    fast_weights = fast_weights[:batch_size]
                else:
                    # Cannot expand smaller batch to larger - fallback to standard forward
                    return self._joint_forward(x)

            # Project fast weights to hidden_dim and add
            # Need a projection layer - create if not exists
            if not hasattr(self, "_fast_weight_proj"):
                hidden_dim = last_hidden.shape[-1]
                self._fast_weight_proj = nn.Linear(
                    plasticity.fast_weight_dim, hidden_dim, bias=False
                ).to(x.device)
                # Initialize with small weights
                nn.init.normal_(self._fast_weight_proj.weight, std=0.01)

            modulation = self._fast_weight_proj(fast_weights)
            modulated_hidden = last_hidden + modulation

            # Apply output layer (last layer in geometry)
            # The output layer is the last Linear layer in geometry._layers
            output_layer = None
            for layer in reversed(geometry._layers):
                if isinstance(layer, nn.Linear):
                    output_layer = layer
                    break

            if output_layer is not None:
                logits = output_layer(modulated_hidden)
                return logits

        # Fallback to standard forward
        return self._joint_forward(x)

    def _joint_forward(self, x: Tensor) -> Tensor:
        """Standard forward through joint system."""
        from computronium.core.pipeline import run_forward

        return run_forward(self.substrate, self.geometry, self.dynamics, x)

    def train_step(self, x: Tensor, y: Tensor, task_id: int | None = None) -> dict[str, float]:
        """Training step using joint system's pipeline with task-masked loss and plasticity stepping."""
        task_id = task_id if task_id is not None else self.current_task
        from computronium.core.continual.training import run_continual_train_step

        metrics, self._psi = run_continual_train_step(self, x, y, task_id, self._psi)
        return metrics

    def set_task(self, task_id: int) -> None:
        self.current_task = task_id

    def reset_plastic_state(self) -> None:
        """Reset plastic state (e.g., at task boundary for new episode)."""
        self._psi = None


__all__ = ["ContinualJointSystem"]
