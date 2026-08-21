"""Legacy EqProp model migrations — Strangler Fig pattern.

This module provides backward-compatible access to legacy eqprop models
while routing new code through the native 5-D ontology implementations.

The migration follows the Strangler Fig pattern:
- Legacy models remain importable and functional
- New code should use the native ontology classes (LazyStateDynamics, HomeostaticCredit)
- Registry aliases handle transparent redirection
"""

from bioplausible.core.ontology import (
    EnergyMinimizationDynamics,
    HomeostaticCredit,
    LazyStateDynamics,
    StateDynamicsConfig,
    System,
    ThermodynamicContrast,
)
from bioplausible.core.system_trainer import (
    compose_system,
    create_eqprop_system,
)

__all__ = [
    "LegacyEqPropAdapter",
    "create_eqprop_system",
    "get_native_eqprop_system",
    "migrate_to_native",
]


def get_native_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    variant: str = "plain",
) -> System:
    """Create a native 5-D ontology EqProp system.

    This replaces the legacy model instantiation with a composed System
    using the ontology primitives.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based settling
        settle_steps: Maximum settling iterations
        lr: Learning rate for parameter updates
        variant: EqProp variant ("plain", "momentum", "sparse", "feedback", "lazy", "homeostatic")

    Returns:
        A composed System implementing the specified EqProp variant.
    """
    from bioplausible.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        ElasticConsolidationUpdate,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SubstrateConfig,
    )

    # Base components
    substrate = DigitalSubstrate(SubstrateConfig(precision="float32"))

    dims = [hidden_dim] * max(num_layers, 1)
    geometry = RecurrentGeometry(
        GeometryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(dims),
            topology_type="recurrent",
        ),
        hidden_dim=hidden_dim,
    )

    # Select dynamics based on variant
    if variant == "lazy":
        dynamics = LazyStateDynamics(
            StateDynamicsConfig(
                dynamics_type="energy_minimization",
                max_steps=settle_steps,
                beta=beta,
            )
        )
    elif variant == "homeostatic":
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig(
                dynamics_type="energy_minimization",
                max_steps=settle_steps,
                beta=beta,
            )
        )
    else:
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig(
                dynamics_type="energy_minimization",
                max_steps=settle_steps,
                beta=beta,
            )
        )

    # Select credit based on variant
    if variant == "homeostatic":
        credit = HomeostaticCredit(
            CreditAssignmentConfig(
                credit_type="homeostatic",
                beta=beta,
            )
        )
    elif variant == "feedback":
        # For feedback variant, use standard thermodynamic contrast
        # The feedback pathway is handled in the geometry/dynamics
        credit = ThermodynamicContrast(
            CreditAssignmentConfig(
                credit_type="thermodynamic_contrast",
                beta=beta,
            )
        )
    else:
        credit = ThermodynamicContrast(
            CreditAssignmentConfig(
                credit_type="thermodynamic_contrast",
                beta=beta,
            )
        )

    # Select update based on variant
    if variant == "momentum":
        update = RiemannianOrthogonalUpdate(
            ParameterUpdateConfig(
                update_type="riemannian_orthogonal",
                step_size=lr,
                ortho_steps=5,
            )
        )
    else:
        update = EuclideanUpdate(
            ParameterUpdateConfig(
                update_type="euclidean",
                step_size=lr,
            )
        )

    return compose_system(substrate, geometry, dynamics, credit, update)


def migrate_to_native(legacy_model) -> System:
    """Migrate a legacy EqProp model to the native 5-D ontology System.

    Args:
        legacy_model: An instance of a legacy EqProp model (StandardEqProp,
                      DirectedEP, LazyEqProp, HomeostaticEqProp, etc.)

    Returns:
        A System instance that replicates the legacy model's behavior
        using the native ontology primitives.
    """
    # Extract configuration from legacy model
    input_dim = getattr(legacy_model, "input_dim", 32)
    hidden_dim = getattr(legacy_model, "hidden_dim", 32)
    output_dim = getattr(legacy_model, "output_dim", 10)
    num_layers = 1
    if hasattr(legacy_model, "config"):
        config = legacy_model.config
        if hasattr(config, "hidden_dims"):
            num_layers = len(config.hidden_dims)
        elif hasattr(config, "num_layers"):
            num_layers = config.num_layers
    if hasattr(legacy_model, "num_layers"):
        num_layers = legacy_model.num_layers

    beta = getattr(legacy_model, "beta", 0.5)
    max_steps = getattr(legacy_model, "max_steps", 30)
    lr = getattr(legacy_model, "lr", 0.01)

    # Determine variant from model class name
    class_name = legacy_model.__class__.__name__.lower()
    if "lazy" in class_name:
        variant = "lazy"
    elif "homeostatic" in class_name:
        variant = "homeostatic"
    elif "momentum" in class_name:
        variant = "momentum"
    elif "directed" in class_name or "feedback" in class_name:
        variant = "feedback"
    elif "sparse" in class_name:
        variant = "sparse"
    else:
        variant = "plain"

    return get_native_eqprop_system(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        beta=beta,
        settle_steps=max_steps,
        lr=lr,
        variant=variant,
    )


class LegacyEqPropAdapter:
    """Adapter to wrap legacy EqProp models for ontology compatibility.

    This adapter allows legacy models to be used through the 5-D System
    interface while maintaining their original train_step implementation.
    """

    def __init__(self, legacy_model):
        self.legacy_model = legacy_model
        self._native_system: System | None = None

    def to_native_system(self) -> System:
        """Get or create the native System equivalent."""
        if self._native_system is None:
            self._native_system = migrate_to_native(self.legacy_model)
        return self._native_system

    def train_step(self, x, y):
        """Delegate to legacy model's train_step."""
        return self.legacy_model.train_step(x, y)

    def forward(self, x):
        """Delegate to legacy model's forward."""
        return self.legacy_model(x)

    def parameters(self):
        """Delegate to legacy model's parameters."""
        return self.legacy_model.parameters()
