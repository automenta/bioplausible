"""Factory functions for composing 6-D JointSystems from configurations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
from computronium.core.plasticity import (
    NullPlasticity as _NullPlasticity,
)
from computronium.core.plasticity import (
    create_fast_weight_plasticity,
    create_routing_plasticity,
    create_rule_state_plasticity,
    create_substrate_coupled_plasticity,
)
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    ElasticConsolidationUpdate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    NaturalGradientUpdate,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    RecurrentGeometry,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    System,
    ThermodynamicContrast,
    substrate_from_config,
)

if TYPE_CHECKING:
    from computronium.core.system_trainer.config import JointSystem
    from computronium.ontology import (
        CreditAssignment,
        Geometry,
        ParameterUpdate,
        PlasticityPrimitive,
        StateDynamics,
        Substrate,
    )


def _credit_from_config(config: CreditAssignmentConfig):
    """Instantiate the credit implementation named by ``config.credit_type``."""
    from computronium.ontology import (
        BackpropCredit,
        HomeostaticCredit,
        LocalGoodnessCredit,
        RandomProjectionsCredit,
        TargetInversionCredit,
        TemporalTraceCredit,
    )

    match config.credit_type.lower():
        case "thermodynamic_contrast" | "equilibrium":
            return ThermodynamicContrast(config)
        case (
            "random_projections" | "feedback_alignment" | ("direct_feedback_alignment")
        ):
            return RandomProjectionsCredit(config)
        case "local_goodness" | "forward_only":
            return LocalGoodnessCredit(config)
        case "temporal_trace" | "spiking":
            return TemporalTraceCredit(config)
        case "target_inversion" | "target_prop":
            return TargetInversionCredit(config)
        case "homeostatic":
            return HomeostaticCredit(config)
        case "gradient" | "backprop":
            return BackpropCredit(config)
        case other:
            raise ValueError(f"Unknown credit_type: {other!r}")


def compose_joint_system[
    TS: Substrate,
    TG: Geometry,
    TD: StateDynamics,
    TP: PlasticityPrimitive,
    TC: CreditAssignment,
    TU: ParameterUpdate,
](
    substrate: TS,
    geometry: TG,
    dynamics: TD,
    plasticity: TP,
    credit: TC,
    update: TU,
) -> JointSystem[TS, TG, TD, TP, TC, TU]:
    """Compose a JointSystem from six orthogonal components.

    This is the primary factory function for creating computronium joint systems
    from the 6-D ontology primitives (S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U).

    The 6-D composition enables combining plasticity mechanisms with any
    credit assignment and update rule, allowing novel architectures like:

    Example 1 - Routing + EqProp (meta-learning credit assignment):
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=RecurrentGeometry(GeometryConfig.recurrent(
                input_dim=784, output_dim=10, hidden_dims=(512, 512, 512)
            ), hidden_dim=512),
            dynamics=EnergyMinimizationDynamics(StateDynamicsConfig.energy_minimization(
                max_steps=20, beta=0.1
            )),
            plasticity=RoutingPlasticity(gate_dim=64, decay=0.99),
            credit=ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast(beta=0.1)),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
        )

    Example 2 - Fast Weights + Backprop (working memory + gradient descent):
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=FeedforwardGeometry(GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256, 128)
            )),
            dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            plasticity=FastWeightPlasticity(fast_weight_dim=512, decay=0.9),
            credit=BackpropCredit(CreditAssignmentConfig.gradient()),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
        )

    Example 3 - Rule State Plasticity for Hebbian meta-learning (Z3 benchmark):
        joint = compose_joint_system(
            substrate=DigitalSubstrate(),
            geometry=FeedforwardGeometry(GeometryConfig.feedforward(
                input_dim=64, output_dim=2, hidden_dims=(128,)
            )),
            dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            plasticity=RuleStatePlasticity(num_operators=8, operator_dim=64),
            credit=LocalGoodnessCredit(CreditAssignmentConfig.local_goodness()),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
        )
        # Freeze theta for Z3 evaluation: joint.plasticity.freeze_theta()

    Args:
        substrate: Physical substrate (digital, analog, memristive, etc.)
        geometry: Network topology and connectivity
        dynamics: State evolution dynamics (settling, spiking, etc.)
        plasticity: Plasticity mechanism (routing, fast weights, rule state, etc.)
        credit: Credit assignment method (thermodynamic contrast, backprop, FA, etc.)
        update: Parameter update rule (SGD, Adam, Muon, spectral, etc.)

    Returns:
        A composed 6-D JointSystem ready for training via train_step().
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class _JointSystem[
        TS: Substrate,
        TG: Geometry,
        TD: StateDynamics,
        TP: PlasticityPrimitive,
        TC: CreditAssignment,
        TU: ParameterUpdate,
    ]:
        substrate: TS
        geometry: TG
        dynamics: TD
        plasticity: TP
        credit: TC
        update: TU

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            """Execute one training step through the family-neutral pipeline."""
            # Preserve lazy ψ initialization side effects before the loop.
            self.plasticity.initial_psi(self.context, batch_size=x.shape[0])
            from computronium.core.pipeline import run_train_step

            return run_train_step(
                self.substrate,
                self.geometry,
                self.dynamics,
                self.credit,
                self.update,
                x,
                y,
            )

        def forward(self, x: Tensor) -> Tensor:
            from computronium.core.pipeline import run_forward

            return run_forward(self.substrate, self.geometry, self.dynamics, x)

        def _make_context(self) -> SystemContext:
            """Create SystemContext from this joint system."""
            from computronium.core.joint.transition import PlasticityConfig
            from computronium.state import StateRegistry, StateVariable, SystemContext

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
            if (
                hasattr(self.plasticity, "config")
                and self.plasticity.config.plastic_state_dims
            ):
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

        @property
        def context(self) -> SystemContext:
            """SystemContext bound to the current θ and component configs."""
            return self._make_context()

        def to_spec(self) -> dict[str, object]:
            """Serialize the JointSystem to a specification dictionary."""
            geometry_dict = dataclasses.asdict(self.geometry.config)
            recurrent_weight = getattr(self.geometry, "_recurrent_weight", None)
            if recurrent_weight is not None:
                geometry_dict["recurrent_weight"] = recurrent_weight.tolist()

            geometry_params = {}
            for name, param in self.geometry.params.items():
                geometry_params[name] = param.tolist()
            geometry_dict["params"] = geometry_params

            return {
                "schema_version": "2.0",  # 6-D schema
                "substrate": dataclasses.asdict(self.substrate.config),
                "geometry": geometry_dict,
                "dynamics": dataclasses.asdict(self.dynamics.config),
                "plasticity": dataclasses.asdict(
                    getattr(self.plasticity, "config", PlasticityConfig.null())
                ),
                "credit": dataclasses.asdict(self.credit.config),
                "update": dataclasses.asdict(self.update.config),
            }

        @classmethod
        def from_spec(cls, spec: dict) -> JointSystem:
            """Reconstruct a JointSystem from a specification dictionary."""
            # Delegate to compose_joint_system_from_configs
            from computronium.core.system_trainer.joint import (
                compose_joint_system_from_configs,
            )

            return compose_joint_system_from_configs(
                SubstrateConfig(**spec["substrate"]),
                GeometryConfig(**spec["geometry"]),
                StateDynamicsConfig(**spec["dynamics"]),
                PlasticityConfig(
                    **spec.get("plasticity", PlasticityConfig.null().__dict__)
                ),
                CreditAssignmentConfig(**spec["credit"]),
                ParameterUpdateConfig(**spec["update"]),
            )

    # Check if plasticity is NullPlasticity (or equivalent)
    if isinstance(plasticity, (_NullPlasticity, NullPlasticity)):
        # For NullPlasticity, we can just use the 5-D system
        from computronium.core.system_trainer.factory import compose_system

        base_system = compose_system(substrate, geometry, dynamics, credit, update)

        # Wrap with a null plasticity interface
        class _NullJointSystem[
            TS: Substrate,
            TG: Geometry,
            TD: StateDynamics,
            TC: CreditAssignment,
            TU: ParameterUpdate,
        ]:
            def __init__(
                self,
                system: System[TS, TG, TD, TC, TU],
            ):
                self._system = system
                self.substrate = system.substrate
                self.geometry = system.geometry
                self.dynamics = system.dynamics
                self.credit = system.credit
                self.update = system.update
                self.plasticity = NullPlasticity()

            def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
                return self._system.train_step(x, y)

            def forward(self, x: Tensor) -> Tensor:
                return self._system.forward(x)

            @property
            def context(self) -> SystemContext:
                """SystemContext bound to the current θ and component configs."""
                return self._make_context()

            def _make_context(self) -> SystemContext:
                from computronium.core.joint.transition import PlasticityConfig
                from computronium.state import (
                    StateRegistry,
                    StateVariable,
                    SystemContext,
                )

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

                # Build configs from components
                substrate_config = self.substrate.config
                geometry_config = self.geometry.config
                dynamics_config = self.dynamics.config
                credit_config = self.credit.config
                update_config = self.update.config
                plasticity_config = PlasticityConfig.null()

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

            def to_spec(self) -> dict[str, object]:
                spec = base_system.to_spec()
                spec["plasticity"] = dataclasses.asdict(PlasticityConfig.null())
                spec["schema_version"] = "2.0"
                return spec

            @classmethod
            def from_spec(cls, spec: dict) -> JointSystem:
                from computronium.core.system_trainer.joint import (
                    compose_joint_system_from_configs,
                )

                return compose_joint_system_from_configs(
                    SubstrateConfig(**spec["substrate"]),
                    GeometryConfig(**spec["geometry"]),
                    StateDynamicsConfig(**spec["dynamics"]),
                    PlasticityConfig(
                        **spec.get("plasticity", PlasticityConfig.null().__dict__)
                    ),
                    CreditAssignmentConfig(**spec["credit"]),
                    ParameterUpdateConfig(**spec["update"]),
                )

        return _NullJointSystem[TS, TG, TD, TC, TU](base_system)  # type: ignore[return-value]

    return _JointSystem[TS, TG, TD, TP, TC, TU](
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        plasticity=plasticity,
        credit=credit,
        update=update,
    )  # type: ignore[return-value]


def compose_joint_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    plasticity: PlasticityConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> JointSystem[
    Substrate,
    Geometry,
    StateDynamics,
    PlasticityPrimitive,
    CreditAssignment,
    ParameterUpdate,
]:
    """Compose a JointSystem from six configuration objects.

    This is the inverse of extract_config(), enabling the round-trip:
    JointSystem --extract_config--> configs --compose_joint_system_from_configs--> JointSystem

    Args:
        substrate: Substrate configuration
        geometry: Geometry configuration
        dynamics: StateDynamics configuration
        plasticity: Plasticity configuration
        credit: CreditAssignment configuration
        update: ParameterUpdate configuration

    Returns:
        A composed JointSystem with default implementations for each layer.
    """
    # Instantiate substrate from config (class named by the explicit type tag)
    substrate_instance = substrate_from_config(substrate)

    # Instantiate geometry from config
    topology_type = geometry.topology_type.lower()
    if topology_type in ("recurrent", "recurrent_attractor"):
        hidden_dim = geometry.hidden_dims[-1] if geometry.hidden_dims else None
        recurrent_weight = None
        if geometry.recurrent_weight is not None:
            recurrent_weight = torch.tensor(geometry.recurrent_weight)
        geometry_instance = RecurrentGeometry(
            geometry, hidden_dim=hidden_dim, recurrent_weight=recurrent_weight
        )
    elif topology_type in ("tile_mesh", "tile"):
        from computronium.ontology import TileGeometry

        geometry_instance = TileGeometry(
            geometry,
            neurons_per_tile=8,
            tiles_per_layer=2,
        )
    else:
        geometry_instance = FeedforwardGeometry(geometry)

    # Instantiate dynamics from config
    dynamics_type = dynamics.dynamics_type.lower()
    if dynamics_type == "energy_minimization":
        dynamics_instance = EnergyMinimizationDynamics(dynamics)
    elif dynamics_type == "predictive_settling":
        dynamics_instance = PredictiveSettlingDynamics(dynamics)
    elif dynamics_type == "spike_integration":
        dynamics_instance = SpikeIntegrationDynamics(dynamics)
    else:
        dynamics_instance = InstantaneousDynamics(dynamics)

    # Instantiate credit from config
    credit_instance = _credit_from_config(credit)

    # Instantiate update from config
    update_type = update.update_type.lower()
    if update_type in ("riemannian_orthogonal", "muon"):
        update_instance = RiemannianOrthogonalUpdate(update)
    elif update_type in ("spectral_constrained", "spectral"):
        update_instance = SpectralConstrainedUpdate(update)
    elif update_type in ("natural_gradient", "fisher"):
        update_instance = NaturalGradientUpdate(update)
    elif update_type in ("elastic_consolidation", "ewc"):
        update_instance = ElasticConsolidationUpdate(update)
    else:
        update_instance = EuclideanUpdate(update)

    # Instantiate plasticity from config
    plasticity_type = plasticity.plasticity_type.lower()
    if plasticity_type == "routing":
        plasticity_instance = create_routing_plasticity(plasticity)
    elif plasticity_type == "fast_weights":
        plasticity_instance = create_fast_weight_plasticity(plasticity)
    elif plasticity_type == "substrate_coupled":
        plasticity_instance = create_substrate_coupled_plasticity(plasticity)
    elif plasticity_type == "rule_state":
        plasticity_instance = create_rule_state_plasticity(plasticity)
    else:
        plasticity_instance = NullPlasticity()

    return compose_joint_system(
        substrate_instance,
        geometry_instance,
        dynamics_instance,
        plasticity_instance,
        credit_instance,
        update_instance,
    )


# Convenience factory for common joint compositions
def create_routing_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    gate_dim: int = 64,
    gate_init_scale: float = 0.1,
) -> JointSystem:
    """Create an EqProp system with RoutingPlasticity (6-D coordinate)."""
    from computronium.core.plasticity import RoutingPlasticity
    from computronium.ontology import (
        CreditAssignmentConfig,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
    )

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers, 1)
    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
        init_scale=0.1,
    )
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    plasticity = RoutingPlasticity(
        gate_dim=gate_dim,
        temperature=1.0,
        decay=0.99,
        learning_rate=0.01,
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


def create_fast_weight_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    fast_weight_dim: int = 512,
) -> JointSystem:
    """Create an EqProp system with FastWeightPlasticity (6-D coordinate)."""
    from computronium.core.plasticity import (
        FastWeightPlasticity,
    )
    from computronium.ontology import (
        CreditAssignmentConfig,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
    )

    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        )
    )

    dims = [hidden_dim] * max(num_layers, 1)
    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(dims),
        init_scale=0.1,
    )
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=0.99,
        learning_rate=0.1,
    )

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


__all__ = [
    "compose_joint_system",
    "compose_joint_system_from_configs",
    "create_routing_eqprop_system",
    "create_fast_weight_eqprop_system",
]
