"""Factory functions for composing 5-D systems from configurations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import Tensor

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
    from computronium.ontology import (
        CreditAssignment,
        Geometry,
        ParameterUpdate,
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
        ThermodynamicContrast,
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


def compose_system[
    TS: Substrate,
    TG: Geometry,
    TD: StateDynamics,
    TC: CreditAssignment,
    TU: ParameterUpdate,
](
    substrate: TS,
    geometry: TG,
    dynamics: TD,
    credit: TC,
    update: TU,
) -> System[TS, TG, TD, TC, TU]:
    """Compose a System from five orthogonal components.

    This is the primary factory function for creating computronium systems
    from the 5-D ontology primitives.

    Example:
        system = compose_system(
            substrate=DigitalSubstrate(),
            geometry=FeedforwardGeometry(GeometryConfig(input_dim=784, output_dim=10)),
            dynamics=InstantaneousDynamics(),
            credit=ThermodynamicContrast(),
            update=EuclideanUpdate(),
        )
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class _ComposedSystem[
        TS: Substrate,
        TG: Geometry,
        TD: StateDynamics,
        TC: CreditAssignment,
        TU: ParameterUpdate,
    ]:
        substrate: TS
        geometry: TG
        dynamics: TD
        credit: TC
        update: TU

        def to_spec(self) -> dict:
            """Serialize the System to a specification dictionary.

            Returns:
                Dictionary containing schema_version and all 5 axis configs.
            """
            geometry_dict = dataclasses.asdict(self.geometry.config)
            # Include recurrent_weight from geometry if present (runtime state)
            recurrent_weight = getattr(self.geometry, "_recurrent_weight", None)
            if recurrent_weight is not None:
                geometry_dict["recurrent_weight"] = recurrent_weight.tolist()

            # Include all geometry parameters for exact round-trip
            geometry_params = {}
            for name, param in self.geometry.params.items():
                geometry_params[name] = param.tolist()
            geometry_dict["params"] = geometry_params

            return {
                "schema_version": "1.0",
                "substrate": dataclasses.asdict(self.substrate.config),
                "geometry": geometry_dict,
                "dynamics": dataclasses.asdict(self.dynamics.config),
                "credit": dataclasses.asdict(self.credit.config),
                "update": dataclasses.asdict(self.update.config),
            }

        @classmethod
        def from_spec(cls, spec: dict) -> System:
            """Reconstruct a System from a specification dictionary.

            Args:
                spec: Dictionary with schema_version and 5 axis configs.

            Returns:
                A composed System instance.
            """
            if spec.get("schema_version") != "1.0":
                raise ValueError(
                    f"Unsupported schema version: {spec.get('schema_version')}"
                )

            from computronium.ontology import (
                CreditAssignmentConfig,
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
            )

            # Reconstruct substrate (class named by the explicit type tag)
            substrate_cfg = SubstrateConfig(**spec["substrate"])
            substrate = substrate_from_config(substrate_cfg)

            # Reconstructed geometry
            geometry_dict = spec["geometry"]
            serialized_params = geometry_dict.pop("params", None)
            # JSON serialization converts tuples to lists; restore tuple types
            if "hidden_dims" in geometry_dict and isinstance(
                geometry_dict["hidden_dims"], list
            ):
                geometry_dict["hidden_dims"] = tuple(geometry_dict["hidden_dims"])
            geometry_cfg = GeometryConfig(**geometry_dict)
            topology_type = geometry_cfg.topology_type.lower()
            if topology_type in ("recurrent", "recurrent_attractor"):
                hidden_dim = (
                    geometry_cfg.hidden_dims[-1] if geometry_cfg.hidden_dims else None
                )
                recurrent_weight = None
                if geometry_cfg.recurrent_weight is not None:
                    recurrent_weight = torch.tensor(geometry_cfg.recurrent_weight)
                geometry = RecurrentGeometry(
                    geometry_cfg,
                    hidden_dim=hidden_dim,
                    recurrent_weight=recurrent_weight,
                )
            elif topology_type in ("tile_mesh", "tile"):
                from computronium.ontology import TileGeometry

                geometry = TileGeometry(
                    geometry_cfg,
                    neurons_per_tile=8,
                    tiles_per_layer=2,
                )
            else:
                geometry = FeedforwardGeometry(geometry_cfg)

            # Restore serialized parameters for exact round-trip
            if serialized_params is not None:
                geometry_params = {
                    k: torch.tensor(v) for k, v in serialized_params.items()
                }
                geometry.update_params(geometry_params)

            # Reconstruct dynamics
            dynamics_cfg = StateDynamicsConfig(**spec["dynamics"])
            dynamics_type = dynamics_cfg.dynamics_type.lower()
            if dynamics_type == "energy_minimization":
                dynamics = EnergyMinimizationDynamics(dynamics_cfg)
            elif dynamics_type == "predictive_settling":
                dynamics = PredictiveSettlingDynamics(dynamics_cfg)
            elif dynamics_type == "spike_integration":
                dynamics = SpikeIntegrationDynamics(dynamics_cfg)
            else:
                dynamics = InstantaneousDynamics(dynamics_cfg)

            # Reconstruct credit
            credit_cfg = CreditAssignmentConfig(**spec["credit"])
            credit = _credit_from_config(credit_cfg)

            # Reconstruct update
            update_cfg = ParameterUpdateConfig(**spec["update"])
            update_type = update_cfg.update_type.lower()
            if update_type in ("riemannian_orthogonal", "muon"):
                update = RiemannianOrthogonalUpdate(update_cfg)
            elif update_type in ("spectral_constrained", "spectral"):
                update = SpectralConstrainedUpdate(update_cfg)
            elif update_type in ("natural_gradient", "fisher"):
                update = NaturalGradientUpdate(update_cfg)
            elif update_type in ("elastic_consolidation", "ewc"):
                update = ElasticConsolidationUpdate(update_cfg)
            else:
                update = EuclideanUpdate(update_cfg)

            return compose_system(substrate, geometry, dynamics, credit, update)

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            """Execute one training step through the family-neutral pipeline."""
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

    return _ComposedSystem[TS, TG, TD, TC, TU](
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=update,
    )  # type: ignore[return-value]


# Convenience factory for common compositions
def create_eqprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    update_momentum: float = 0.9,
) -> System:
    """Create an Equilibrium Propagation system (classic EqProp coordinate)."""
    from computronium.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
        ThermodynamicContrast,
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
    geometry = RecurrentGeometry(
        geometry_cfg,
        hidden_dim=hidden_dim,
    )

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
            momentum=update_momentum,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_backprop_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
) -> System:
    """Create a standard Backprop system (baseline coordinate)."""
    from computronium.ontology import (
        BackpropCredit,
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
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

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(
        GeometryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(dims),
            num_layers=num_layers,
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = BackpropCredit(
        CreditAssignmentConfig(
            credit_type="gradient",
            beta=0.5,
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

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_fa_system(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.1,
) -> System:
    """Create a Feedback Alignment system."""
    from computronium.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        ParameterUpdateConfig,
        RandomProjectionsCredit,
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

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(
        GeometryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(dims),
            num_layers=num_layers,
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = RandomProjectionsCredit(
        CreditAssignmentConfig(
            credit_type="random_projections",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=feedback_scale,
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

    return compose_system(substrate, geometry, dynamics, credit, update)


def extract_config(system: System) -> dict[str, object]:
    """Extract configuration from a composed System.

    Returns a dictionary mapping layer names to their configuration objects.
    This enables round-trip: System -> configs -> System.
    """
    return {
        "substrate": system.substrate.config,
        "geometry": system.geometry.config,
        "dynamics": system.dynamics.config,
        "credit": system.credit.config,
        "update": system.update.config,
    }


def compose_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> System:
    """Compose a System from five configuration objects.

    This is the inverse of extract_config(), enabling the round-trip:
    System --extract_config--> configs --compose_system_from_configs--> System

    Args:
        substrate: Substrate configuration
        geometry: Geometry configuration
        dynamics: StateDynamics configuration
        credit: CreditAssignment configuration
        update: ParameterUpdate configuration

    Returns:
        A composed System with default implementations for each layer.
    """
    from computronium.ontology import (
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        InstantaneousDynamics,
        NaturalGradientUpdate,
        PredictiveSettlingDynamics,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SpectralConstrainedUpdate,
        SpikeIntegrationDynamics,
    )

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

    return compose_system(
        substrate_instance,
        geometry_instance,
        dynamics_instance,
        credit_instance,
        update_instance,
    )


__all__ = [
    "compose_system",
    "compose_system_from_configs",
    "create_backprop_system",
    "create_eqprop_system",
    "create_fa_system",
    "extract_config",
]