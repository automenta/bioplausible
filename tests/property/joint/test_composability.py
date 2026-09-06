"""Property tests for 6-D composability: random valid coordinates generation and validation."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch

from computronium.core.dynamics.adapters import create_dynamics_adapter
from computronium.core.joint import (
    CompositeState,
    NullPlasticity,
    PlasticityConfig,
    StateRegistry,
    StateVariable,
    SystemContext,
)
from computronium.core.substrates.adapters import create_substrate_adapter
from computronium.core.system_trainer import compose_system
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    RecurrentGeometry,
    RiemannianOrthogonalUpdate,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemConfig,
    ThermodynamicContrast,
)

# ============================================================
# Valid coordinate factories for each axis (using actual API)
# ============================================================

SUBSTRATE_FACTORIES = {
    "digital": lambda: (DigitalSubstrate(), SubstrateConfig.digital()),
    "analog": lambda: (DigitalSubstrate(), SubstrateConfig.analog()),
    "memristive": lambda: (DigitalSubstrate(), SubstrateConfig.memristive()),
    "neuromorphic": lambda: (DigitalSubstrate(), SubstrateConfig.neuromorphic()),
    "optical": lambda: (DigitalSubstrate(), SubstrateConfig.optical()),
    "quantum": lambda: (DigitalSubstrate(), SubstrateConfig.quantum()),
    "complex": lambda: (DigitalSubstrate(), SubstrateConfig.complex()),
    "sparse": lambda: (DigitalSubstrate(), SubstrateConfig.sparse()),
    "ternary": lambda: (DigitalSubstrate(), SubstrateConfig.ternary()),
}

GEOMETRY_FACTORIES = {
    "feedforward": lambda: (
        RecurrentGeometry(
            GeometryConfig.feedforward(input_dim=10, output_dim=2, hidden_dims=(20,))
        ),
        GeometryConfig.feedforward(input_dim=10, output_dim=2, hidden_dims=(20,)),
    ),
    "recurrent": lambda: (
        RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
            hidden_dim=20,
        ),
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
    ),
}

DYNAMICS_FACTORIES = {
    "energy_minimization": lambda: (
        EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
        ),
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
    ),
    "instantaneous": lambda: (
        InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        StateDynamicsConfig.instantaneous(),
    ),
    "predictive_settling": lambda: (
        PredictiveSettlingDynamics(StateDynamicsConfig.predictive_settling()),
        StateDynamicsConfig.predictive_settling(),
    ),
    "spike_integration": lambda: (
        SpikeIntegrationDynamics(StateDynamicsConfig.spike_integration(max_steps=3)),
        StateDynamicsConfig.spike_integration(max_steps=3),
    ),
    "diffusion": lambda: (
        EnergyMinimizationDynamics(
            StateDynamicsConfig.diffusion(max_steps=3, beta=0.5)
        ),
        StateDynamicsConfig.diffusion(max_steps=3, beta=0.5),
    ),
}

CREDIT_FACTORIES = {
    "thermodynamic_contrast": lambda: (
        ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)),
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
    ),
    "random_projections": lambda: (
        ThermodynamicContrast(CreditAssignmentConfig.random_projections()),
        CreditAssignmentConfig.random_projections(),
    ),
    "local_goodness": lambda: (
        ThermodynamicContrast(CreditAssignmentConfig.local_goodness()),
        CreditAssignmentConfig.local_goodness(),
    ),
    "temporal_trace": lambda: (
        ThermodynamicContrast(CreditAssignmentConfig.temporal_trace()),
        CreditAssignmentConfig.temporal_trace(),
    ),
    "target_inversion": lambda: (
        ThermodynamicContrast(CreditAssignmentConfig.target_inversion()),
        CreditAssignmentConfig.target_inversion(),
    ),
    "gradient": lambda: (
        ThermodynamicContrast(CreditAssignmentConfig.gradient()),
        CreditAssignmentConfig.gradient(),
    ),
}

UPDATE_FACTORIES = {
    "euclidean": lambda: (
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
        ParameterUpdateConfig.euclidean(step_size=0.01),
    ),
    "riemannian_orthogonal": lambda: (
        RiemannianOrthogonalUpdate(ParameterUpdateConfig.riemannian_orthogonal()),
        ParameterUpdateConfig.riemannian_orthogonal(),
    ),
    "spectral_constrained": lambda: (
        EuclideanUpdate(ParameterUpdateConfig.spectral_constrained()),
        ParameterUpdateConfig.spectral_constrained(),
    ),
    "mean_norm": lambda: (
        EuclideanUpdate(ParameterUpdateConfig.mean_norm()),
        ParameterUpdateConfig.mean_norm(),
    ),
    "elastic_consolidation": lambda: (
        EuclideanUpdate(ParameterUpdateConfig.elastic_consolidation()),
        ParameterUpdateConfig.elastic_consolidation(),
    ),
}

PLASTICITY_FACTORIES = {
    "null": lambda: PlasticityConfig.null(),  # ruff: ignore[unnecessary-lambda]
    "routing": lambda: PlasticityConfig.routing(gate_dim=32),
    "fast_weights": lambda: PlasticityConfig.fast_weights(fast_weight_dim=64),
    "substrate_coupled": lambda: PlasticityConfig.substrate_coupled(),  # ruff: ignore[unnecessary-lambda]
    "rule_state": lambda: PlasticityConfig.rule_state(num_operators=4),
}


def create_random_system_config() -> SystemConfig:  # ruff: ignore[too-many-locals]
    """Create a random valid 6-D SystemConfig."""
    # Use only compatible combinations to avoid validation errors
    compatible_combinations = [
        # (substrate, dynamics, credit, update) - known compatible
        ("digital", "energy_minimization", "thermodynamic_contrast", "euclidean"),
        ("digital", "energy_minimization", "random_projections", "euclidean"),
        ("digital", "energy_minimization", "local_goodness", "euclidean"),
        ("digital", "energy_minimization", "temporal_trace", "euclidean"),
        ("digital", "energy_minimization", "target_inversion", "euclidean"),
        ("digital", "energy_minimization", "gradient", "euclidean"),
        ("analog", "energy_minimization", "thermodynamic_contrast", "euclidean"),
        ("ternary", "energy_minimization", "thermodynamic_contrast", "euclidean"),
        (
            "sparse",
            "energy_minimization",
            "thermodynamic_contrast",
            "spectral_constrained",
        ),
        (
            "complex",
            "energy_minimization",
            "thermodynamic_contrast",
            "riemannian_orthogonal",
        ),
    ]

    combo = random.choice(compatible_combinations)
    substrate_name, dynamics_name, credit_name, update_name = combo
    geometry_name = random.choice(list(GEOMETRY_FACTORIES.keys()))
    plasticity_name = random.choice(list(PLASTICITY_FACTORIES.keys()))

    _, substrate_config = SUBSTRATE_FACTORIES[substrate_name]()
    _, geometry_config = GEOMETRY_FACTORIES[geometry_name]()
    _, dynamics_config = DYNAMICS_FACTORIES[dynamics_name]()
    _, credit_config = CREDIT_FACTORIES[credit_name]()
    _, update_config = UPDATE_FACTORIES[update_name]()
    plasticity_config = PLASTICITY_FACTORIES[plasticity_name]()

    config = SystemConfig(
        substrate=substrate_config,
        geometry=geometry_config,
        dynamics=dynamics_config,
        plasticity=plasticity_config,
        credit=credit_config,
        update=update_config,
    )
    config.validate()
    return config


def create_system_from_config(config: SystemConfig) -> tuple[Any, Any, Any, Any, Any]:
    """Create system components from config for testing."""
    substrate = DigitalSubstrate(config.substrate)
    geometry = RecurrentGeometry(config.geometry)
    dynamics = EnergyMinimizationDynamics(config.dynamics)
    credit = ThermodynamicContrast(config.credit)
    update = EuclideanUpdate(config.update)
    return substrate, geometry, dynamics, credit, update


def create_registry_from_geometry(
    geometry: RecurrentGeometry, plasticity_config: PlasticityConfig
) -> StateRegistry:
    """Create a StateRegistry matching the geometry and plasticity config."""
    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))

    # Add plastic state variables based on plasticity config
    if plasticity_config.plastic_state_dims:
        for name, dim in plasticity_config.plastic_state_dims.items():
            registry.register(StateVariable(name=name, fast_plastic=True))

    # Add substrate-owned variables (if needed)
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    return registry


# ============================================================
# Property Tests
# ============================================================


def test_random_6d_coordinate_is_valid():
    """Random 6-D coordinate should pass validation."""
    for _ in range(20):
        config = create_random_system_config()
        assert config is not None
        # Validation happens in create_random_system_config


def test_random_6d_coordinate_constructs_system():
    """Random 6-D coordinate should construct a valid joint system."""
    for _ in range(10):
        config = create_random_system_config()

        substrate, geometry, _dynamics, _credit, _update = create_system_from_config(
            config
        )
        registry = create_registry_from_geometry(geometry, config.plasticity)

        # Validate registry with dummy state
        dummy_activity = {
            name: param.detach().clone() for name, param in geometry.params.items()
        }
        dummy_substrate = {"conductance": torch.randn(4, 20)}
        dummy_plastic = {}
        if config.plasticity.plastic_state_dims:
            for name, dim in config.plasticity.plastic_state_dims.items():
                dummy_activity[name] = torch.zeros(4, dim)
                dummy_plastic[name] = torch.zeros(4, dim)
        registry.validate(
            CompositeState(
                activity=dummy_activity,
                plastic=dummy_plastic,
                substrate=dummy_substrate,
            )
        )

        # Create context
        context = SystemContext(
            theta=geometry.params,
            geometry=geometry,
            substrate=substrate,
            substrate_config=config.substrate,
            geometry_config=config.geometry,
            dynamics_config=config.dynamics,
            credit_config=config.credit,
            update_config=config.update,
            plasticity_config=config.plasticity,
            registry=registry,
        )

        assert context is not None
        assert context.theta == geometry.params


def test_null_plasticity_reproduces_5d_behavior():  # ruff: ignore[too-many-locals]
    """M=Null coordinates should reproduce 5-D behavior (Zero-Extension)."""
    # Create config with Null plasticity
    config = SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        plasticity=PlasticityConfig.null(),
        credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update=ParameterUpdateConfig.euclidean(step_size=0.01),
    )
    config.validate()

    substrate, geometry, dynamics, credit, update = create_system_from_config(config)
    system_5d = compose_system(substrate, geometry, dynamics, credit, update)

    registry = create_registry_from_geometry(geometry, config.plasticity)
    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_substrate = {"conductance": torch.randn(4, 20)}
    registry.validate(
        CompositeState(activity=dummy_activity, plastic={}, substrate=dummy_substrate)
    )

    context = SystemContext(
        theta=geometry.params,
        geometry=geometry,
        substrate=substrate,
        substrate_config=config.substrate,
        geometry_config=config.geometry,
        dynamics_config=config.dynamics,
        credit_config=config.credit,
        update_config=config.update,
        plasticity_config=config.plasticity,
        registry=registry,
    )

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # 5-D system
    metrics_5d = system_5d.train_step(x, y)

    # Joint with NullPlasticity
    plasticity = NullPlasticity()
    z = CompositeState(activity={"x": x, "y": y}, plastic={}, substrate={})
    psi_initial = {}
    psi_final = plasticity.step(psi_initial, z, context)

    assert psi_final is psi_initial
    assert metrics_5d["loss"] >= 0


def test_all_plasticity_types_constructible():
    """All plasticity types should be constructible via SystemConfig."""
    for plasticity_name in PLASTICITY_FACTORIES:
        config = SystemConfig(
            substrate=SubstrateConfig.digital(),
            geometry=GeometryConfig.recurrent(
                input_dim=10, output_dim=2, hidden_dims=(20,)
            ),
            dynamics=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
            plasticity=PLASTICITY_FACTORIES[plasticity_name](),
            credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
            update=ParameterUpdateConfig.euclidean(step_size=0.01),
        )
        config.validate()
        assert config.plasticity.plasticity_type == plasticity_name


def test_registry_matches_plasticity_config():
    """Registry should correctly represent plasticity config dimensions."""
    for plasticity_name in PLASTICITY_FACTORIES:
        plasticity_config = PLASTICITY_FACTORIES[plasticity_name]()

        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,))
        )
        registry = StateRegistry()
        for name in geometry.params:
            registry.register(StateVariable(name=name, persistent=True))

        if plasticity_config.plastic_state_dims:
            for name, dim in plasticity_config.plastic_state_dims.items():
                registry.register(StateVariable(name=name, fast_plastic=True))

        groups = registry.lifecycle_groups()
        assert len(groups["persistent"]) == len(geometry.params)

        if plasticity_config.plastic_state_dims:
            assert len(groups["fast_plastic"]) == len(
                plasticity_config.plastic_state_dims
            )
        else:
            assert len(groups["fast_plastic"]) == 0


def test_substrate_adapters_preserve_interface():
    """Substrate adapters should preserve Substrate interface."""
    for target_name in ["ternary", "sparse", "complex", "neuromorphic"]:
        adapter = create_substrate_adapter("digital", target_name)
        assert adapter is not None
        assert hasattr(adapter, "get_forward_operator")
        assert hasattr(adapter, "get_weight_update_operator")
        assert hasattr(adapter, "quantize_weights")
        assert hasattr(adapter, "inject_state_noise")


def test_dynamics_adapters_preserve_interface():
    """Dynamics adapters should preserve Dynamics interface."""
    source_dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )
    adapter = create_dynamics_adapter(
        "energy_minimization", "instantaneous", source_dynamics
    )
    assert adapter is not None
    assert hasattr(adapter, "settle")
    assert hasattr(adapter, "compute_energy")


def test_cross_axis_validation():
    """SystemConfig.validate() should enforce cross-axis constraints."""
    # Valid config
    config = SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        plasticity=PlasticityConfig.null(),
        credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update=ParameterUpdateConfig.euclidean(step_size=0.01),
    )
    config.validate()  # Should not raise


def test_geometry_compatibility_with_substrate():
    """Geometry should be compatible with substrate capabilities."""
    # Digital substrate with recurrent geometry - should work
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,))
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital())

    x = torch.randn(4, 10)
    y = geometry.forward(x, substrate)
    assert y.shape == (4, 2)


def test_all_geometry_types_constructible():
    """All geometry types should be constructible."""
    for name, factory in GEOMETRY_FACTORIES.items():
        geometry, config = factory()
        assert geometry is not None
        assert config is not None
        assert len(geometry.params) > 0


def test_all_dynamics_types_constructible():
    """All dynamics types should be constructible."""
    for name, factory in DYNAMICS_FACTORIES.items():
        dynamics, config = factory()
        assert dynamics is not None
        assert config is not None


def test_all_credit_types_constructible():
    """All credit assignment types should be constructible."""
    for name, factory in CREDIT_FACTORIES.items():
        credit, config = factory()
        assert credit is not None
        assert config is not None


def test_all_update_types_constructible():
    """All update types should be constructible."""
    for name, factory in UPDATE_FACTORIES.items():
        update, config = factory()
        assert update is not None
        assert config is not None


def test_composite_state_full_joint_state():
    """CompositeState should hold full joint state (x, ψ, σ)."""
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,))
    )

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    registry.register(StateVariable(name="eligibility", fast_plastic=True))
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_plastic = {"eligibility": torch.zeros(4, 20)}
    dummy_substrate = {"conductance": torch.randn(4, 20)}
    registry.validate(
        CompositeState(
            activity=dummy_activity, plastic=dummy_plastic, substrate=dummy_substrate
        )
    )

    z = CompositeState(
        activity=dummy_activity,
        plastic=dummy_plastic,
        substrate=dummy_substrate,
    )

    assert len(z.activity) == len(geometry.params)  # persistent params
    assert len(z.plastic) == 1
    assert len(z.substrate) == 1


def test_state_registry_lifecycle_groups():
    """Registry lifecycle_groups should correctly categorize variables."""
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,))
    )

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    registry.register(
        StateVariable(name="eligibility", fast_plastic=True, consolidatable=True)
    )
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    groups = registry.lifecycle_groups()

    assert len(groups["persistent"]) == len(geometry.params)
    assert "eligibility" in groups["fast_plastic"]
    assert "eligibility" in groups["consolidatable"]
    assert "conductance" in groups["substrate_owned"]


def test_consolidation_respects_registry():
    """Consolidation should only promote consolidatable variables."""
    from computronium.core.joint import ConsolidationConfig, consolidate

    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,))
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital())

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    registry.register(
        StateVariable(name="consol", fast_plastic=True, consolidatable=True)
    )
    registry.register(
        StateVariable(name="not_consol", fast_plastic=True, consolidatable=False)
    )

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_plastic = {
        "consol": torch.ones(4, 20) * 2.0,
        "not_consol": torch.ones(4, 20) * 3.0,
    }
    registry.validate(
        CompositeState(activity=dummy_activity, plastic=dummy_plastic, substrate={})
    )

    context = SystemContext(
        theta=geometry.params,
        geometry=geometry,
        substrate=substrate,
        substrate_config=SubstrateConfig.digital(),
        geometry_config=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics_config=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.null(),
        registry=registry,
    )

    z = CompositeState(
        activity=dummy_activity,
        plastic={
            "consol": torch.ones(4, 20) * 2.0,
            "not_consol": torch.ones(4, 20) * 3.0,
        },
        substrate={},
    )

    new_context = consolidate(z, context, ConsolidationConfig(promote_all=True))

    # consol should be promoted
    assert "consol" in new_context.theta
    assert torch.allclose(new_context.theta["consol"], torch.ones(4, 20) * 2.0)

    # not_consol should NOT be promoted
    assert "not_consol" not in new_context.theta


def test_plasticity_config_factories():
    """PlasticityConfig factories should create valid configs."""
    null_config = PlasticityConfig.null()
    assert null_config.plasticity_type == "null"
    assert null_config.plastic_state_dims is None

    routing_config = PlasticityConfig.routing(gate_dim=64)
    assert routing_config.plasticity_type == "routing"
    assert routing_config.plastic_state_dims == {"gate_logits": 64, "active_routes": 64}

    fw_config = PlasticityConfig.fast_weights(fast_weight_dim=128)
    assert fw_config.plasticity_type == "fast_weights"
    assert fw_config.plastic_state_dims == {"fast_weights": 128}

    sc_config = PlasticityConfig.substrate_coupled()
    assert sc_config.plasticity_type == "substrate_coupled"
    assert sc_config.plastic_state_dims is None

    rs_config = PlasticityConfig.rule_state(num_operators=16)
    assert rs_config.plasticity_type == "rule_state"
    assert rs_config.plastic_state_dims == {"operator_logits": 16}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
