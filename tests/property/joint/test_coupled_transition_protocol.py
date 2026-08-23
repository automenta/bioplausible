"""Property tests for CoupledTransition protocol and LegacyDynamicsAsCoupledTransition."""

from __future__ import annotations

import torch

from bioplausible.core.joint import (
    CompositeState,
    CoupledTransition,
    LegacyDynamicsAsCoupledTransition,
    NullPlasticity,
    PlasticityConfig,
    StateRegistry,
    StateVariable,
    SystemContext,
)
from bioplausible.core.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemConfig,
    ThermodynamicContrast,
)
from bioplausible.core.system_trainer import compose_system


def _create_system_context() -> SystemContext:
    """Create a minimal SystemContext for testing."""
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )
    dynamics_config = StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5)
    credit_config = CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    update_config = ParameterUpdateConfig.euclidean(step_size=0.01)
    plasticity_config = PlasticityConfig.null()

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))

    return SystemContext(
        theta=geometry.params,
        geometry=geometry,
        substrate=substrate,
        substrate_config=SubstrateConfig.digital(),
        geometry_config=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics_config=dynamics_config,
        credit_config=credit_config,
        update_config=update_config,
        plasticity_config=plasticity_config,
        registry=registry,
    )


def test_coupled_transition_protocol_exists():
    """Verify CoupledTransition protocol is defined and checkable."""
    # Should be a runtime-checkable protocol
    assert hasattr(CoupledTransition, "__instancecheck__")

    # NullPlasticity should be usable in a CoupledTransition implementation
    class TestTransition:
        def step(self, z: CompositeState, context: SystemContext) -> CompositeState:
            return z

    assert isinstance(TestTransition(), CoupledTransition)


def test_null_plasticity_as_transition_component():
    """NullPlasticity can be used as plasticity component."""
    plasticity = NullPlasticity()
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert plasticity.config.plasticity_type == "null"


def test_legacy_dynamics_as_coupled_transition():
    """LegacyDynamicsAsCoupledTransition wraps 5-D system as joint transition."""
    # Create 5-D system
    system_5d = compose_system(
        DigitalSubstrate(SubstrateConfig.digital()),
        RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
            hidden_dim=20,
        ),
        EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5)
        ),
        ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)),
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
    )

    # Wrap as joint transition
    wrapper = LegacyDynamicsAsCoupledTransition(system_5d)
    assert isinstance(wrapper, CoupledTransition)

    # Test step with inference (no target)
    context = _create_system_context()
    z = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={},
        substrate={},
    )

    z_next = wrapper.step(z, context)
    assert "x" in z_next.activity
    assert "output" in z_next.activity


def test_system_config_6d_construction():
    """Test SystemConfig can be constructed with 6 axes."""
    config = SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics=StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5),
        plasticity=PlasticityConfig.null(),
        credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update=ParameterUpdateConfig.euclidean(step_size=0.01),
    )

    # Validate should pass
    config.validate()

    # Check plasticity axis is present
    assert config.plasticity is not None
    assert config.plasticity.plasticity_type == "null"


def test_system_config_5d_backward_compat():
    """Test SystemConfig can be constructed with 5 axes (plasticity defaults to Null)."""
    config = SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics=StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5),
        credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update=ParameterUpdateConfig.euclidean(step_size=0.01),
    )

    # Validate should pass
    config.validate()

    # Plasticity should default to Null
    assert config.plasticity is not None
    assert config.plasticity.plasticity_type == "null"


def test_plasticity_config_factories():
    """Test PlasticityConfig factory methods."""
    null_config = PlasticityConfig.null()
    assert null_config.plasticity_type == "null"

    routing_config = PlasticityConfig.routing(gate_dim=64)
    assert routing_config.plasticity_type == "routing"
    assert routing_config.plastic_state_dims["gate_logits"] == 64

    fw_config = PlasticityConfig.fast_weights(fast_weight_dim=512)
    assert fw_config.plasticity_type == "fast_weights"
    assert fw_config.plastic_state_dims["fast_weights"] == 512

    sc_config = PlasticityConfig.substrate_coupled()
    assert sc_config.plasticity_type == "substrate_coupled"

    rs_config = PlasticityConfig.rule_state(num_operators=8)
    assert rs_config.plasticity_type == "rule_state"
    assert rs_config.plastic_state_dims["operator_logits"] == 8
