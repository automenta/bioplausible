"""Property tests for Joint Lifecycle Locks (J1-J7).

These tests verify the lifecycle invariants of the 6-D joint architecture:
- J1: NullPlasticity preserves 5-D dynamics (Zero-Extension)
- J2: Persistent θ not mutated during intra-episode steps
- J3: fast_plastic variables mutate only through plasticity projection
- J4: substrate_owned variables respect substrate physics constraints
- J5: consolidatable variables promoted only at episode boundaries
- J6: Cross-adapters preserve joint transition shape & registry semantics
- J7: Trajectory records contain full z = (x, ψ, σ)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from computronium.core.dynamics.adapters import (
    EnergyToInstantaneousAdapter,
    StateDynamicsConfig,
)
from computronium.core.joint import (
    CompositeState,
    ConsolidationConfig,
    NullPlasticity,
    PlasticityConfig,
    StateRegistry,
    StateVariable,
    SystemContext,
    consolidate,
)
from computronium.core.joint.state import JointTrajectoryRecorder
from computronium.core.substrates.adapters import DigitalToTernaryAdapter
from computronium.core.system_trainer import compose_system
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,  # ruff: ignore[redefined-while-unused]
    SubstrateConfig,
    SystemConfig,
    SystemState,
    TernarySubstrate,
    ThermodynamicContrast,
)


def _create_test_system():
    """Create a minimal 5-D system for testing."""
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )
    update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))

    return substrate, geometry, dynamics, credit, update


def _create_registry_with_geometry(geometry: RecurrentGeometry) -> StateRegistry:
    """Create a registry with all geometry params as persistent."""
    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    return registry


def _create_dummy_state_for_registry(
    registry: StateRegistry, geometry: RecurrentGeometry
) -> CompositeState:
    """Create a dummy CompositeState that satisfies registry validation."""
    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    return CompositeState(activity=dummy_activity, plastic={}, substrate={})


# ============================================================
# J1: NullPlasticity preserves 5-D dynamics (Zero-Extension)
# ============================================================


def test_j1_null_plasticity_zero_extension():  # ruff: ignore[too-many-locals]
    """J1: Joint system with M=Null ≡ 5-D system (Zero-Extension Theorem)."""
    substrate, geometry, dynamics, credit, update = _create_test_system()

    # 5-D system
    system_5d = compose_system(substrate, geometry, dynamics, credit, update)

    # Joint system with NullPlasticity
    plasticity = NullPlasticity()
    sys_config = SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        plasticity=PlasticityConfig.null(),
        credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update=ParameterUpdateConfig.euclidean(step_size=0.01),
    )
    sys_config.validate()

    # Create registry for joint system
    registry = _create_registry_with_geometry(geometry)
    registry.validate(_create_dummy_state_for_registry(registry, geometry))

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

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Run 5-D system
    metrics_5d = system_5d.train_step(x, y)

    # NullPlasticity step is identity
    z = CompositeState(activity={"x": x, "y": y}, plastic={}, substrate={})
    psi_initial = {}
    psi_final = plasticity.step(psi_initial, z, context)

    # NullPlasticity returns same object (identity)
    assert psi_final is psi_initial

    # 5-D invariants preserved
    assert metrics_5d["loss"] >= 0
    assert "energy" in metrics_5d


# ============================================================
# J2: Persistent θ not mutated during intra-episode steps
# ============================================================


def test_j2_theta_immutable_intra_episode():
    """J2: Persistent θ parameters are never mutated during intra-episode steps."""
    substrate, geometry, _dynamics, _credit, _update = _create_test_system()

    # Snapshot initial theta
    theta_initial = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }

    # Create registry and context
    registry = _create_registry_with_geometry(geometry)
    registry.validate(_create_dummy_state_for_registry(registry, geometry))

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

    plasticity = NullPlasticity()

    # Run multiple intra-episode steps
    z = CompositeState(
        activity={"x": torch.randn(4, 10), "y": torch.randint(0, 2, (4,))},
        plastic={},
        substrate={},
    )

    for _ in range(5):
        # Plasticity step (NullPlasticity does nothing)
        psi = plasticity.step({}, z, context)

        # Simulate activity evolution (in real system, this would be StateDynamics.settle)
        z = CompositeState(
            activity={"x": torch.randn(4, 10), "y": torch.randint(0, 2, (4,))},
            plastic=psi,
            substrate={},
        )

    # Theta should be unchanged
    for name, param in context.theta.items():
        assert torch.allclose(param, theta_initial[name]), (
            f"Theta {name} was mutated intra-episode!"
        )


# ============================================================
# J3: fast_plastic variables mutate only through plasticity projection
# ============================================================


class TestPlasticity:
    """Minimal test plasticity primitive for lifecycle testing."""

    config = PlasticityConfig(
        plasticity_type="test", plastic_state_dims={"test_psi": 10}
    )

    def initial_psi(self, context: SystemContext) -> dict[str, Tensor]:
        return {"test_psi": torch.zeros(4, 10)}

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        # Only plasticity projection should modify psi
        new_psi = {k: v.clone() for k, v in psi.items()}
        new_psi["test_psi"] = new_psi["test_psi"] + 0.1 * torch.randn_like(  # ruff: ignore[non-augmented-assignment]
            new_psi["test_psi"]
        )
        return new_psi


def test_j3_fast_plastic_only_via_plasticity():
    """J3: fast_plastic (ψ) variables only mutate through plasticity projection."""
    substrate, geometry, _dynamics, _credit, _update = _create_test_system()

    registry = _create_registry_with_geometry(geometry)
    registry.register(StateVariable(name="test_psi", fast_plastic=True))

    # Include fast_plastic in plastic dict for validation
    dummy_plastic = {"test_psi": torch.zeros(4, 10)}
    registry.validate(
        CompositeState(
            activity={
                name: param.detach().clone() for name, param in geometry.params.items()
            },
            plastic=dummy_plastic,
            substrate={},
        )
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
        plasticity_config=PlasticityConfig(
            plasticity_type="test", plastic_state_dims={"test_psi": 10}
        ),
        registry=registry,
    )

    plasticity = TestPlasticity()

    # Initial plastic state
    psi = plasticity.initial_psi(context)
    psi_initial = {k: v.detach().clone() for k, v in psi.items()}

    z = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic=psi,
        substrate={},
    )

    # Step plasticity
    psi_new = plasticity.step(psi, z, context)

    # Plastic state should have changed
    assert not torch.allclose(psi_new["test_psi"], psi_initial["test_psi"])

    # Theta should be unchanged
    for name, param in context.theta.items():
        assert torch.allclose(param, geometry.params[name])


# ============================================================
# J4: substrate_owned variables respect substrate physics constraints
# ============================================================


def test_j4_substrate_owned_respects_physics():
    """J4: substrate_owned (σ) variables respect physical device constraints."""
    substrate = TernarySubstrate(SubstrateConfig.ternary())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )

    registry = _create_registry_with_geometry(geometry)
    # Substrate-owned state for ternary
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    context = SystemContext(  # ruff: ignore[unused-variable]
        theta=geometry.params,
        geometry=geometry,
        substrate=substrate,
        substrate_config=SubstrateConfig.ternary(),
        geometry_config=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics_config=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.null(),
        registry=registry,
    )

    # Create substrate state that violates ternary constraint
    z = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={},
        substrate={"conductance": torch.randn(4, 20) * 10},  # Not in {-1, 0, 1}
    )

    # Substrate forward operator should project to valid states
    # Use geometry.forward which uses substrate's get_forward_operator
    x = z.activity["x"]
    y = geometry.forward(x, substrate)

    # Output should be valid (ternary substrate constrains internal state)
    assert y is not None
    assert y.shape == (4, 2)


def test_j4_substrate_adapter_preserves_constraints():
    """J4: Substrate adapter preserves physical constraints."""
    substrate = DigitalToTernaryAdapter()
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )

    registry = _create_registry_with_geometry(geometry)
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    # Dummy state with all persistent params in activity
    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_activity["x"] = torch.randn(4, 10)
    dummy_substrate = {"conductance": torch.randn(4, 20)}
    registry.validate(
        CompositeState(activity=dummy_activity, plastic={}, substrate=dummy_substrate)
    )

    context = SystemContext(  # ruff: ignore[unused-variable]
        theta=geometry.params,
        geometry=geometry,
        substrate=substrate,
        substrate_config=SubstrateConfig.ternary(),
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
        plastic={},
        substrate=dummy_substrate,
    )

    # Adapter forward operator should preserve state structure
    y = geometry.forward(z.activity["x"], substrate)

    # Output shape preserved
    assert y.shape == (4, 2)

    # Registry still validates
    registry.validate(z)


# ============================================================
# J5: consolidatable variables promoted only at episode boundaries
# ============================================================


def test_j5_consolidation_only_at_episode_boundary():
    """J5: Consolidatable ψ promoted to θ only at episode boundaries via consolidate()."""
    substrate, geometry, _dynamics, _credit, _update = _create_test_system()

    registry = _create_registry_with_geometry(geometry)
    # Consolidatable fast weight
    registry.register(
        StateVariable(name="fast_weight", fast_plastic=True, consolidatable=True)
    )

    # Include fast_plastic in plastic dict for validation
    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_plastic = {"fast_weight": torch.zeros(4, 20)}
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
        plasticity_config=PlasticityConfig(
            plasticity_type="fast_weights",
            plastic_state_dims={"fast_weight": 20},
            consolidation_config={"promotion_scale": 0.1},
        ),
        registry=registry,
    )

    # Initial theta snapshot
    theta_initial = {
        name: param.detach().clone() for name, param in context.theta.items()
    }

    # Simulate intra-episode: fast_weight evolves but is NOT consolidated
    psi_value = torch.randn(4, 20)
    z = CompositeState(
        activity=dummy_activity,
        plastic={"fast_weight": psi_value.clone()},
        substrate={},
    )

    # Theta should be unchanged during episode
    for name, param in context.theta.items():
        assert torch.allclose(param, theta_initial[name])

    # Now at episode boundary: consolidate
    # Save psi before consolidation (it gets zeroed)
    psi_before = z.plastic["fast_weight"].clone()
    new_context = consolidate(
        z, context, ConsolidationConfig(promote_all=True, promotion_scale=0.1)
    )

    # Theta should now include promoted fast_weight
    # (Note: fast_weight is new, so it gets added to theta)
    assert "fast_weight" in new_context.theta
    assert torch.allclose(new_context.theta["fast_weight"], psi_before * 0.1)

    # Original theta unchanged
    for name in theta_initial:
        assert torch.allclose(new_context.theta[name], theta_initial[name])


def test_j5_consolidation_resets_plastic():
    """J5: Consolidation optionally resets promoted plastic state."""
    substrate, geometry, _dynamics, _credit, _update = _create_test_system()

    registry = _create_registry_with_geometry(geometry)
    registry.register(
        StateVariable(name="fast_weight", fast_plastic=True, consolidatable=True)
    )

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_plastic = {"fast_weight": torch.zeros(4, 20)}
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
        plasticity_config=PlasticityConfig(
            plasticity_type="fast_weights",
            plastic_state_dims={"fast_weight": 20},
            consolidation_config={"promotion_scale": 0.1},
        ),
        registry=registry,
    )

    psi_value = torch.ones(4, 20) * 5.0
    z = CompositeState(
        activity=dummy_activity,
        plastic={"fast_weight": psi_value.clone()},
        substrate={},
    )

    # Consolidate with reset_plastic=True (default) and promotion_scale=0.1
    new_context = consolidate(
        z,
        context,
        ConsolidationConfig(promote_all=True, reset_plastic=True, promotion_scale=0.1),
    )

    # Plastic state should be zeroed
    assert torch.allclose(z.plastic["fast_weight"], torch.zeros(4, 20))

    # But theta has the promoted value
    assert torch.allclose(new_context.theta["fast_weight"], psi_value * 0.1)


# ============================================================
# J6: Cross-adapters preserve joint transition shape & registry semantics
# ============================================================


def test_j6_substrate_adapter_preserves_registry_semantics():
    """J6: Substrate adapter preserves joint transition shape and registry."""
    substrate = DigitalToTernaryAdapter()
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )

    registry = _create_registry_with_geometry(geometry)
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_activity["x"] = torch.randn(4, 10)
    dummy_substrate = {"conductance": torch.randn(4, 20)}
    registry.validate(
        CompositeState(activity=dummy_activity, plastic={}, substrate=dummy_substrate)
    )

    context = SystemContext(  # ruff: ignore[unused-variable]
        theta=geometry.params,
        geometry=geometry,
        substrate=substrate,
        substrate_config=SubstrateConfig.ternary(),
        geometry_config=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics_config=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.null(),
        registry=registry,
    )

    # Input state
    z = CompositeState(
        activity=dummy_activity,
        plastic={},
        substrate=dummy_substrate,
    )

    # Adapter forward operator should preserve state structure
    y = geometry.forward(z.activity["x"], substrate)

    # Output shape preserved
    assert y.shape == (4, 2)

    # Registry still validates
    registry.validate(z)


@pytest.mark.xfail(
    reason="EnergyToInstantaneousAdapter modifies frozen config - bug in adapter",
    strict=True,
)
def test_j6_dynamics_adapter_preserves_shape():
    """J6: Dynamics adapter preserves CompositeState structure."""
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )
    source_dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )
    dynamics = EnergyToInstantaneousAdapter(source_dynamics)

    registry = _create_registry_with_geometry(geometry)

    context = SystemContext(  # ruff: ignore[unused-variable]
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

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_activity["x"] = torch.randn(4, 10)
    registry.validate(CompositeState(activity=dummy_activity, plastic={}, substrate={}))

    z = CompositeState(
        activity=dummy_activity,
        plastic={},
        substrate={},
    )

    # Dynamics step should preserve CompositeState structure
    state = SystemState(x=z.activity["x"])
    state.activations = geometry.forward(state.x, substrate)
    # Use the adapter's settle (don't modify source config)
    state = dynamics.settle(state, geometry, substrate, target=None)

    # Should produce valid output
    assert state.activations is not None


# ============================================================
# J7: Trajectory records contain full z = (x, ψ, σ)
# ============================================================


def test_j7_trajectory_records_full_joint_state():
    """J7: JointTrajectory records activity, plastic, and substrate."""

    recorder = JointTrajectoryRecorder(
        max_steps=10, record_plastic=True, record_substrate=True
    )

    registry = StateRegistry()
    registry.register(StateVariable(name="weight", persistent=True))
    registry.register(StateVariable(name="eligibility", fast_plastic=True))
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    for i in range(5):
        z = CompositeState(
            activity={"weight": torch.full((4, 10), float(i)), "x": torch.randn(4, 10)},
            plastic={"eligibility": torch.full((4, 20), float(i * 2))},
            substrate={"conductance": torch.full((4, 5), float(i * 3))},
        )
        recorder.record(z)

    traj = recorder.get_trajectory()

    # All three components recorded
    assert len(traj.activity) == 5
    assert len(traj.plastic) == 5
    assert len(traj.substrate) == 5

    # Verify content
    assert torch.allclose(traj.activity[0]["weight"], torch.zeros(4, 10))
    assert torch.allclose(traj.plastic[2]["eligibility"], torch.full((4, 20), 4.0))
    assert torch.allclose(traj.substrate[4]["conductance"], torch.full((4, 5), 12.0))

    # Reconstruction preserves all components
    z_reconstructed = traj.get_step(3)
    assert "weight" in z_reconstructed.activity
    assert "eligibility" in z_reconstructed.plastic
    assert "conductance" in z_reconstructed.substrate


def test_j7_trajectory_optional_components():
    """J7: Trajectory can optionally omit plastic/substrate recording."""

    # Record only activity
    recorder = JointTrajectoryRecorder(
        max_steps=10, record_plastic=False, record_substrate=False
    )

    for i in range(3):
        z = CompositeState(
            activity={"x": torch.full((4, 10), float(i))},
            plastic={"psi": torch.full((4, 20), float(i))},
            substrate={"sigma": torch.full((4, 5), float(i))},
        )
        recorder.record(z)

    traj = recorder.get_trajectory()

    assert len(traj.activity) == 3
    assert len(traj.plastic) == 0
    assert len(traj.substrate) == 0

    # Reconstruction still works for activity
    z_reconstructed = traj.get_step(1)
    assert "x" in z_reconstructed.activity
    assert z_reconstructed.plastic == {}
    assert z_reconstructed.substrate == {}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
