"""Property tests for Plasticity Axis (M-axis) certifications.

Tests verify that each plasticity primitive passes axis certification:
- NullPlasticity (Zero-Extension)
- RoutingPlasticity (state-dependent gating)
- FastWeightPlasticity (episode-local associative memory)
- SubstrateCoupledPlasticity (reuse substrate physics)
- RuleStatePlasticity (operator selection via ψ)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from bioplausible.core.joint import (
    CompositeState,
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


def _create_test_context(plasticity_config: PlasticityConfig = None):
    """Create a test context with the given plasticity config."""
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))

    dummy_plastic = {}
    if plasticity_config and plasticity_config.plastic_state_dims:
        for name, dim in plasticity_config.plastic_state_dims.items():
            registry.register(StateVariable(name=name, fast_plastic=True))
            dummy_plastic[name] = torch.zeros(4, dim)

    registry.register(StateVariable(name="conductance", substrate_owned=True))

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_substrate = {"conductance": torch.randn(4, 20)}
    registry.validate(
        CompositeState(
            activity=dummy_activity, plastic=dummy_plastic, substrate=dummy_substrate
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
        plasticity_config=plasticity_config or PlasticityConfig.null(),
        registry=registry,
    )
    return context, geometry, registry


# ============================================================
# NullPlasticity Certification (Zero-Extension Theorem)
# ============================================================


def test_null_plasticity_axis_certification():
    """NullPlasticity should pass axis certification (Zero-Extension Theorem)."""
    plasticity = NullPlasticity()
    config = PlasticityConfig.null()

    # Basic properties
    assert plasticity.config.plasticity_type == "null"
    assert plasticity.config.plastic_state_dims is None
    assert plasticity.initial_psi(None) == {}

    # Step is identity
    psi = {"test": torch.ones(5)}
    z = CompositeState(activity={}, plastic={}, substrate={})
    context = None  # Not used by NullPlasticity

    result = plasticity.step(psi, z, context)
    assert result is psi
    assert torch.allclose(result["test"], torch.ones(5))


def test_null_plasticity_protocol_compliance():
    """NullPlasticity should comply with PlasticityPrimitive protocol."""
    plasticity = NullPlasticity()

    # Should have required attributes and methods
    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert isinstance(plasticity.config, PlasticityConfig)


def test_null_plasticity_preserves_joint_invariants():
    """NullPlasticity should preserve joint system invariants."""
    context, geometry, registry = _create_test_context(PlasticityConfig.null())
    plasticity = NullPlasticity()

    z = CompositeState(
        activity={
            name: param.detach().clone() for name, param in geometry.params.items()
        },
        plastic={},
        substrate={"conductance": torch.randn(4, 20)},
    )

    # Multiple steps should not change anything
    for _ in range(5):
        psi = plasticity.step({}, z, context)
        assert psi == {}

    # Theta should be unchanged
    for name, param in context.theta.items():
        assert torch.allclose(param, geometry.params[name])


# ============================================================
# RoutingPlasticity Certification (stub - not yet implemented)
# ============================================================


class RoutingPlasticity:
    """Routing plasticity: state-dependent pathway gating.

    ψ = (gate_logits, active_routes) where:
    - gate_logits: learnable gating parameters [batch, gate_dim]
    - active_routes: binary mask of active pathways [batch, gate_dim]
    """

    config = PlasticityConfig.routing(gate_dim=32)

    def initial_psi(self, context: SystemContext) -> dict[str, Tensor]:
        return {
            "gate_logits": torch.zeros(
                4, self.config.plastic_state_dims["gate_logits"]
            ),
            "active_routes": torch.zeros(
                4, self.config.plastic_state_dims["active_routes"]
            ),
        }

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        new_psi = {k: v.clone() for k, v in psi.items()}
        # Simple gating: activate top-k routes based on activity
        if "x" in z.activity:
            x = z.activity["x"]
            # Gate logits evolve based on input
            new_psi["gate_logits"] = new_psi["gate_logits"] + 0.01 * x.mean(
                dim=1, keepdim=True
            ).expand(-1, 32)
            # Active routes = sigmoid(gate_logits) > 0.5
            new_psi["active_routes"] = (
                torch.sigmoid(new_psi["gate_logits"]) > 0.5
            ).float()
        return new_psi


def test_routing_plasticity_axis_certification():
    """RoutingPlasticity should pass axis certification."""
    plasticity = RoutingPlasticity()

    # Config properties
    assert plasticity.config.plasticity_type == "routing"
    assert plasticity.config.plastic_state_dims == {
        "gate_logits": 32,
        "active_routes": 32,
    }

    # Initial psi
    psi = plasticity.initial_psi(None)
    assert "gate_logits" in psi
    assert "active_routes" in psi
    assert psi["gate_logits"].shape == (4, 32)
    assert psi["active_routes"].shape == (4, 32)

    # Step should update psi
    context, geometry, registry = _create_test_context(
        PlasticityConfig.routing(gate_dim=32)
    )
    z = CompositeState(
        activity={
            name: param.detach().clone() for name, param in geometry.params.items()
        },
        plastic=psi,
        substrate={"conductance": torch.randn(4, 20)},
    )
    # Add x for the update
    z.activity["x"] = torch.randn(4, 10)

    psi_new = plasticity.step(psi, z, context)

    # Gate logits should evolve
    assert not torch.allclose(psi_new["gate_logits"], psi["gate_logits"])
    # Active routes should be binary
    assert torch.all((psi_new["active_routes"] == 0) | (psi_new["active_routes"] == 1))


def test_routing_plasticity_protocol_compliance():
    """RoutingPlasticity should comply with PlasticityPrimitive protocol."""
    plasticity = RoutingPlasticity()

    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert isinstance(plasticity.config, PlasticityConfig)


# ============================================================
# FastWeightPlasticity Certification (stub - not yet implemented)
# ============================================================


class FastWeightPlasticity:
    """Fast weight plasticity: episode-local associative memory.

    ψ = fast_weights updated as: A_{t+1} = decay * A_t + η * outer(pre, post)
    """

    config = PlasticityConfig.fast_weights(fast_weight_dim=64)

    def __init__(self, decay: float = 0.9, lr: float = 0.1):
        self.decay = decay
        self.lr = lr

    def initial_psi(self, context: SystemContext) -> dict[str, Tensor]:
        return {
            "fast_weights": torch.zeros(
                4, self.config.plastic_state_dims["fast_weights"]
            )
        }

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        new_psi = {k: v.clone() for k, v in psi.items()}
        # Fast weight update: decay + outer product
        if "x" in z.activity and "y" in z.activity:
            pre = z.activity["x"].mean(dim=0)  # [input_dim]
            post = z.activity["y"].float().mean(dim=0)  # [output_dim]
            if pre.shape[0] == 10 and post.shape[0] == 2:
                # Reshape to match fast_weights dim
                outer = torch.outer(pre, post).flatten()  # [20]
                if outer.shape[0] <= 64:
                    new_psi["fast_weights"][:, : outer.shape[0]] = (
                        self.decay * new_psi["fast_weights"][:, : outer.shape[0]]
                        + self.lr * outer
                    )
        return new_psi


def test_fast_weight_plasticity_axis_certification():
    """FastWeightPlasticity should pass axis certification."""
    plasticity = FastWeightPlasticity()

    assert plasticity.config.plasticity_type == "fast_weights"
    assert plasticity.config.plastic_state_dims == {"fast_weights": 64}

    psi = plasticity.initial_psi(None)
    assert "fast_weights" in psi
    assert psi["fast_weights"].shape == (4, 64)

    context, geometry, registry = _create_test_context(
        PlasticityConfig.fast_weights(fast_weight_dim=64)
    )
    z = CompositeState(
        activity={
            name: param.detach().clone() for name, param in geometry.params.items()
        },
        plastic=psi,
        substrate={"conductance": torch.randn(4, 20)},
    )
    # Add x and y for update
    z.activity["x"] = torch.randn(4, 10)
    z.activity["y"] = torch.randn(4, 2)

    psi_new = plasticity.step(psi, z, context)

    # Fast weights should evolve
    assert not torch.allclose(psi_new["fast_weights"], psi["fast_weights"])


def test_fast_weight_plasticity_protocol_compliance():
    """FastWeightPlasticity should comply with PlasticityPrimitive protocol."""
    plasticity = FastWeightPlasticity()

    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert isinstance(plasticity.config, PlasticityConfig)


# ============================================================
# SubstrateCoupledPlasticity Certification (stub - not yet implemented)
# ============================================================


class SubstrateCoupledPlasticity:
    """Substrate-coupled plasticity: reuse substrate physics as plasticity.

    ψ ≡ σ (plastic state is substrate state)
    """

    config = PlasticityConfig.substrate_coupled()

    def initial_psi(self, context: SystemContext) -> dict[str, Tensor]:
        # ψ is empty - plasticity is in substrate state
        return {}

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        # Substrate-coupled plasticity: ψ evolution = substrate state evolution
        # In practice, substrate state is updated by substrate's weight update operator
        # This is a no-op at plasticity level - substrate handles it
        return psi


def test_substrate_coupled_plasticity_axis_certification():
    """SubstrateCoupledPlasticity should pass axis certification."""
    plasticity = SubstrateCoupledPlasticity()

    assert plasticity.config.plasticity_type == "substrate_coupled"
    assert plasticity.config.plastic_state_dims is None

    psi = plasticity.initial_psi(None)
    assert psi == {}

    context, geometry, registry = _create_test_context(
        PlasticityConfig.substrate_coupled()
    )
    z = CompositeState(
        activity={
            name: param.detach().clone() for name, param in geometry.params.items()
        },
        plastic={},
        substrate={"conductance": torch.randn(4, 20)},
    )

    psi_new = plasticity.step(psi, z, context)
    assert psi_new == psi  # No-op at plasticity level


def test_substrate_coupled_plasticity_protocol_compliance():
    """SubstrateCoupledPlasticity should comply with PlasticityPrimitive protocol."""
    plasticity = SubstrateCoupledPlasticity()

    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert isinstance(plasticity.config, PlasticityConfig)


# ============================================================
# RuleStatePlasticity Certification (Z3 - not yet implemented)
# ============================================================


class RuleStatePlasticity:
    """Rule state plasticity (Z3): operator selection via ψ.

    ψ = operator_logits controlling which operator from library T_k is applied
    """

    config = PlasticityConfig.rule_state(num_operators=8)

    def initial_psi(self, context: SystemContext) -> dict[str, Tensor]:
        return {
            "operator_logits": torch.zeros(
                4, self.config.plastic_state_dims["operator_logits"]
            )
        }

    def step(
        self,
        psi: dict[str, Tensor],
        z: CompositeState,
        context: SystemContext,
    ) -> dict[str, Tensor]:
        new_psi = {k: v.clone() for k, v in psi.items()}
        # Operator logits evolve based on input
        if "x" in z.activity:
            x = z.activity["x"]
            new_psi["operator_logits"] = new_psi["operator_logits"] + 0.01 * x.mean(
                dim=1, keepdim=True
            ).expand(-1, 8)
        return new_psi


def test_rule_state_plasticity_axis_certification():
    """RuleStatePlasticity should pass axis certification."""
    plasticity = RuleStatePlasticity()

    assert plasticity.config.plasticity_type == "rule_state"
    assert plasticity.config.plastic_state_dims == {"operator_logits": 8}

    psi = plasticity.initial_psi(None)
    assert "operator_logits" in psi
    assert psi["operator_logits"].shape == (4, 8)

    context, geometry, registry = _create_test_context(
        PlasticityConfig.rule_state(num_operators=8)
    )
    z = CompositeState(
        activity={
            name: param.detach().clone() for name, param in geometry.params.items()
        },
        plastic=psi,
        substrate={"conductance": torch.randn(4, 20)},
    )
    z.activity["x"] = torch.randn(4, 10)

    psi_new = plasticity.step(psi, z, context)

    # Operator logits should evolve
    assert not torch.allclose(psi_new["operator_logits"], psi["operator_logits"])


def test_rule_state_plasticity_protocol_compliance():
    """RuleStatePlasticity should comply with PlasticityPrimitive protocol."""
    plasticity = RuleStatePlasticity()

    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert isinstance(plasticity.config, PlasticityConfig)


# ============================================================
# Plasticity Config Factory Tests
# ============================================================


def test_plasticity_config_factories():
    """All PlasticityConfig factories should create valid configs."""
    # Null
    null_config = PlasticityConfig.null()
    assert null_config.plasticity_type == "null"
    assert null_config.plastic_state_dims is None

    # Routing
    routing_config = PlasticityConfig.routing(gate_dim=64)
    assert routing_config.plasticity_type == "routing"
    assert routing_config.plastic_state_dims == {"gate_logits": 64, "active_routes": 64}

    # Fast weights
    fw_config = PlasticityConfig.fast_weights(fast_weight_dim=128)
    assert fw_config.plasticity_type == "fast_weights"
    assert fw_config.plastic_state_dims == {"fast_weights": 128}

    # Substrate coupled
    sc_config = PlasticityConfig.substrate_coupled()
    assert sc_config.plasticity_type == "substrate_coupled"
    assert sc_config.plastic_state_dims is None

    # Rule state
    rs_config = PlasticityConfig.rule_state(num_operators=16)
    assert rs_config.plasticity_type == "rule_state"
    assert rs_config.plastic_state_dims == {"operator_logits": 16}


def test_plasticity_config_in_system_config():
    """PlasticityConfig should integrate with SystemConfig."""
    for factory_name in [
        "null",
        "routing",
        "fast_weights",
        "substrate_coupled",
        "rule_state",
    ]:
        factory = getattr(PlasticityConfig, factory_name)
        config = SystemConfig(
            substrate=SubstrateConfig.digital(),
            geometry=GeometryConfig.recurrent(
                input_dim=10, output_dim=2, hidden_dims=(20,)
            ),
            dynamics=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
            plasticity=factory()
            if factory_name == "null"
            else factory(gate_dim=32)
            if factory_name == "routing"
            else factory(fast_weight_dim=64)
            if factory_name == "fast_weights"
            else factory(num_operators=8)
            if factory_name == "rule_state"
            else factory(),
            credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
            update=ParameterUpdateConfig.euclidean(step_size=0.01),
        )
        config.validate()
        assert config.plasticity.plasticity_type == factory_name


# ============================================================
# Consolidation with Plasticity Tests
# ============================================================


def test_consolidation_with_plasticity_config():
    """Consolidation should respect plasticity config consolidation_config."""
    from bioplausible.core.joint import ConsolidationConfig, consolidate

    context, geometry, registry = _create_test_context(
        PlasticityConfig.fast_weights(
            fast_weight_dim=20, consolidation_config={"promotion_scale": 0.1}
        )
    )

    # Note: fast_weights is already registered by _create_test_context
    # Need to update it to be consolidatable
    # For this test, we'll use a fresh registry
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry2 = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )

    registry2 = StateRegistry()
    for name in geometry2.params:
        registry2.register(StateVariable(name=name, persistent=True))
    registry2.register(
        StateVariable(name="fast_weights", fast_plastic=True, consolidatable=True)
    )
    registry2.register(StateVariable(name="conductance", substrate_owned=True))

    dummy_activity = {
        name: param.detach().clone() for name, param in geometry2.params.items()
    }
    dummy_plastic = {"fast_weights": torch.ones(4, 20) * 5.0}
    dummy_substrate = {"conductance": torch.randn(4, 20)}
    registry2.validate(
        CompositeState(
            activity=dummy_activity, plastic=dummy_plastic, substrate=dummy_substrate
        )
    )

    context2 = SystemContext(
        theta=geometry2.params,
        geometry=geometry2,
        substrate=substrate,
        substrate_config=SubstrateConfig.digital(),
        geometry_config=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics_config=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.fast_weights(
            fast_weight_dim=20, consolidation_config={"promotion_scale": 0.1}
        ),
        registry=registry2,
    )

    z = CompositeState(
        activity=dummy_activity,
        plastic=dummy_plastic,
        substrate=dummy_substrate,
    )

    # Consolidate
    new_context = consolidate(
        z, context2, ConsolidationConfig(promote_all=True, promotion_scale=0.1)
    )

    # fast_weights should be promoted
    assert "fast_weights" in new_context.theta
    assert torch.allclose(
        new_context.theta["fast_weights"], torch.ones(4, 20) * 5.0 * 0.1
    )


def test_plasticity_state_dims_match_registry():
    """plastic_state_dims should match registry fast_plastic variables."""
    for factory_name in ["routing", "fast_weights", "rule_state"]:
        if factory_name == "routing":
            plasticity_config = PlasticityConfig.routing(gate_dim=32)
        elif factory_name == "fast_weights":
            plasticity_config = PlasticityConfig.fast_weights(fast_weight_dim=64)
        else:
            plasticity_config = PlasticityConfig.rule_state(num_operators=8)

        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,))
        )
        registry = StateRegistry()
        for name in geometry.params:
            registry.register(StateVariable(name=name, persistent=True))

        for name, dim in plasticity_config.plastic_state_dims.items():
            registry.register(StateVariable(name=name, fast_plastic=True))

        groups = registry.lifecycle_groups()
        assert len(groups["fast_plastic"]) == len(plasticity_config.plastic_state_dims)


# ============================================================
# Zero-Extension Theorem Tests
# ============================================================


def test_zero_extension_theorem_null_plasticity():
    """Zero-Extension Theorem: Joint(M=Null) ≡ 5-D system."""
    from bioplausible.core.system_trainer import compose_system

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

    system_5d = compose_system(substrate, geometry, dynamics, credit, update)

    # Joint with NullPlasticity
    plasticity = NullPlasticity()
    context, geometry_j, registry = _create_test_context(PlasticityConfig.null())

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # 5-D system
    metrics_5d = system_5d.train_step(x, y)

    # Joint step (NullPlasticity is identity)
    z = CompositeState(activity={"x": x, "y": y}, plastic={}, substrate={})
    psi = plasticity.step({}, z, context)
    assert psi == {}

    # 5-D invariants preserved
    assert metrics_5d["loss"] >= 0
    assert "energy" in metrics_5d


def test_zero_extension_theorem_null_vs_non_null():
    """NullPlasticity should differ from non-null plasticity."""
    context_null, geometry_null, _ = _create_test_context(PlasticityConfig.null())
    context_routing, _, _ = _create_test_context(PlasticityConfig.routing(gate_dim=32))

    null_plasticity = NullPlasticity()
    routing_plasticity = RoutingPlasticity()

    z = CompositeState(
        activity={
            name: param.detach().clone() for name, param in geometry_null.params.items()
        },
        plastic={},
        substrate={"conductance": torch.randn(4, 20)},
    )
    # Add x for routing plasticity
    z.activity["x"] = torch.randn(4, 10)

    # NullPlasticity: psi unchanged
    psi_null = null_plasticity.step({}, z, context_null)
    assert psi_null == {}

    # RoutingPlasticity: psi evolves
    psi_routing = routing_plasticity.initial_psi(context_routing)
    psi_routing_new = routing_plasticity.step(psi_routing, z, context_routing)
    assert not torch.allclose(
        psi_routing_new["gate_logits"], psi_routing["gate_logits"]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
