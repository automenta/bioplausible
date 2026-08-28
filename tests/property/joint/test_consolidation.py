"""Property tests for episode-boundary consolidation."""

from __future__ import annotations

import torch
from torch import Tensor

from computronium.core.joint import (
    CompositeState,
    ConsolidationConfig,
    PlasticityConfig,
    StateRegistry,
    StateVariable,
    SystemContext,
    consolidate,
)
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
)


def _create_context_with_consolidatable() -> tuple[SystemContext, dict[str, Tensor]]:
    """Create a context with consolidatable plastic variables."""
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    # Add a consolidatable plastic variable
    registry.register(
        StateVariable(name="fast_weights", fast_plastic=True, consolidatable=True)
    )

    theta = dict(geometry.params)
    theta["fast_weights"] = torch.zeros(
        20, 20, requires_grad=True
    )  # Base value for consolidation

    return (
        SystemContext(
            theta=theta,
            geometry=geometry,
            substrate=substrate,
            substrate_config=SubstrateConfig.digital(),
            geometry_config=GeometryConfig.recurrent(
                input_dim=10, output_dim=2, hidden_dims=(20,)
            ),
            dynamics_config=StateDynamicsConfig.energy_minimization(
                max_steps=5, beta=0.5
            ),
            credit_config=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
            update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
            plasticity_config=PlasticityConfig.null(),
            registry=registry,
        ),
        theta,
    )


def test_consolidation_promotes_consolidatable():
    """Consolidation promotes consolidatable ψ to θ at episode boundaries."""
    context, original_theta = _create_context_with_consolidatable()

    z_final = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"fast_weights": torch.ones(20, 20) * 0.5},  # ψ = 0.5
        substrate={},
    )

    new_context = consolidate(
        z_final, context, ConsolidationConfig(promotion_scale=1.0)
    )

    # fast_weights should be promoted: θ_new = θ_old + ψ
    assert "fast_weights" in new_context.theta
    expected = original_theta["fast_weights"] + torch.ones(20, 20) * 0.5
    assert torch.allclose(new_context.theta["fast_weights"], expected)


def test_consolidation_scale():
    """Consolidation respects promotion_scale."""
    context, original_theta = _create_context_with_consolidatable()

    z_final = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"fast_weights": torch.ones(20, 20)},
        substrate={},
    )

    # Scale 0.5: θ_new = θ_old + 0.5 * ψ
    new_context = consolidate(
        z_final, context, ConsolidationConfig(promotion_scale=0.5)
    )
    expected = original_theta["fast_weights"] + torch.ones(20, 20) * 0.5
    assert torch.allclose(new_context.theta["fast_weights"], expected)


def test_consolidation_resets_plastic():
    """Consolidation resets promoted plastic state if configured."""
    context, _ = _create_context_with_consolidatable()

    z_final = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"fast_weights": torch.ones(20, 20)},
        substrate={},
    )

    # With reset_plastic=True (default)
    new_context = consolidate(z_final, context, ConsolidationConfig(reset_plastic=True))
    assert torch.allclose(z_final.plastic["fast_weights"], torch.zeros(20, 20))

    # Without reset
    z_final2 = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"fast_weights": torch.ones(20, 20) * 2},
        substrate={},
    )
    new_context2 = consolidate(
        z_final2, context, ConsolidationConfig(reset_plastic=False)
    )
    assert torch.allclose(z_final2.plastic["fast_weights"], torch.ones(20, 20) * 2)


def test_consolidation_only_promotes_consolidatable():
    """Consolidation only promotes variables marked consolidatable."""
    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    # Non-consolidatable plastic
    registry.register(StateVariable(name="eligibility", fast_plastic=True))

    theta = dict(geometry.params)

    context = SystemContext(
        theta=theta,
        geometry=geometry,
        substrate=substrate,
        substrate_config=SubstrateConfig.digital(),
        geometry_config=GeometryConfig.recurrent(
            input_dim=10, output_dim=2, hidden_dims=(20,)
        ),
        dynamics_config=StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.null(),
        registry=registry,
    )

    z_final = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"eligibility": torch.ones(20, 20)},
        substrate={},
    )

    # eligibility is not consolidatable, should not be promoted
    new_context = consolidate(z_final, context)
    assert "eligibility" not in new_context.theta
    assert len(new_context.theta) == len(context.theta)


def test_consolidation_creates_new_context():
    """Consolidation returns new SystemContext (immutability)."""
    context, _ = _create_context_with_consolidatable()

    z_final = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"fast_weights": torch.ones(20, 20)},
        substrate={},
    )

    new_context = consolidate(z_final, context)

    # New context object
    assert new_context is not context
    # Theta should be different
    assert new_context.theta is not context.theta
    # Other fields should be same (immutable)
    assert new_context.geometry is context.geometry
    assert new_context.substrate is context.substrate


def test_consolidation_config_defaults():
    """ConsolidationConfig defaults."""
    config = ConsolidationConfig()
    assert config.promote_all is True
    assert config.promotion_scale == 1.0
    assert config.reset_plastic is True
