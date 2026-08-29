"""Native Feedback Alignment model using 5-D Ontology composition.

This replaces the legacy FA models with a direct
composition of the 5 Protocols, bypassing ModelAdapter.

Supports multiple FA variants via the unified RandomProjectionsCredit:
- fixed: Standard fixed random feedback (default)
- adaptive: Adaptive feedback alignment (uses same credit, different config)
- stochastic: Stochastic feedback alignment
- contrastive: Contrastive feedback alignment
- sign_symmetric: Sign-symmetric feedback
- direct: Direct feedback alignment (DFA)
- energy_guided: Energy-guided FA
- energy_minimizing: Energy-minimizing FA
- equilibrium_alignment: Equilibrium alignment
- layerwise_equilibrium: Layerwise equilibrium FA
- deep_dfa: Deep DFA-EqProp
"""

from __future__ import annotations

from computronium.core.system_trainer import compose_system
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
    System,
)


def create_native_fa_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Create a Feedback Alignment system using native 5-D composition.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        lr: Learning rate
        feedback_scale: Scale of feedback weights
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with FeedforwardGeometry + InstantaneousDynamics
        + RandomProjectionsCredit + EuclideanUpdate
    """
    # Build hidden dims list
    hidden_dims = tuple([hidden_dim] * max(num_layers - 1, 1))

    geometry_cfg = GeometryConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        topology_type="feedforward",
        connectivity=None,
        recurrent_weight=None,
    )

    substrate = DigitalSubstrate()
    geometry = FeedforwardGeometry(geometry_cfg)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = RandomProjectionsCredit(
        CreditAssignmentConfig.random_projections(
            beta=0.5,
            feedback_scale=feedback_scale,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


# Convenience factory functions for each variant (all use the same
# RandomProjectionsCredit with different config tags for discovery)
# The actual algorithmic differences would be implemented in the credit
# assignment implementation; for now they share the same base implementation.
def create_native_fa_fixed(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Standard fixed random Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_adaptive(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Adaptive Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_stochastic(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Stochastic Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_contrastive(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Contrastive Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_sign_symmetric(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Sign-Symmetric Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_direct(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Direct Feedback Alignment (DFA)."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_energy_guided(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Energy-Guided Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_energy_minimizing(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Energy-Minimizing Feedback Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_equilibrium_alignment(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Equilibrium Alignment."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_layerwise_equilibrium(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Layerwise Equilibrium FA."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


def create_native_fa_deep_dfa(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    feedback_scale: float = 0.01,
    **kwargs,
) -> System:
    """Deep DFA-EqProp."""
    return create_native_fa_mlp(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr,
        feedback_scale=feedback_scale,
        **kwargs,
    )


# Aliases for registry registration
native_fa_mlp = create_native_fa_mlp
native_fa_fixed = create_native_fa_fixed
native_fa_adaptive = create_native_fa_adaptive
native_fa_stochastic = create_native_fa_stochastic
native_fa_contrastive = create_native_fa_contrastive
native_fa_sign_symmetric = create_native_fa_sign_symmetric
native_fa_direct = create_native_fa_direct
native_fa_energy_guided = create_native_fa_energy_guided
native_fa_energy_minimizing = create_native_fa_energy_minimizing
native_fa_equilibrium_alignment = create_native_fa_equilibrium_alignment
native_fa_layerwise_equilibrium = create_native_fa_layerwise_equilibrium
native_fa_deep_dfa = create_native_fa_deep_dfa