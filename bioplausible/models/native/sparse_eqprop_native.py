"""Native Sparse Equilibrium Propagation using 5-D Ontology composition.

This replaces the legacy SparseEquilibrium variant with a direct composition
of the 5 Protocols, using SparseSubstrate for dynamic sparsity masks and
efficient sparse matrix multiplication.
"""

from __future__ import annotations

from bioplausible.core.ontology import (
    CreditAssignmentConfig,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    System,
    ThermodynamicContrast,
)
from bioplausible.core.substrates.sparse_substrate import SparseSubstrate
from bioplausible.core.system_trainer import compose_system


def create_native_sparse_eqprop(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    sparsity: float = 0.5,
    sparsity_type: str = "unstructured",
    n_m_ratio: tuple[int, int] = (2, 4),
    block_size: tuple[int, int] = (8, 8),
    update_mask_frequency: int = 100,
    prune_criterion: str = "magnitude",
    regrow_criterion: str = "gradient",
    **kwargs,
) -> System:
    """Create a Sparse Equilibrium Propagation system using native 5-D composition.

    Sparse Equilibrium uses dynamic sparsity masks during settling and learning,
    enabling training on sparsity-constrained hardware (e.g., N:M structured
    sparsity on Ampere+ GPUs, block sparsity on accelerators).

    The substrate maintains binary masks for each weight matrix and applies
    sparse matrix multiplication where supported, falling back to dense masked
    matmul when sparse kernels are unavailable.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based methods
        settle_steps: Maximum settling iterations
        lr: Learning rate
        sparsity: Target sparsity ratio [0, 1]
        sparsity_type: "unstructured", "n_m", "block", or "channel"
        n_m_ratio: (N, M) for N:M structured sparsity (e.g., 2:4)
        block_size: (height, width) for block sparsity
        update_mask_frequency: Steps between mask updates
        prune_criterion: "magnitude", "gradient", "random", or "snip"
        regrow_criterion: "gradient" or "random"
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with SparseSubstrate + RecurrentGeometry
        + EnergyMinimizationDynamics + ThermodynamicContrast + EuclideanUpdate
    """
    hidden_dims = tuple([hidden_dim] * max(num_layers, 1))

    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        init_scale=0.1,
    )

    # Use SparseSubstrate for dynamic sparsity
    substrate = SparseSubstrate(
        sparsity_type=sparsity_type,  # type: ignore[arg-type]
        n_m_ratio=n_m_ratio,
        block_size=block_size,
        update_mask_frequency=update_mask_frequency,
        prune_criterion=prune_criterion,  # type: ignore[arg-type]
        regrow_criterion=regrow_criterion,  # type: ignore[arg-type]
    )
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(
            beta=beta,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


# Alias for registry registration
native_sparse_eqprop = create_native_sparse_eqprop
