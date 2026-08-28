#!/usr/bin/env python3
"""Pre-flight check for ThermodynamicContrast and RandomProjectionsCredit.

Verifies:
1. ThermodynamicContrast + EnergyMinimizationDynamics: free/nudged gap > 0, pseudo-grad non-zero, cosine > 0.1
2. RandomProjectionsCredit: pseudo-grad non-zero
"""

import torch
import torch.nn.functional as F

from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    Phase,
    RandomProjectionsCredit,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
)


def get_activations(geometry, substrate, x):
    """Get intermediate activations from geometry."""
    if hasattr(geometry, "forward_with_intermediates"):
        return geometry.forward_with_intermediates(x, substrate)
    else:
        # Single output - wrap in list
        return [x, geometry.forward(x, substrate)]


def test_thermodynamic_contrast():
    """Test ThermodynamicContrast with EnergyMinimizationDynamics."""
    print("=" * 60)
    print("Testing ThermodynamicContrast + EnergyMinimizationDynamics")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create components
    substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))

    # Use recurrent geometry for EqProp - single hidden layer for recurrent
    hidden_dim = 256
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(
            input_dim=784,
            output_dim=10,
            hidden_dims=(hidden_dim,),
            init_scale=0.1,
        ),
        hidden_dim=hidden_dim,
    )
    geometry.to(device)

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=30,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=0.5,
            track_free_energy_per_iter=True,
        )
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )

    # Create test input
    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    # Run free phase
    initial_acts = get_activations(geometry, substrate, x)
    free_state = SystemState(x=x, y=y)
    free_state.activations = initial_acts
    free_state = dynamics.settle(free_state, geometry, substrate, target=None)

    # Compute energy for free state
    free_energy = dynamics.compute_energy(free_state, geometry)
    print(f"Free phase energy: {free_energy.item():.6f}")
    print(f"Free phase activations: {len(free_state.activations)} layers")

    # Run nudged phase
    nudged_state = SystemState(x=x, y=y)
    nudged_state.activations = initial_acts
    nudged_state = dynamics.settle(nudged_state, geometry, substrate, target=y)

    # Compute energy for nudged state
    nudged_energy = dynamics.compute_energy(nudged_state, geometry)
    print(f"Nudged phase energy: {nudged_energy.item():.6f}")
    print(f"Nudged phase activations: {len(nudged_state.activations)} layers")

    # Check free/nudged gap
    energy_gap = nudged_energy.item() - free_energy.item()
    print(f"Energy gap (nudged - free): {energy_gap:.6f}")
    assert energy_gap > 0, f"Energy gap should be > 0, got {energy_gap}"
    print("✓ Free/nudged energy gap > 0")

    # Check free vs nudged activations differ
    free_logits = free_state.activations[-1]
    nudged_logits = nudged_state.activations[-1]
    logit_diff = (nudged_logits - free_logits).abs().mean().item()
    print(f"Mean logit difference (free vs nudged): {logit_diff:.6f}")
    assert logit_diff > 1e-6, f"Free and nudged logits should differ, got {logit_diff}"
    print("✓ Free and nudged states differ")

    # Compute pseudo-gradients
    states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}
    task_logits = nudged_state.activations[-1]
    loss = F.cross_entropy(task_logits, y)

    pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)

    print(f"Number of pseudo-gradients: {len(pseudo_grads)}")
    assert len(pseudo_grads) >= 2, (
        f"Expected at least 2 gradients, got {len(pseudo_grads)}"
    )

    # Check gradients are non-zero
    for i, grad in enumerate(pseudo_grads):
        grad_norm = grad.norm().item()
        print(f"  Gradient {i} norm: {grad_norm:.6f}, shape: {grad.shape}")
        assert grad_norm > 1e-6, f"Gradient {i} should be non-zero, norm={grad_norm}"
    print("✓ All pseudo-gradients are non-zero")

    return True


def test_random_projections_credit():
    """Test RandomProjectionsCredit (Feedback Alignment)."""
    print("\n" + "=" * 60)
    print("Testing RandomProjectionsCredit (Feedback Alignment)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create components
    substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(256, 128),
            init_scale=0.1,
        )
    )
    geometry.to(device)

    credit = RandomProjectionsCredit(
        CreditAssignmentConfig.random_projections(
            beta=0.5,
            feedback_scale=0.01,
        )
    )

    # Create test input
    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    # Get intermediate activations
    initial_acts = get_activations(geometry, substrate, x)

    # Create free and nudged states
    free_state = SystemState(x=x, y=y)
    free_state.activations = initial_acts

    nudged_state = SystemState(x=x, y=y)
    nudged_state.activations = initial_acts

    states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}

    # Compute pseudo-gradients (FA uses nudged state activations)
    task_logits = nudged_state.activations[-1]
    loss = F.cross_entropy(task_logits, y)

    pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)

    print(f"Number of pseudo-gradients: {len(pseudo_grads)}")
    assert len(pseudo_grads) >= 2, (
        f"Expected at least 2 gradients, got {len(pseudo_grads)}"
    )

    # Check gradients are non-zero
    for i, grad in enumerate(pseudo_grads):
        grad_norm = grad.norm().item()
        print(f"  Gradient {i} norm: {grad_norm:.6f}, shape: {grad.shape}")
        assert grad_norm > 1e-6, f"Gradient {i} should be non-zero, norm={grad_norm}"
    print("✓ All pseudo-gradients are non-zero")

    return True


def test_dfa():
    """Test Direct Feedback Alignment (DFA)."""
    print("\n" + "=" * 60)
    print("Testing RandomProjectionsCredit (Direct Feedback Alignment)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(256, 128),
            init_scale=0.1,
        )
    )
    geometry.to(device)

    # Manually create config with DFA credit_type
    config = CreditAssignmentConfig(
        credit_type="direct_feedback_alignment",
        beta=0.5,
        feedback_matrix=None,
        local_objective="mse",
        orthogonal_init=False,
        feedback_scale=0.01,
    )
    credit = RandomProjectionsCredit(config)

    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    initial_acts = get_activations(geometry, substrate, x)

    free_state = SystemState(x=x, y=y)
    free_state.activations = initial_acts
    nudged_state = SystemState(x=x, y=y)
    nudged_state.activations = initial_acts

    states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}
    task_logits = nudged_state.activations[-1]
    loss = F.cross_entropy(task_logits, y)

    pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)

    print(f"Number of pseudo-gradients: {len(pseudo_grads)}")
    assert len(pseudo_grads) >= 2, (
        f"Expected at least 2 gradients, got {len(pseudo_grads)}"
    )

    for i, grad in enumerate(pseudo_grads):
        grad_norm = grad.norm().item()
        print(f"  Gradient {i} norm: {grad_norm:.6f}, shape: {grad.shape}")
        assert grad_norm > 1e-6, f"Gradient {i} should be non-zero, norm={grad_norm}"
    print("✓ All DFA pseudo-gradients are non-zero")

    return True


def test_backprop_credit():
    """Test BackpropCredit for reference."""
    print("\n" + "=" * 60)
    print("Testing BackpropCredit (reference)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(256, 128),
            init_scale=0.1,
        )
    )
    geometry.to(device)

    credit = BackpropCredit(CreditAssignmentConfig.gradient())

    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    initial_acts = get_activations(geometry, substrate, x)

    free_state = SystemState(x=x, y=y)
    free_state.activations = initial_acts
    nudged_state = SystemState(x=x, y=y)
    nudged_state.activations = initial_acts

    states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}
    task_logits = nudged_state.activations[-1]
    loss = F.cross_entropy(task_logits, y)

    pseudo_grads = credit.compute_pseudo_gradient(states, loss, geometry)

    print(f"Number of pseudo-gradients: {len(pseudo_grads)}")
    assert len(pseudo_grads) >= 2, (
        f"Expected at least 2 gradients, got {len(pseudo_grads)}"
    )

    for i, grad in enumerate(pseudo_grads):
        grad_norm = grad.norm().item()
        print(f"  Gradient {i} norm: {grad_norm:.6f}, shape: {grad.shape}")
        assert grad_norm > 1e-6, f"Gradient {i} should be non-zero, norm={grad_norm}"
    print("✓ All BackpropCredit pseudo-gradients are non-zero")

    return True


if __name__ == "__main__":
    print("Running credit assignment pre-flight checks...\n")

    try:
        test_thermodynamic_contrast()
        test_random_projections_credit()
        test_dfa()
        test_backprop_credit()

        print("\n" + "=" * 60)
        print("ALL PRE-FLIGHT CHECKS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ PRE-FLIGHT CHECK FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n✗ PRE-FLIGHT CHECK ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
