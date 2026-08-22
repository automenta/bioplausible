"""Property tests for StateRegistry and CompositeState."""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.core.joint import (
    CompositeState,
    StateRegistry,
    StateVariable,
)


def test_state_variable_creation():
    """Test StateVariable lifecycle roles."""
    # Persistent only (θ)
    theta_var = StateVariable(name="weight", persistent=True)
    assert theta_var.persistent
    assert not theta_var.fast_plastic

    # Fast plastic only (ψ)
    psi_var = StateVariable(name="eligibility", fast_plastic=True)
    assert psi_var.fast_plastic
    assert not psi_var.persistent

    # Substrate owned only (σ)
    sigma_var = StateVariable(name="conductance", substrate_owned=True)
    assert sigma_var.substrate_owned
    assert not sigma_var.persistent

    # Consolidatable (ψ → θ)
    consol_var = StateVariable(name="fast_weight", fast_plastic=True, consolidatable=True)
    assert consol_var.consolidatable
    assert consol_var.fast_plastic

    # Invalid: consolidatable without fast_plastic
    try:
        StateVariable(name="bad", consolidatable=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Invalid: no lifecycle role
    try:
        StateVariable(name="bad")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_state_registry_register():
    """Test StateRegistry registration and lookup."""
    registry = StateRegistry()

    registry.register(StateVariable(name="weight", persistent=True))
    registry.register(StateVariable(name="eligibility", fast_plastic=True))
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    assert "weight" in registry
    assert "eligibility" in registry
    assert "conductance" in registry
    assert len(registry) == 3

    # Duplicate registration should fail
    try:
        registry.register(StateVariable(name="weight", persistent=True))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_state_registry_lifecycle_groups():
    """Test lifecycle group extraction."""
    registry = StateRegistry()
    registry.register(StateVariable(name="weight", persistent=True))
    registry.register(StateVariable(name="eligibility", fast_plastic=True, consolidatable=True))
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    groups = registry.lifecycle_groups()

    assert "weight" in groups["persistent"]
    assert "eligibility" in groups["fast_plastic"]
    assert "eligibility" in groups["consolidatable"]
    assert "conductance" in groups["substrate_owned"]


def test_composite_state_creation():
    """Test CompositeState creation and manipulation."""
    x = torch.randn(4, 10)
    psi = torch.randn(4, 20)
    sigma = torch.randn(4, 5)

    z = CompositeState(
        activity={"x": x, "hidden": torch.randn(4, 20)},
        plastic={"eligibility": psi},
        substrate={"conductance": sigma},
    )

    assert "x" in z.activity
    assert "eligibility" in z.plastic
    assert "conductance" in z.substrate


def test_composite_state_clone():
    """Test CompositeState deep clone with detached tensors."""
    z = CompositeState(
        activity={"x": torch.randn(4, 10, requires_grad=True)},
        plastic={"psi": torch.randn(4, 10, requires_grad=True)},
        substrate={"sigma": torch.randn(4, 10, requires_grad=True)},
    )

    z_clone = z.clone()

    # Cloned tensors should be detached
    assert not z_clone.activity["x"].requires_grad
    assert not z_clone.plastic["psi"].requires_grad
    assert not z_clone.substrate["sigma"].requires_grad

    # Values should match
    assert torch.allclose(z_clone.activity["x"], z.activity["x"])


def test_composite_state_detach():
    """Test in-place detach."""
    z = CompositeState(
        activity={"x": torch.randn(4, 10, requires_grad=True)},
        plastic={"psi": torch.randn(4, 10, requires_grad=True)},
        substrate={"sigma": torch.randn(4, 10, requires_grad=True)},
    )

    z.detach_()

    assert not z.activity["x"].requires_grad
    assert not z.plastic["psi"].requires_grad
    assert not z.substrate["sigma"].requires_grad


def test_composite_state_to_device():
    """Test device transfer."""
    z = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"psi": torch.randn(4, 10)},
        substrate={"sigma": torch.randn(4, 10)},
    )

    z_cpu = z.to("cpu")
    assert z_cpu.activity["x"].device.type == "cpu"
    assert z_cpu.plastic["psi"].device.type == "cpu"
    assert z_cpu.substrate["sigma"].device.type == "cpu"


def test_state_registry_validate():
    """Test StateRegistry validates CompositeState."""
    registry = StateRegistry()
    registry.register(StateVariable(name="weight", persistent=True))
    registry.register(StateVariable(name="eligibility", fast_plastic=True))
    registry.register(StateVariable(name="conductance", substrate_owned=True))

    # Valid state
    z_valid = CompositeState(
        activity={"weight": torch.randn(10, 10)},
        plastic={"eligibility": torch.randn(10, 10)},
        substrate={"conductance": torch.randn(10, 10)},
    )
    registry.validate(z_valid)  # Should not raise

    # Missing persistent
    z_missing = CompositeState(
        activity={},
        plastic={"eligibility": torch.randn(10, 10)},
        substrate={"conductance": torch.randn(10, 10)},
    )
    try:
        registry.validate(z_missing)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_composite_state_empty():
    """Test empty state creation."""
    z = CompositeState.empty()
    assert z.activity == {}
    assert z.plastic == {}
    assert z.substrate == {}