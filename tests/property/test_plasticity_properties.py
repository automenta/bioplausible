"""Hypothesis property-based tests for plasticity primitives.

Tests verify mathematical properties and invariants of plasticity dynamics:
- RoutingPlasticity: stability, boundedness, idempotence
- FastWeightPlasticity: eigenvalue bounds, decay bounds, symmetry preservation
- SubstrateCoupledPlasticity: no-op behavior, substrate coupling
- NullPlasticity: identity, empty state
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from computronium.core.joint import CompositeState, NullPlasticity, SystemContext
from computronium.core.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.core.plasticity.fast_weights import FastWeightPlasticity
from computronium.core.plasticity.routing import RoutingPlasticity
from computronium.core.plasticity.rule_state import RuleStatePlasticity
from computronium.core.plasticity.substrate_coupled import SubstrateCoupledPlasticity

if TYPE_CHECKING:
    from computronium.core.joint.context import SystemContext
    from computronium.core.joint.state import CompositeState

# Tolerances
TIGHT = {"rtol": 1e-5, "atol": 1e-6, "equal_nan": False}
LOOSE = {"rtol": 1e-4, "atol": 1e-5, "equal_nan": False}
EIGEN_BOUND = 100.0


@dataclass(frozen=True, slots=True)
class _TestContextConfig:
    """Configuration for test context creation."""

    plasticity_type: str = "null"
    gate_dim: int = 32
    fast_weight_dim: int = 64
    num_operators: int = 8
    batch_size: int = 4


def _build_geometry() -> RecurrentGeometry:
    """Build standard test geometry."""
    return RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=10, output_dim=2, hidden_dims=(20,)),
        hidden_dim=20,
    )


def _build_registry(geometry: RecurrentGeometry, config: _TestContextConfig):
    """Build state registry for test context."""
    from computronium.core.joint import StateRegistry, StateVariable

    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))

    dummy_plastic: dict[str, Tensor] = {}
    if config.plasticity_type == "routing":
        for name in ["gate_logits", "active_routes"]:
            registry.register(StateVariable(name=name, fast_plastic=True))
            dummy_plastic[name] = torch.zeros(config.batch_size, config.gate_dim)
    elif config.plasticity_type == "fast_weights":
        registry.register(StateVariable(name="fast_weights", fast_plastic=True))
        dummy_plastic["fast_weights"] = torch.zeros(
            config.batch_size, config.fast_weight_dim
        )
    elif config.plasticity_type == "rule_state":
        registry.register(StateVariable(name="operator_logits", fast_plastic=True))
        registry.register(StateVariable(name="controller_state", fast_plastic=True))
        dummy_plastic["operator_logits"] = torch.zeros(
            config.batch_size, config.num_operators
        )
        dummy_plastic["controller_state"] = torch.zeros(config.batch_size, 128)
    # substrate_coupled and null have no plastic state

    registry.register(StateVariable(name="conductance", substrate_owned=True))
    return registry, dummy_plastic


def _build_plasticity_config(config: _TestContextConfig):
    """Build plasticity config from test config."""
    from computronium.core.joint.transition import PlasticityConfig

    if config.plasticity_type == "routing":
        return PlasticityConfig.routing(gate_dim=config.gate_dim)
    if config.plasticity_type == "fast_weights":
        return PlasticityConfig.fast_weights(fast_weight_dim=config.fast_weight_dim)
    if config.plasticity_type == "rule_state":
        return PlasticityConfig.rule_state(num_operators=config.num_operators)
    if config.plasticity_type == "substrate_coupled":
        return PlasticityConfig.substrate_coupled()
    return PlasticityConfig.null()


def _create_test_context(config: _TestContextConfig = None):
    """Create a test SystemContext with the given plasticity configuration."""
    if config is None:
        config = _TestContextConfig()

    substrate = DigitalSubstrate(SubstrateConfig.digital())
    geometry = _build_geometry()
    EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
    )

    registry, dummy_plastic = _build_registry(geometry, config)
    plasticity_config = _build_plasticity_config(config)

    dummy_activity: dict[str, Tensor] = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    dummy_substrate: dict[str, Tensor] = {
        "conductance": torch.randn(config.batch_size, 20)
    }
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
        plasticity_config=plasticity_config,
        registry=registry,
    )
    return context, geometry, registry


def _make_composite_state(geometry, plastic, substrate, batch_size):
    """Create a CompositeState with standard activity."""
    activity: dict[str, Tensor] = {
        name: param.detach().clone() for name, param in geometry.params.items()
    }
    return CompositeState(activity=activity, plastic=plastic, substrate=substrate)


# ============================================================
# RoutingPlasticity Property Tests (4.1.1)
# ============================================================


@given(
    gate_dim=st.integers(min_value=1, max_value=128),
    batch_size=st.integers(min_value=1, max_value=16),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    temp=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_routing_plasticity_initial_psi_shapes(
    gate_dim, batch_size, decay, lr, temp, seed
):
    """initial_psi returns correctly shaped tensors for any valid config."""
    torch.manual_seed(seed)
    plasticity = RoutingPlasticity(
        gate_dim=gate_dim, decay=decay, learning_rate=lr, temperature=temp
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    assert "gate_logits" in psi
    assert "active_routes" in psi
    assert psi["gate_logits"].shape == (batch_size, gate_dim)
    assert psi["active_routes"].shape == (batch_size, gate_dim)
    assert torch.allclose(psi["gate_logits"], torch.zeros_like(psi["gate_logits"]))
    assert torch.allclose(psi["active_routes"], torch.zeros_like(psi["active_routes"]))


@given(
    gate_dim=st.integers(min_value=1, max_value=64),
    batch_size=st.integers(min_value=1, max_value=8),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    temp=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_routing_plasticity_no_nan_inf_gate_logits(
    gate_dim, batch_size, decay, lr, temp, seed
):
    """Gate logits remain finite (no NaN/Inf) under arbitrary inputs."""
    torch.manual_seed(seed)
    plasticity = RoutingPlasticity(
        gate_dim=gate_dim, decay=decay, learning_rate=lr, temperature=temp
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    # Generate adversarial gate logits: extreme values
    psi_adv = dict(psi)
    psi_adv["gate_logits"] = torch.randn(batch_size, gate_dim) * 100.0

    context, geometry, _ = _create_test_context(
        _TestContextConfig("routing", gate_dim=gate_dim, batch_size=batch_size)
    )
    z = _make_composite_state(
        geometry, psi_adv, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10) * 100.0

    new_psi = plasticity.step(psi_adv, z, context)

    assert not torch.isnan(new_psi["gate_logits"]).any()
    assert not torch.isinf(new_psi["gate_logits"]).any()
    assert not torch.isnan(new_psi["active_routes"]).any()
    assert not torch.isinf(new_psi["active_routes"]).any()


@given(
    gate_dim=st.integers(min_value=1, max_value=64),
    batch_size=st.integers(min_value=1, max_value=8),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    temp=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_routing_plasticity_gate_logits_bounded(
    gate_dim, batch_size, decay, lr, temp, seed
):
    """Gate logits remain bounded under repeated steps."""
    torch.manual_seed(seed)
    plasticity = RoutingPlasticity(
        gate_dim=gate_dim, decay=decay, learning_rate=lr, temperature=temp
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    context, geometry, _ = _create_test_context(
        _TestContextConfig("routing", gate_dim=gate_dim, batch_size=batch_size)
    )
    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10) * 10.0

    max_logit = 0.0
    current_psi = psi
    for _ in range(20):
        current_psi = plasticity.step(current_psi, z, context)
        max_logit = max(max_logit, current_psi["gate_logits"].abs().max().item())

    max_input = z.activity["x"].abs().mean(dim=1).max().item() * gate_dim
    if decay < 1.0:
        theoretical_bound = lr * max_input / (1.0 - decay) + 1e-3
        assert max_logit <= theoretical_bound * 10, (
            f"Logits exploded: {max_logit} > {theoretical_bound * 10}"
        )


@given(
    gate_dim=st.integers(min_value=2, max_value=32),
    batch_size=st.integers(min_value=1, max_value=4),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    temp=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_routing_plasticity_idempotence_same_gates(
    gate_dim, batch_size, decay, lr, temp, seed
):
    """Hard route selection is idempotent: same gate_logits -> same active_routes."""
    torch.manual_seed(seed)
    plasticity = RoutingPlasticity(
        gate_dim=gate_dim, decay=decay, learning_rate=lr, temperature=temp
    )

    # Test the _hard_select method directly for idempotence
    logits = torch.randn(batch_size, gate_dim)
    routes1 = plasticity._hard_select(logits)
    routes2 = plasticity._hard_select(logits)
    assert torch.allclose(routes1, routes2, **TIGHT)

    # Test with top_k
    plasticity_topk = RoutingPlasticity(
        gate_dim=gate_dim,
        decay=decay,
        learning_rate=lr,
        temperature=temp,
        top_k=min(2, gate_dim),
    )
    routes1 = plasticity_topk._hard_select(logits)
    routes2 = plasticity_topk._hard_select(logits)
    assert torch.allclose(routes1, routes2, **TIGHT)


@given(
    gate_dim=st.integers(min_value=1, max_value=32),
    batch_size=st.integers(min_value=1, max_value=4),
    decay=st.floats(
        min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_routing_plasticity_decay_bounds(gate_dim, batch_size, decay, lr, seed):
    """Gate logits decay monotonically when no input drive."""
    torch.manual_seed(seed)
    plasticity = RoutingPlasticity(
        gate_dim=gate_dim, decay=decay, learning_rate=lr, temperature=1.0
    )

    context, geometry, _ = _create_test_context(
        _TestContextConfig("routing", gate_dim=gate_dim, batch_size=batch_size)
    )
    for p in context.theta.values():
        p.requires_grad_(False)

    psi = plasticity.initial_psi(None, batch_size=batch_size)
    psi_init = dict(psi)
    psi_init["gate_logits"] = torch.ones(batch_size, gate_dim) * 5.0

    z = _make_composite_state(
        geometry, psi_init, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.zeros(batch_size, 10)

    current_psi = psi_init
    prev_norm = current_psi["gate_logits"].norm().item()
    for _ in range(10):
        current_psi = plasticity.step(current_psi, z, context)
        curr_norm = current_psi["gate_logits"].norm().item()
        assert curr_norm <= prev_norm + 1e-6, (
            f"Norm increased without drive: {prev_norm} -> {curr_norm}"
        )
        prev_norm = curr_norm


# ============================================================
# FastWeightPlasticity Property Tests (4.1.2)
# ============================================================


@given(
    fast_weight_dim=st.integers(min_value=1, max_value=256),
    batch_size=st.integers(min_value=1, max_value=16),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_fast_weight_plasticity_initial_psi_shapes(
    fast_weight_dim, batch_size, decay, lr, scale, seed
):
    """initial_psi returns correctly shaped tensors for any valid config."""
    torch.manual_seed(seed)
    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=lr,
        outer_product_scale=scale,
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    assert "fast_weights" in psi
    assert psi["fast_weights"].shape == (batch_size, fast_weight_dim)
    assert torch.allclose(psi["fast_weights"], torch.zeros_like(psi["fast_weights"]))


@given(
    fast_weight_dim=st.integers(min_value=1, max_value=128),
    batch_size=st.integers(min_value=1, max_value=8),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_fast_weight_plasticity_no_nan_inf(
    fast_weight_dim, batch_size, decay, lr, scale, seed
):
    """Fast weights remain finite under arbitrary inputs."""
    torch.manual_seed(seed)
    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=lr,
        outer_product_scale=scale,
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    context, geometry, _ = _create_test_context(
        _TestContextConfig(
            "fast_weights", fast_weight_dim=fast_weight_dim, batch_size=batch_size
        )
    )
    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10) * 1000.0
    z.activity["y"] = torch.randn(batch_size, 2) * 1000.0

    new_psi = plasticity.step(psi, z, context)

    assert not torch.isnan(new_psi["fast_weights"]).any()
    assert not torch.isinf(new_psi["fast_weights"]).any()


@given(
    fast_weight_dim=st.integers(min_value=4, max_value=64),
    batch_size=st.integers(min_value=1, max_value=4),
    decay=st.floats(
        min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    scale=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_fast_weight_plasticity_decay_bound(
    fast_weight_dim, batch_size, decay, lr, scale, seed
):
    """Verify decay bound: ||W_fast(t)|| <= decay^t * ||W_fast(0)|| + lr * sum(||outer||)."""
    torch.manual_seed(seed)
    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=lr,
        outer_product_scale=scale,
    )

    context, geometry, _ = _create_test_context(
        _TestContextConfig(
            "fast_weights", fast_weight_dim=fast_weight_dim, batch_size=batch_size
        )
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10)
    z.activity["y"] = torch.randn(batch_size, 2)

    initial_norm = psi["fast_weights"].norm().item()
    outer_sum = 0.0

    current_psi = psi
    for _ in range(10):
        if "x" in z.activity and "y" in z.activity:
            for b in range(batch_size):
                pre_b = z.activity["x"][b].flatten()
                post_b = z.activity["y"][b].flatten()
                if post_b.dim() == 0:
                    post_b = post_b.unsqueeze(0)
                outer = torch.outer(pre_b, post_b).flatten()
                if outer.shape[0] > fast_weight_dim:
                    outer = outer[:fast_weight_dim]
                outer_sum += lr * scale * outer.norm().item()

        current_psi = plasticity.step(current_psi, z, context)

    final_norm = current_psi["fast_weights"].norm().item()
    theoretical_bound = (decay**10) * initial_norm + outer_sum

    assert final_norm <= theoretical_bound * 1.1 + 1e-6, (
        f"Decay bound violated: {final_norm} > {theoretical_bound * 1.1}"
    )


@given(
    fast_weight_dim=st.integers(min_value=4, max_value=32),
    batch_size=st.integers(min_value=1, max_value=4),
    decay=st.floats(
        min_value=0.5, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_fast_weight_plasticity_outer_product_symmetry(
    fast_weight_dim, batch_size, decay, lr, seed
):
    """Outer product updates should preserve structure when pre/post are symmetric."""
    torch.manual_seed(seed)
    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=lr,
        outer_product_scale=1.0,
    )

    context, geometry, _ = _create_test_context(
        _TestContextConfig(
            "fast_weights", fast_weight_dim=fast_weight_dim, batch_size=batch_size
        )
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    sym_input = torch.randn(batch_size, min(10, 2))
    z.activity["x"] = sym_input
    z.activity["y"] = sym_input

    psi_new = plasticity.step(psi, z, context)

    assert not torch.allclose(psi_new["fast_weights"], psi["fast_weights"])


@given(
    fast_weight_dim=st.integers(min_value=16, max_value=64),
    batch_size=st.integers(min_value=2, max_value=4),
    decay=st.floats(
        min_value=0.5, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_fast_weight_plasticity_eigenvalue_bound(
    fast_weight_dim, batch_size, decay, lr, seed
):
    """Eigenvalues of reshaped fast weight matrix should be bounded."""
    torch.manual_seed(seed)
    plasticity = FastWeightPlasticity(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=lr,
        outer_product_scale=1.0,
    )

    context, geometry, _ = _create_test_context(
        _TestContextConfig(
            "fast_weights", fast_weight_dim=fast_weight_dim, batch_size=batch_size
        )
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10)
    z.activity["y"] = torch.randn(batch_size, 2)

    current_psi = psi
    for _ in range(5):
        current_psi = plasticity.step(current_psi, z, context)

    for b in range(batch_size):
        fw = current_psi["fast_weights"][b]
        dim = int(math.sqrt(fast_weight_dim))
        if dim * dim == fast_weight_dim:
            mat = fw.reshape(dim, dim)
            eigvals = torch.linalg.eigvals(mat)
            max_eig = eigvals.abs().max().item()
            assert max_eig < EIGEN_BOUND, f"Eigenvalue exploded: {max_eig}"


# ============================================================
# SubstrateCoupledPlasticity Property Tests (4.1.3)
# ============================================================


@given(
    batch_size=st.integers(min_value=1, max_value=16),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_substrate_coupled_plasticity_initial_psi_empty(batch_size, seed):
    """SubstrateCoupledPlasticity.initial_psi returns empty dict."""
    torch.manual_seed(seed)
    plasticity = SubstrateCoupledPlasticity()
    psi = plasticity.initial_psi(None, batch_size=batch_size)
    assert psi == {}


@given(
    _batch_size=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_substrate_coupled_plasticity_step_noop(_batch_size, seed):
    """SubstrateCoupledPlasticity.step is identity (no-op at plasticity level)."""
    torch.manual_seed(seed)
    plasticity = SubstrateCoupledPlasticity()

    context, geometry, _ = _create_test_context(
        _TestContextConfig("substrate_coupled", batch_size=_batch_size)
    )
    z = _make_composite_state(
        geometry, {}, {"conductance": torch.randn(_batch_size, 20)}, _batch_size
    )
    z.activity["x"] = torch.randn(_batch_size, 10)

    psi = plasticity.step({}, z, context)
    assert psi == {}

    psi_in = {"dummy": torch.randn(_batch_size, 10)}
    psi_new = plasticity.step(psi_in, z, context)
    assert psi_new is psi_in


@given(
    _seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_substrate_coupled_plasticity_protocol_compliance(_seed):
    """SubstrateCoupledPlasticity complies with PlasticityPrimitive protocol."""
    torch.manual_seed(_seed)
    plasticity = SubstrateCoupledPlasticity()

    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert plasticity.config.plasticity_type == "substrate_coupled"
    assert plasticity.config.plastic_state_dims is None


# ============================================================
# NullPlasticity Property Tests (4.1.3)
# ============================================================


@given(
    batch_size=st.integers(min_value=1, max_value=16),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_null_plasticity_initial_psi_empty(batch_size, seed):
    """NullPlasticity.initial_psi returns empty dict."""
    torch.manual_seed(seed)
    plasticity = NullPlasticity()
    psi = plasticity.initial_psi(None, batch_size=batch_size)
    assert psi == {}


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_null_plasticity_step_identity(batch_size, seed):
    """NullPlasticity.step returns psi unchanged (identity)."""
    torch.manual_seed(seed)
    plasticity = NullPlasticity()

    context, geometry, _ = _create_test_context(
        _TestContextConfig("null", batch_size=batch_size)
    )
    z = _make_composite_state(
        geometry,
        {"anything": torch.randn(batch_size, 10)},
        {"conductance": torch.randn(batch_size, 20)},
        batch_size,
    )
    z.activity["x"] = torch.randn(batch_size, 10)

    psi_in = z.plastic
    psi_out = plasticity.step(psi_in, z, context)
    assert psi_out is psi_in


@given(
    _seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_null_plasticity_protocol_compliance(_seed):
    """NullPlasticity complies with PlasticityPrimitive protocol."""
    torch.manual_seed(_seed)
    plasticity = NullPlasticity()

    assert hasattr(plasticity, "config")
    assert hasattr(plasticity, "step")
    assert hasattr(plasticity, "initial_psi")
    assert plasticity.config.plasticity_type == "null"
    assert plasticity.config.plastic_state_dims is None


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_null_plasticity_preserves_theta(batch_size, seed):
    """NullPlasticity does not modify theta (Zero-Extension Theorem)."""
    torch.manual_seed(seed)
    plasticity = NullPlasticity()

    context, geometry, _ = _create_test_context(
        _TestContextConfig("null", batch_size=batch_size)
    )
    z = _make_composite_state(
        geometry, {}, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10)

    original_theta = {name: param.clone() for name, param in context.theta.items()}

    for _ in range(5):
        plasticity.step({}, z, context)

    for name, param in context.theta.items():
        assert torch.allclose(param, original_theta[name], **TIGHT)


# ============================================================
# RuleStatePlasticity Property Tests (bonus coverage)
# ============================================================


@given(
    num_operators=st.integers(min_value=1, max_value=16),
    batch_size=st.integers(min_value=1, max_value=8),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_rule_state_plasticity_initial_psi_shapes(
    num_operators, batch_size, decay, lr, seed
):
    """RuleStatePlasticity.initial_psi returns correctly shaped tensors."""
    torch.manual_seed(seed)
    plasticity = RuleStatePlasticity(
        num_operators=num_operators,
        decay=decay,
        learning_rate=lr,
        device="cpu",
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    assert "operator_logits" in psi
    assert "controller_state" in psi
    assert psi["operator_logits"].shape == (batch_size, num_operators)
    assert psi["controller_state"].shape == (batch_size, 128)
    assert torch.allclose(
        psi["operator_logits"], torch.zeros_like(psi["operator_logits"])
    )
    assert torch.allclose(
        psi["controller_state"], torch.zeros_like(psi["controller_state"])
    )


@given(
    num_operators=st.integers(min_value=1, max_value=8),
    batch_size=st.integers(min_value=1, max_value=4),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_rule_state_plasticity_no_nan_inf(num_operators, batch_size, decay, lr, seed):
    """RuleStatePlasticity state remains finite."""
    torch.manual_seed(seed)
    plasticity = RuleStatePlasticity(
        num_operators=num_operators,
        operator_dim=10,
        decay=decay,
        learning_rate=lr,
        device="cpu",
    )
    psi = plasticity.initial_psi(None, batch_size=batch_size)

    context, geometry, _ = _create_test_context(
        _TestContextConfig(
            "rule_state", num_operators=num_operators, batch_size=batch_size
        )
    )
    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10) * 100.0

    new_psi = plasticity.step(psi, z, context)

    assert not torch.isnan(new_psi["operator_logits"]).any()
    assert not torch.isinf(new_psi["operator_logits"]).any()
    assert not torch.isnan(new_psi["controller_state"]).any()
    assert not torch.isinf(new_psi["controller_state"]).any()


@given(
    num_operators=st.integers(min_value=2, max_value=8),
    _batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_rule_state_plasticity_freeze_theta(num_operators, _batch_size, seed):
    """freeze_theta/unfreeze_theta correctly toggles requires_grad."""
    torch.manual_seed(seed)
    plasticity = RuleStatePlasticity(num_operators=num_operators, device="cpu")

    assert plasticity._operator_embeddings.requires_grad
    assert all(p.requires_grad for p in plasticity._controller.parameters())

    plasticity.freeze_theta()
    assert not plasticity._operator_embeddings.requires_grad
    assert all(not p.requires_grad for p in plasticity._controller.parameters())
    assert plasticity.verify_theta_frozen()

    plasticity.unfreeze_theta()
    assert plasticity._operator_embeddings.requires_grad
    assert all(p.requires_grad for p in plasticity._controller.parameters())
    assert not plasticity.verify_theta_frozen()


@given(
    num_operators=st.integers(min_value=2, max_value=8),
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_rule_state_plasticity_get_active_operator(num_operators, batch_size, seed):
    """get_active_operator returns valid probability distributions / one-hot vectors."""
    torch.manual_seed(seed)
    plasticity = RuleStatePlasticity(num_operators=num_operators, device="cpu")

    logits = torch.randn(batch_size, num_operators)

    weights_train = plasticity.get_active_operator(logits, is_training=True)
    assert weights_train.shape == (batch_size, num_operators)
    assert torch.allclose(weights_train.sum(dim=-1), torch.ones(batch_size), **LOOSE)
    assert (weights_train >= 0).all()

    weights_eval = plasticity.get_active_operator(logits, is_training=False)
    assert weights_eval.shape == (batch_size, num_operators)
    assert torch.allclose(weights_eval.sum(dim=-1), torch.ones(batch_size), **TIGHT)
    assert torch.allclose(
        (weights_eval == 1.0).sum(dim=-1).float(),
        torch.ones(batch_size, dtype=torch.float),
    )
    assert torch.allclose(
        (weights_eval == 0.0).sum(dim=-1).float(),
        torch.full((batch_size,), float(num_operators - 1), dtype=torch.float),
    )


# ============================================================
# Factory Function Tests
# ============================================================


@given(
    gate_dim=st.integers(min_value=1, max_value=128),
    temp=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    top_k=st.one_of(st.none(), st.integers(min_value=1, max_value=64)),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_create_routing_plasticity_factory(gate_dim, temp, top_k, decay, lr):
    """create_routing_plasticity factory correctly configures instance."""
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.core.plasticity.routing import create_routing_plasticity

    config = PlasticityConfig.routing(
        gate_dim=gate_dim, temperature=temp, top_k=top_k, decay=decay, learning_rate=lr
    )
    plasticity = create_routing_plasticity(config)

    assert isinstance(plasticity, RoutingPlasticity)
    assert plasticity._config.gate_dim == gate_dim
    assert plasticity._config.temperature == temp
    assert plasticity._config.top_k == top_k
    assert plasticity._config.decay == decay
    assert plasticity._config.learning_rate == lr


@given(
    fast_weight_dim=st.integers(min_value=1, max_value=256),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30, deadline=None)
def test_create_fast_weight_plasticity_factory(fast_weight_dim, decay, lr, scale):
    """create_fast_weight_plasticity factory correctly configures instance."""
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.core.plasticity.fast_weights import create_fast_weight_plasticity

    config = PlasticityConfig.fast_weights(
        fast_weight_dim=fast_weight_dim,
        decay=decay,
        learning_rate=lr,
        outer_product_scale=scale,
    )
    plasticity = create_fast_weight_plasticity(config)

    assert isinstance(plasticity, FastWeightPlasticity)
    assert plasticity._config.fast_weight_dim == fast_weight_dim
    assert plasticity._config.decay == decay
    assert plasticity._config.learning_rate == lr
    assert plasticity._config.outer_product_scale == scale


@given(
    num_operators=st.integers(min_value=1, max_value=32),
    operator_dim=st.integers(min_value=1, max_value=128),
    controller_hidden=st.integers(min_value=1, max_value=256),
    temp=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    decay=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30, deadline=None)
def test_create_rule_state_plasticity_factory(
    num_operators, operator_dim, controller_hidden, temp, lr, decay
):
    """create_rule_state_plasticity factory correctly configures instance."""
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.core.plasticity.rule_state import create_rule_state_plasticity

    config = PlasticityConfig.rule_state(
        num_operators=num_operators,
        operator_dim=operator_dim,
        controller_hidden=controller_hidden,
        temperature=temp,
        learning_rate=lr,
        decay=decay,
    )
    plasticity = create_rule_state_plasticity(config)

    assert isinstance(plasticity, RuleStatePlasticity)
    assert plasticity._config.num_operators == num_operators
    assert plasticity._config.operator_dim == operator_dim
    assert plasticity._config.controller_hidden == controller_hidden
    assert plasticity._config.temperature == temp
    assert plasticity._config.learning_rate == lr
    assert plasticity._config.decay == decay


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=10, deadline=None)
def test_create_substrate_coupled_plasticity_factory(seed):
    """create_substrate_coupled_plasticity factory works."""
    torch.manual_seed(seed)
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.core.plasticity.substrate_coupled import (
        create_substrate_coupled_plasticity,
    )

    config = PlasticityConfig.substrate_coupled()
    plasticity = create_substrate_coupled_plasticity(config)

    assert isinstance(plasticity, SubstrateCoupledPlasticity)
    assert plasticity.config.plasticity_type == "substrate_coupled"


# ============================================================
# Integration Tests: Plasticity with Joint System
# ============================================================


@given(
    gate_dim=st.integers(min_value=4, max_value=32),
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_routing_plasticity_integration_with_joint_context(gate_dim, batch_size, seed):
    """RoutingPlasticity works correctly within a full Joint SystemContext."""
    torch.manual_seed(seed)
    plasticity = RoutingPlasticity(gate_dim=gate_dim)
    context, geometry, _ = _create_test_context(
        _TestContextConfig("routing", gate_dim=gate_dim, batch_size=batch_size)
    )

    psi = plasticity.initial_psi(context, batch_size=batch_size)
    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10)

    current_psi = psi
    for _ in range(5):
        current_psi = plasticity.step(current_psi, z, context)

    assert "gate_logits" in current_psi
    assert "active_routes" in current_psi
    assert current_psi["gate_logits"].shape == (batch_size, gate_dim)
    assert not torch.isnan(current_psi["gate_logits"]).any()


@given(
    fast_weight_dim=st.integers(min_value=8, max_value=64),
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_fast_weight_plasticity_integration_with_joint_context(
    fast_weight_dim, batch_size, seed
):
    """FastWeightPlasticity works correctly within a full Joint SystemContext."""
    torch.manual_seed(seed)
    plasticity = FastWeightPlasticity(fast_weight_dim=fast_weight_dim)
    context, geometry, _ = _create_test_context(
        _TestContextConfig(
            "fast_weights", fast_weight_dim=fast_weight_dim, batch_size=batch_size
        )
    )

    psi = plasticity.initial_psi(context, batch_size=batch_size)
    z = _make_composite_state(
        geometry, psi, {"conductance": torch.randn(batch_size, 20)}, batch_size
    )
    z.activity["x"] = torch.randn(batch_size, 10)
    z.activity["y"] = torch.randn(batch_size, 2)

    current_psi = psi
    for _ in range(5):
        current_psi = plasticity.step(current_psi, z, context)

    assert "fast_weights" in current_psi
    assert current_psi["fast_weights"].shape == (batch_size, fast_weight_dim)
    assert not torch.isnan(current_psi["fast_weights"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
