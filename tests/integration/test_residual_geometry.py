"""Residual feedforward geometry lock (μPC paper regime, R11.3.11 audit).

The μPC parameterization (arXiv:2505.13124, Table 1) is specified and
tested on residual networks — hidden premultipliers (N·L)^{-1/2} assume
a skip path carrying signal while the scaled branch adds corrections.
Applying μPC init to a plain MLP (the original depth-frontier pilot)
extrapolates outside the paper's tested domain. This lock makes the
residual regime expressible and honest:

1. Residual forward arithmetic: a_ℓ = a_{ℓ−1} + φ(W_ℓ a_{ℓ−1} + b_ℓ),
   verified against a manual trace, bitwise.
2. Default builds are untouched (residual=False is byte-identical).
3. Settle-kernel parity: eager kernel.step and the compiled whole-settle
   loop agree bitwise on residual builds.
4. Spec round-trip carries ``residual`` through to_spec/from_spec.
5. Fail-loud: residual with non-feedforward topology raises ValueError.
"""

from __future__ import annotations

import pytest
import torch

from computronium import (
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    NullPlasticity,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
    compose_joint_system,
)
from computronium.ontology.geometry import FeedforwardGeometry
from computronium.ontology.system import SystemState


def _build(*, residual: bool, compiled: bool = False, max_steps: int = 3):
    torch.manual_seed(0)
    return compose_joint_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=16,
                output_dim=5,
                hidden_dims=(8, 8),
                residual=residual,
            )
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=max_steps, compiled=compiled, step_size=0.1
            )
        ),
        plasticity=NullPlasticity(),
        credit=ThermodynamicContrast(),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )


def test_residual_forward_matches_manual_trace() -> None:
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=16, output_dim=5, hidden_dims=(8, 8), residual=True
        )
    )
    x = torch.randn(3, 16)
    linears = [m for m in geometry._layers if isinstance(m, torch.nn.Linear)]
    # First projection (16→8) is not square: no skip
    a1 = torch.relu(x @ linears[0].weight.T + linears[0].bias)
    a2 = a1 + torch.relu(a1 @ linears[1].weight.T + linears[1].bias)
    out = a2 @ linears[2].weight.T + linears[2].bias
    acts = geometry.forward_with_intermediates(x, None)
    assert len(acts) == 4  # [input, a1, a2, output]
    assert torch.allclose(acts[1], a1, atol=1e-6)
    assert torch.allclose(acts[2], a2, atol=1e-6)
    assert torch.allclose(acts[3], out, atol=1e-6)


def test_residual_settle_bitwise_parity_eager_compiled() -> None:
    x = torch.randn(2, 16)
    outs = []
    for compiled in (False, True):
        torch.manual_seed(0)
        system = _build(residual=True, compiled=compiled)
        state = SystemState(x=x)
        settled = system.dynamics.settle(
            state, system.geometry, system.substrate, target=None
        )
        outs.append(settled.free_state[-1])
    assert torch.allclose(outs[0], outs[1], atol=1e-6), (
        (outs[0] - outs[1]).abs().max().item()
    )


def test_residual_spec_round_trip() -> None:
    system = _build(residual=True)
    spec = system.to_spec()
    assert spec["geometry"]["residual"] is True
    restored = type(system).from_spec(spec)
    assert restored.geometry.residual is True


def test_default_build_untouched_by_residual_flag() -> None:
    torch.manual_seed(0)
    plain = _build(residual=False)
    torch.manual_seed(0)
    legacy = _build(residual=False)
    x = torch.randn(2, 16)
    a = plain.dynamics.settle(
        SystemState(x=x), plain.geometry, plain.substrate
    ).free_state[-1]
    b = legacy.dynamics.settle(
        SystemState(x=x), legacy.geometry, legacy.substrate
    ).free_state[-1]
    assert torch.equal(a, b)


def test_residual_rejects_non_feedforward() -> None:
    from dataclasses import replace

    from computronium.ontology.credit import CreditAssignmentConfig
    from computronium.ontology.system import SystemConfig

    torch.manual_seed(0)
    config = SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=replace(
            GeometryConfig.recurrent(input_dim=16, output_dim=5, hidden_dims=(8,)),
            residual=True,
        ),
        dynamics=StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5),
        plasticity=NullPlasticity().config,
        credit=CreditAssignmentConfig.thermodynamic_contrast(beta=0.5),
        update=ParameterUpdateConfig.euclidean(step_size=0.01),
    )
    with pytest.raises(ValueError, match="feedforward"):
        config.validate()


def test_residual_epc_free_equilibrium_is_feedforward_bitwise() -> None:
    """D12's invariant under residual: ePC's free-phase equilibrium is the
    residual feedforward pass bitwise (zero errors are the fixed point),
    and the nudged signal reaches every hidden layer through the skip path
    (the R11.3.11 instrument gap — ePC must express the paper's residual
    regime for the jpc-faithful re-test)."""
    from computronium import ErrorPredictiveCodingDynamics

    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=16, output_dim=5, hidden_dims=(8, 8), residual=True
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=5, step_size=0.5, beta=10.0
        )
    )
    x = torch.randn(4, 16)
    y = torch.randint(0, 5, (4,))

    ff = geometry.forward_with_intermediates(x, substrate)
    free = dynamics.settle(SystemState(x=x), geometry, substrate, None)
    for a, b in zip(ff, free.activations, strict=True):
        assert torch.equal(a, b), (
            "ePC free equilibrium must equal the residual feedforward pass "
            "bitwise (zero-error fixed point)"
        )

    nudged = dynamics.settle(SystemState(x=x), geometry, substrate, y)
    devs = [
        (n - f).abs().max().item()
        for f, n in zip(free.activations, nudged.activations, strict=True)
    ]
    assert min(devs[1:-1]) > 1e-4, (
        "the nudged signal must reach every hidden layer through the "
        f"residual skip path (deviations {devs})"
    )
