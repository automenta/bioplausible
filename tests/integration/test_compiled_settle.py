"""Compiled layered-settle lock (CP-6 enablement).

``StateDynamicsConfig.compiled=True`` runs ``PredictiveSettlingDynamics``'s
layered settle as one ``torch.compile`` graph. Locks:

1. parity — compiled and eager settles agree on the same inputs (probe
   measured bitwise equality at depth 8 / 60 steps; the lock uses a tiny
   geometry so the suite stays cheap);
2. guard rails — compiled config falls back to the eager path without
   error on recurrent geometries and per-iteration energy tracking.
"""

from typing import TYPE_CHECKING, cast

import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    compose_system,
)
from computronium.ontology import (
    EnergyMinimizationDynamics,
    FeedforwardGeometry,
    PredictiveSettlingDynamics,
)

if TYPE_CHECKING:
    from computronium.state import CompositeState


def _config(compiled: bool) -> StateDynamicsConfig:
    return StateDynamicsConfig.predictive_settling(
        max_steps=5, step_size=0.1, compiled=compiled
    )


def _build(hidden_dims, credit, compiled: bool):
    torch.manual_seed(0)  # seed BEFORE geometry construction (weights draw from RNG)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=hidden_dims
        )
    )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=PredictiveSettlingDynamics(_config(compiled)),
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )


def test_compiled_settle_matches_eager() -> None:
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )
    x = torch.randn(16, 784)
    acts = []
    for compiled in (False, True):
        system = _build((32, 32), credit, compiled)
        settled = system.dynamics.settle(
            cast("CompositeState", SystemState(x=x)), system.geometry, system.substrate
        )
        assert settled.activations is not None
        acts.append(settled.activations)
    max_dev = max(
        (a - b).abs().max().item() for a, b in zip(acts[0], acts[1], strict=True)
    )
    assert max_dev < 1e-6, f"compiled settle diverged from eager: {max_dev}"


def test_compiled_config_falls_back_cleanly() -> None:
    torch.manual_seed(0)
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(input_dim=784, output_dim=10, hidden_dims=(32,))
    )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=PredictiveSettlingDynamics(_config(compiled=True)),
        credit=BackpropCredit(),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )
    x = torch.randn(16, 784)
    settled = system.dynamics.settle(
        cast("CompositeState", SystemState(x=x)), system.geometry, system.substrate
    )
    assert settled.activations is not None
    assert all(torch.isfinite(t).all() for t in settled.activations)


def test_compiled_flag_in_config_round_trip() -> None:
    config = _config(compiled=True)
    assert config.compiled is True
    assert config.dynamics_type == "predictive_settling"


def _eqprop_system(compiled: bool):
    torch.manual_seed(0)  # seed BEFORE geometry construction
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(32, 32)
            )
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=5, step_size=0.5, beta=0.5, compiled=compiled
            )
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.05)),
    )


def test_compiled_eqprop_settle_matches_eager() -> None:
    x = torch.randn(16, 784)
    acts = []
    for compiled in (False, True):
        system = _eqprop_system(compiled)
        settled = system.dynamics.settle(
            cast("CompositeState", SystemState(x=x)),
            system.geometry,
            system.substrate,
        )
        assert settled.activations is not None
        acts.append(settled.activations)
    max_dev = max(
        (a - b).abs().max().item() for a, b in zip(acts[0], acts[1], strict=True)
    )
    assert max_dev < 1e-5, f"compiled EqProp settle diverged: {max_dev}"


def test_compiled_eqprop_settle_builds_autograd_graph() -> None:
    """Thermo credit differentiates through the settle — compiled path must too."""
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))
    grad_norms = []
    for compiled in (False, True):
        system = _eqprop_system(compiled)
        with torch.enable_grad():
            settled = system.dynamics.settle(
                cast("CompositeState", SystemState(x=x)),
                system.geometry,
                system.substrate,
            )
            assert settled.activations is not None
            loss = torch.nn.functional.cross_entropy(settled.activations[-1], y)
            grads = torch.autograd.grad(
                loss,
                [p for p in system.geometry.params.values() if p.requires_grad],
                allow_unused=True,
            )
        norms = [g.norm().item() if g is not None else 0.0 for g in grads]
        assert any(n > 0 for n in norms), "compiled settle broke the autograd graph"
        grad_norms.append(norms)
    for eager_n, compiled_n in zip(*grad_norms, strict=True):
        assert abs(eager_n - compiled_n) <= 1e-3 * (1.0 + eager_n)
