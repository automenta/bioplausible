"""Latency proxy lock (R11.2.14).

`estimate_train_step_flops` is the deterministic comparator for the
task-scale latency claim: structure-derived FLOPs per train_step, no
measurement. Locks:

1. determinism — same system, same number, twice;
2. ordering — proxy orders systems the same way measured walltime does
   (deep vs shallow settle, settle-step scaling), the claim the research
   track needs.
"""

import time

import pytest
import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.profiling import estimate_train_step_flops


def _system(hidden_dims: tuple[int, ...], max_steps: int):
    torch.manual_seed(0)
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=64, output_dim=10, hidden_dims=hidden_dims
            )
        ),
        dynamics=PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=max_steps, step_size=0.1)
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )


def test_proxy_is_deterministic() -> None:
    system = _system((32, 32), max_steps=5)
    a = estimate_train_step_flops(system, batch_size=64)
    b = estimate_train_step_flops(system, batch_size=64)
    assert a == b > 0


def test_proxy_scales_with_depth_and_settle_steps() -> None:
    shallow = estimate_train_step_flops(_system((32,), max_steps=5), 64)
    deep = estimate_train_step_flops(_system((32,) * 4, max_steps=5), 64)
    assert deep > shallow
    few = estimate_train_step_flops(_system((32, 32), max_steps=2), 64)
    many = estimate_train_step_flops(_system((32, 32), max_steps=10), 64)
    assert many == few * 5


def test_proxy_ordering_matches_measured_walltime() -> None:
    systems = (
        ("shallow", _system((32,), max_steps=2)),
        ("deep", _system((64, 64, 64), max_steps=8)),
    )
    proxy = [estimate_train_step_flops(s, batch_size=32) for _, s in systems]
    assert proxy[1] > proxy[0]

    x = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    walltime = []
    for _, system in systems:
        for _ in range(3):  # warmup
            system.train_step(x, y)
        t0 = time.perf_counter()
        for _ in range(10):
            system.train_step(x, y)
        walltime.append(time.perf_counter() - t0)
    assert walltime[1] > walltime[0], (
        f"proxy ordering {proxy} contradicts measured walltime {walltime}"
    )


def test_proxy_rejects_non_layered_geometry() -> None:
    from computronium.ontology import GraphGeometry

    torch.manual_seed(0)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=GraphGeometry(
            GeometryConfig.graph(input_dim=8, output_dim=2, edge_index=[[1], [0]])
        ),
        dynamics=PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=2)
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )
    with pytest.raises(ValueError, match="layered"):
        estimate_train_step_flops(system, batch_size=4)
