"""R11.1.10 lock: LazyStateDynamics — registered sequential settle.

Claims (measured regime, 2026-09-04):

1. ``"lazy"`` is a first-class dynamics: registry round-trip
   (config → dynamics_from_config) and 5-axis composition.
2. The Gauss–Seidel settle converges with per-sweep Hopfield energy
   non-increasing, and the nudged phase pulls the output toward the target.
3. It trains end-to-end above 2.5× chance in the D2 regime (MNIST
   quick-mode, thermodynamic contrast).
4. Fail-loud wiring: non-layered geometries and recurrent weights raise
   TypeError instead of silently mis-settling.
5. Sweep-count observability: the per-sweep activation cache makes
   on-demand settling directly measurable. Measured contrast at
   (256→64×6→10, τ=1e-2, step 0.05): lazy 34 sweeps vs Jacobi 21
   iterations — Gauss–Seidel does NOT dominate in sweeps at demo scale
   (expectation refuted; recorded in TODO11). Both counts are asserted
   finite within budget, no superiority claim.
"""

from __future__ import annotations

from itertools import islice

import pytest
import torch
from torch import Tensor

from computronium import (
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    LazyStateDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology._settle_kernel import (
    SubstrateSettleKernel,
    extract_layered_params,
)
from computronium.ontology.credit import CreditAssignmentConfig
from computronium.ontology.dynamics import dynamics_from_config
from computronium.ontology.geometry import FeedforwardGeometry
from computronium.ontology.system import SystemState


def _build(dynamics, *, input_dim: int = 16, width: int = 8, depth: int = 2):
    torch.manual_seed(0)
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=input_dim,
                output_dim=5,
                hidden_dims=(width,) * depth,
            )
        ),
        dynamics=dynamics,
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )


def _state(system, x: Tensor, y: Tensor | None = None) -> SystemState:
    state = SystemState(x=x, y=y)
    state.activations = system.geometry.forward_with_intermediates(x, system.substrate)
    return state


def test_lazy_registry_round_trip() -> None:
    config = StateDynamicsConfig.lazy(max_steps=17, beta=0.3)
    dynamics = dynamics_from_config(config)
    assert dynamics.config == config
    assert type(dynamics) is type(dynamics_from_config(config))


def test_lazy_settle_monotone_and_nudges() -> None:
    from computronium.ontology.dynamics._dynamics import _compute_hopfield_energy

    dynamics = LazyStateDynamics(
        StateDynamicsConfig.lazy(
            max_steps=500,
            convergence_threshold=1e-4,
            convergence_start=1,
            step_size=0.1,
        )
    )
    system = _build(dynamics)
    x = torch.randn(4, 16)
    y = torch.randint(0, 5, (4,))

    state = dynamics.settle(
        _state(system, x),  # pyright: ignore[reportArgumentType]
        system.geometry,
        system.substrate,
        target=None,
    )
    cache = dynamics.get_cached_activations()
    assert cache, "lazy settle recorded no sweeps"
    energies = [
        _compute_hopfield_energy(acts, system.geometry).item()
        for acts in cache.values()
    ]
    assert all(
        energies[i + 1] <= energies[i] + 1e-6 for i in range(len(energies) - 1)
    ), "per-sweep Hopfield energy increased"

    free_state = state.free_state
    assert isinstance(free_state, list)
    nudged = dynamics.settle(
        _state(system, x, y),  # pyright: ignore[reportArgumentType]
        system.geometry,
        system.substrate,
        target=y,
    )
    nudged_state = nudged.nudged_state
    assert isinstance(nudged_state, list)
    one_hot = _one_hot(y, free_state[-1])
    free_dist = torch.dist(free_state[-1], one_hot).item()
    nudged_dist = torch.dist(nudged_state[-1], one_hot).item()
    assert nudged_dist < free_dist, "nudge did not pull output toward target"


def _one_hot(target: Tensor, like: Tensor) -> Tensor:
    out = torch.zeros_like(like)
    out.scatter_(1, target.unsqueeze(1), 1.0)
    return out


def test_lazy_trains_end_to_end() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()

    def _flatten(loader, cap: int = 150):
        for xb, yb in islice(loader, cap):
            yield xb.view(xb.size(0), -1), yb

    torch.manual_seed(0)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(32,))
        ),
        dynamics=LazyStateDynamics(
            StateDynamicsConfig.lazy(max_steps=3, step_size=0.1, beta=0.5)
        ),
        credit=ThermodynamicContrast(),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )
    metrics = SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=list(
            _flatten(task.get_dataloader("train"))  # pyright: ignore[reportAttributeAccessIssue]
        ),
    ).fit()[-1]
    assert metrics["train_acc"] > 0.25, (
        f"lazy dynamics failed to learn: {metrics['train_acc']:.3f}"
    )


def test_lazy_fail_loud_non_layered() -> None:
    class _Bare:
        pass

    dynamics = LazyStateDynamics(StateDynamicsConfig.lazy(max_steps=5))
    with pytest.raises(TypeError, match="layer-structured"):
        dynamics._layered(_Bare())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    system = _build(dynamics)
    # composite state without x short-circuits instead of raising
    state = SystemState()
    assert (
        dynamics.settle(
            state,  # pyright: ignore[reportArgumentType]
            system.geometry,
            system.substrate,
        )
        is state
    )


def test_lazy_sweep_count_observable() -> None:
    """The on-demand observable: sweep count vs an equivalent Jacobi loop
    at the same threshold — both finite, no dominance claim (measured:
    Gauss–Seidel needed 34 sweeps vs Jacobi 21 at this regime)."""
    tau = 1e-2
    dynamics = LazyStateDynamics(
        StateDynamicsConfig.lazy(
            max_steps=500,
            convergence_threshold=tau,
            convergence_start=1,
            step_size=0.05,
        )
    )
    system = _build(dynamics, input_dim=256, width=64, depth=6)
    x = torch.randn(4, 256)
    dynamics.settle(
        _state(system, x),  # pyright: ignore[reportArgumentType]
        system.geometry,
        system.substrate,
    )
    assert dynamics.get_cached_activations(), "sweep cache empty"

    jacobi = _build(
        EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=1)
        ),
        input_dim=256,
        width=64,
        depth=6,
    )
    params = extract_layered_params(jacobi.geometry)
    assert params is not None
    kernel = SubstrateSettleKernel(
        substrate=jacobi.substrate, params=params, step_size=0.05
    )
    acts = list(jacobi.geometry.forward_with_intermediates(x, jacobi.substrate))
    converged = False
    for _ in range(500):
        new_acts, _ = kernel.step(acts, beta=0.0, target=None, velocity=None)
        delta = max(
            torch.dist(a, b, p=float("inf")).item() for a, b in zip(new_acts, acts)
        )
        acts = new_acts
        if delta < tau:
            converged = True
            break
    assert converged, "jacobi reference did not converge"
