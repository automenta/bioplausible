"""Shared helpers for ontology property locks (L1–L7).

Internal module: all helpers are `_`-prefixed per AGENTS.md.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch
from torch import Tensor

from computronium.core.ontology import System, SystemState

# ----------------------------------------------------------------------
# Constants (do not inline shapes)
# ----------------------------------------------------------------------
WIDTH = 32
DEPTH = 4
BATCH = 64
SETTLE_ITERS = 50

# Tolerances
BITWISE = 0  # exact ==
TIGHT = dict(rtol=1e-5, atol=1e-6)
LOOSE = dict(rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------
# Device & determinism
# ----------------------------------------------------------------------
def select_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def seeded(seed: int) -> Generator[None]:
    """Set global torch/python seeds for the context."""
    cpu_state = torch.get_rng_state()
    cuda_available = torch.cuda.is_available()
    cuda_state = torch.cuda.get_rng_state() if cuda_available else None
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_available and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


def enable_deterministic_cuda() -> None:
    """Enable deterministic algorithms on CUDA (may skip some ops)."""
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def tiny_batch(seed: int) -> tuple[Tensor, Tensor]:
    """Synthetic batch: (x, y) with fixed shapes from constants."""
    with seeded(seed):
        x = torch.randn(BATCH, WIDTH, device=select_device())
        y = torch.randint(0, 10, (BATCH,), device=select_device())
    return x, y


# ----------------------------------------------------------------------
# Protocol conformance (TypeIs narrowers per Protocol)
# ----------------------------------------------------------------------
def _conforms(obj: object, methods: dict[str, object]) -> bool:
    """Runtime protocol check: obj has all methods with callable values."""
    return all(hasattr(obj, name) and callable(getattr(obj, name)) for name in methods)


def conforms(obj: object, methods: dict[str, object]) -> bool:
    """Public re-export of _conforms for test use."""
    return _conforms(obj, methods)


# ----------------------------------------------------------------------
# Settling helpers
# ----------------------------------------------------------------------
def settle_phases(
    system: System,
    x: Tensor,
    y: Tensor,
) -> tuple[SystemState, SystemState]:
    """Run free phase (target=None) then nudged phase (target=y).

    Mirrors compose_system ordering: free first, then nudged.
    """
    state = SystemState(x=x, y=y)

    # 1. Substrate + Geometry: Forward pass
    state.activations = system.geometry.forward(x, system.substrate)
    if state.activations is not None:
        state.activations = system.substrate.inject_state_noise(state.activations)

    # 2. Free phase
    free_state = system.dynamics.settle(
        state, system.geometry, system.substrate, target=None
    )
    free_state.energy = system.dynamics.compute_energy(free_state, system.geometry)

    # 3. Nudged phase
    nudged_state = system.dynamics.settle(
        state, system.geometry, system.substrate, target=y
    )
    nudged_state.energy = system.dynamics.compute_energy(nudged_state, system.geometry)
    nudged_state.loss = system._compute_loss(nudged_state, y)

    return free_state, nudged_state


def perturb_nonlocal(state: SystemState, layer: int, eps: float) -> SystemState:
    """Return a new state with entries outside layer `layer`'s pre/post support modified.

    For feedforward geometries: modifies activations at all layers except `layer`
    and `layer+1` (the pre/post of weight matrix `layer`).
    """
    if state.activations is None:
        return state

    acts = state.activations
    if not isinstance(acts, list):
        return state  # single tensor - no layer structure

    new_acts = []
    for i, act in enumerate(acts):
        if i == layer or i == layer + 1:
            new_acts.append(act)
        else:
            # Perturb non-local activations
            noise = torch.randn_like(act) * eps
            new_acts.append(act + noise)

    new_state = SystemState(
        x=state.x,
        y=state.y,
        activations=new_acts,
        free_state=state.free_state,
        nudged_state=state.nudged_state,
        pseudo_gradients=state.pseudo_gradients,
        energy=state.energy,
        loss=state.loss,
        metrics=state.metrics.copy() if state.metrics else {},
    )
    return new_state


# ----------------------------------------------------------------------
# Registry helpers for L6/L7
# ----------------------------------------------------------------------
def _all_registered_model_names() -> list[str]:
    """Get all registered model names from Registry."""
    from computronium.core.registry import ComponentCategory, Registry

    try:
        return list(Registry._components.get(ComponentCategory.MODEL, {}).keys())
    except Exception:
        return []


def _round_trip_configs(system: System) -> System:
    """Serialize system configs to JSON and reconstruct.

    For now, this is a placeholder that re-creates from configs.
    Full implementation would need JSON serialization of all 5 layer configs.
    """
    # Extract configs
    substrate_cfg = system.substrate.config
    geometry_cfg = system.geometry.config
    dynamics_cfg = system.dynamics.config
    credit_cfg = system.credit.config
    update_cfg = system.update.config

    # Reconstruct from configs
    from computronium.core.ontology import (
        BackpropCredit,
        DigitalSubstrate,
        ElasticConsolidationUpdate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        InstantaneousDynamics,
        LocalGoodnessCredit,
        NaturalGradientUpdate,
        RandomProjectionsCredit,
        RecurrentGeometry,
        RiemannianOrthogonalUpdate,
        SpectralConstrainedUpdate,
        TargetInversionCredit,
        TemporalTraceCredit,
        ThermodynamicContrast,
    )

    # Map substrate type to class
    substrate_map = {
        "digital": DigitalSubstrate,
        "analog": lambda c: DigitalSubstrate(c),
        "memristor": lambda c: DigitalSubstrate(c),
        "optical": lambda c: DigitalSubstrate(c),
        "neuromorphic": lambda c: DigitalSubstrate(c),
        "quantum": lambda c: DigitalSubstrate(c),
    }
    substrate_cls = substrate_map.get(substrate_cfg.device, DigitalSubstrate)
    substrate = substrate_cls(substrate_cfg)

    # Map geometry type to class
    if geometry_cfg.topology_type == "recurrent":
        geometry = RecurrentGeometry(geometry_cfg)
    elif geometry_cfg.topology_type == "tile_mesh":
        from computronium.core.ontology import TileGeometry

        geometry = TileGeometry(geometry_cfg)
    else:
        geometry = FeedforwardGeometry(geometry_cfg)

    # Map dynamics type to class
    dynamics_map = {
        "instantaneous": InstantaneousDynamics,
        "energy_minimization": EnergyMinimizationDynamics,
        "predictive_settling": EnergyMinimizationDynamics,
        "spike_integration": EnergyMinimizationDynamics,
    }
    dynamics_cls = dynamics_map.get(dynamics_cfg.dynamics_type, InstantaneousDynamics)
    dynamics = dynamics_cls(dynamics_cfg)

    # Map credit type to class
    credit_map = {
        "thermodynamic_contrast": ThermodynamicContrast,
        "random_projections": RandomProjectionsCredit,
        "local_goodness": LocalGoodnessCredit,
        "temporal_trace": TemporalTraceCredit,
        "target_inversion": TargetInversionCredit,
        "gradient": BackpropCredit,
    }
    credit_cls = credit_map.get(credit_cfg.credit_type, BackpropCredit)
    credit = credit_cls(credit_cfg)

    # Map update type to class
    update_map = {
        "euclidean": EuclideanUpdate,
        "riemannian_orthogonal": RiemannianOrthogonalUpdate,
        "spectral_constrained": SpectralConstrainedUpdate,
        "natural_gradient": NaturalGradientUpdate,
        "elastic_consolidation": ElasticConsolidationUpdate,
    }
    update_cls = update_map.get(update_cfg.update_type, EuclideanUpdate)
    update = update_cls(update_cfg)

    from computronium.core.system_trainer import compose_system

    return compose_system(substrate, geometry, dynamics, credit, update)
