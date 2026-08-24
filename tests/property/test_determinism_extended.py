"""Determinism Lock (L5) for All 6-D Coordinates.

Verifies that same seed + same device = bitwise equal params & metrics
for any valid 6-D coordinate combination.
"""

import pytest
import torch

from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
from computronium.core.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    RandomProjectionsCredit,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
)
from computronium.core.plasticity.fast_weights import create_fast_weight_plasticity
from computronium.core.plasticity.routing import create_routing_plasticity
from computronium.core.system_trainer import compose_joint_system
from tests.property._support import (
    BITWISE,
    DEPTH,
    WIDTH,
    enable_deterministic_cuda,
    seeded,
    select_device,
    tiny_batch,
)

STEPS = 5


# ----------------------------------------------------------------------
# System Factories for 6-D Coordinates
# ----------------------------------------------------------------------


def _make_6d_system(coordinate: dict, device: str = "cpu") -> object:
    """Create a JointSystem from a 6-D coordinate dict."""
    substrate_type = coordinate.get("substrate", "digital")
    geometry_type = coordinate.get("geometry", "feedforward")
    dynamics_type = coordinate.get("dynamics", "instantaneous")
    plasticity_type = coordinate.get("plasticity", "null")
    credit_type = coordinate.get("credit", "backprop")
    update_type = coordinate.get("update", "euclidean")

    # Substrate
    if substrate_type == "digital":
        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
    else:
        raise ValueError(f"Unknown substrate: {substrate_type}")

    # Geometry
    if geometry_type == "feedforward":
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=WIDTH,
                output_dim=10,
                hidden_dims=(WIDTH,) * (DEPTH - 1),
                init_scale=0.1,
            )
        )
    elif geometry_type == "recurrent":
        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=WIDTH,
                output_dim=10,
                hidden_dims=(WIDTH,) * (DEPTH - 1),
                init_scale=0.1,
            ),
            hidden_dim=WIDTH,
        )
    else:
        raise ValueError(f"Unknown geometry: {geometry_type}")

    # Dynamics
    if dynamics_type == "energy_minimization":
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=10, beta=0.5, step_size=0.1
            )
        )
    elif dynamics_type == "instantaneous":
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    elif dynamics_type == "predictive_settling":
        from computronium.core.ontology import PredictiveSettlingDynamics

        dynamics = PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=10, step_size=0.1)
        )
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")

    # Plasticity
    if plasticity_type == "null":
        plasticity = NullPlasticity()
    elif plasticity_type == "routing":
        plasticity = create_routing_plasticity(PlasticityConfig.routing(gate_dim=64))
    elif plasticity_type == "fast_weights":
        plasticity = create_fast_weight_plasticity(
            PlasticityConfig.fast_weights(fast_weight_dim=512)
        )
    else:
        raise ValueError(f"Unknown plasticity: {plasticity_type}")

    # Credit
    if credit_type == "backprop":
        credit = BackpropCredit(CreditAssignmentConfig.gradient())
    elif credit_type == "thermodynamic_contrast":
        credit = ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        )
    elif credit_type == "random_projections":
        credit = RandomProjectionsCredit(CreditAssignmentConfig.random_projections())
    else:
        raise ValueError(f"Unknown credit: {credit_type}")

    # Update
    if update_type == "euclidean":
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))
    else:
        raise ValueError(f"Unknown update: {update_type}")

    return compose_joint_system(
        substrate, geometry, dynamics, plasticity, credit, update
    )


def _run_train_step(system, x, y):
    """Run a single train_step and return metrics and params."""
    metrics = system.train_step(x, y)
    params = {k: v.clone() for k, v in system.geometry.params.items()}
    return metrics, params


# ----------------------------------------------------------------------
# Valid 6-D Coordinates for Testing
# ----------------------------------------------------------------------


VALID_6D_COORDINATES = [
    # Standard combinations (null plasticity = 5-D equivalence)
    {
        "substrate": "digital",
        "geometry": "feedforward",
        "dynamics": "instantaneous",
        "plasticity": "null",
        "credit": "backprop",
        "update": "euclidean",
    },
    {
        "substrate": "digital",
        "geometry": "feedforward",
        "dynamics": "instantaneous",
        "plasticity": "null",
        "credit": "thermodynamic_contrast",
        "update": "euclidean",
    },
    {
        "substrate": "digital",
        "geometry": "feedforward",
        "dynamics": "instantaneous",
        "plasticity": "null",
        "credit": "random_projections",
        "update": "euclidean",
    },
    {
        "substrate": "digital",
        "geometry": "recurrent",
        "dynamics": "energy_minimization",
        "plasticity": "null",
        "credit": "thermodynamic_contrast",
        "update": "euclidean",
    },
    # With routing plasticity
    {
        "substrate": "digital",
        "geometry": "recurrent",
        "dynamics": "energy_minimization",
        "plasticity": "routing",
        "credit": "thermodynamic_contrast",
        "update": "euclidean",
    },
    {
        "substrate": "digital",
        "geometry": "feedforward",
        "dynamics": "instantaneous",
        "plasticity": "routing",
        "credit": "backprop",
        "update": "euclidean",
    },
    # With fast weights plasticity
    {
        "substrate": "digital",
        "geometry": "recurrent",
        "dynamics": "energy_minimization",
        "plasticity": "fast_weights",
        "credit": "thermodynamic_contrast",
        "update": "euclidean",
    },
    {
        "substrate": "digital",
        "geometry": "feedforward",
        "dynamics": "instantaneous",
        "plasticity": "fast_weights",
        "credit": "backprop",
        "update": "euclidean",
    },
    # With predictive settling dynamics
    {
        "substrate": "digital",
        "geometry": "feedforward",
        "dynamics": "predictive_settling",
        "plasticity": "null",
        "credit": "thermodynamic_contrast",
        "update": "euclidean",
    },
    {
        "substrate": "digital",
        "geometry": "recurrent",
        "dynamics": "predictive_settling",
        "plasticity": "null",
        "credit": "thermodynamic_contrast",
        "update": "euclidean",
    },
]


def _coordinate_id(coord: dict) -> str:
    """Generate a readable test ID from a coordinate."""
    return "/".join([
        coord["substrate"],
        coord["geometry"],
        coord["dynamics"],
        coord["plasticity"],
        coord["credit"],
        coord["update"],
    ])


# ----------------------------------------------------------------------
# Determinism Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("coordinate", VALID_6D_COORDINATES, ids=_coordinate_id)
def test_l5_determinism_lock_6d(coordinate):
    """Same seed, same device, two runs of train_step: metrics and params bitwise equal for 6-D coordinates."""
    device = select_device()

    # CPU always
    with seeded(42):
        sys1 = _make_6d_system(coordinate, device=device)
        if hasattr(sys1.geometry, "to"):
            sys1.geometry.to(device)
        if hasattr(sys1.substrate, "to"):
            sys1.substrate.to(device)
        x, y = tiny_batch(42)
        metrics1, params1 = _run_train_step(sys1, x, y)

    with seeded(42):
        sys2 = _make_6d_system(coordinate, device=device)
        if hasattr(sys2.geometry, "to"):
            sys2.geometry.to(device)
        if hasattr(sys2.substrate, "to"):
            sys2.substrate.to(device)
        x, y = tiny_batch(42)
        metrics2, params2 = _run_train_step(sys2, x, y)

    assert _metrics_equal(metrics1, metrics2, BITWISE), (
        f"Metrics differ for {_coordinate_id(coordinate)}"
    )
    assert _params_equal(params1, params2, BITWISE), (
        f"Params differ for {_coordinate_id(coordinate)}"
    )

    # GPU with deterministic settings (may skip if op lacks deterministic impl)
    if torch.cuda.is_available():

        def _run_gpu():
            enable_deterministic_cuda()
            with seeded(42):
                sys3 = _make_6d_system(coordinate, device="cuda")
                if hasattr(sys3.geometry, "to"):
                    sys3.geometry.to("cuda")
                if hasattr(sys3.substrate, "to"):
                    sys3.substrate.to("cuda")
                x, y = tiny_batch(42)
                metrics3, params3 = _run_train_step(sys3, x, y)
            return metrics3, params3

        try:
            metrics3, params3 = _run_gpu()
            assert _metrics_equal(metrics1, metrics3, BITWISE), (
                f"GPU metrics differ for {_coordinate_id(coordinate)}"
            )
            assert _params_equal(params1, params3, BITWISE), (
                f"GPU params differ for {_coordinate_id(coordinate)}"
            )
        except RuntimeError as e:
            if "deterministic" in str(e).lower():
                pytest.skip(f"GPU deterministic op not available: {e}")
            raise


def _params_equal(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], tol: dict | int = BITWISE
) -> bool:
    """Compare two parameter dicts."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if isinstance(tol, int) and tol == BITWISE:
            if not torch.equal(a[k], b[k]):
                return False
        elif not torch.allclose(a[k], b[k], **tol):
            return False
    return True


def _metrics_equal(
    a: dict[str, float], b: dict[str, float], tol: dict | int = BITWISE
) -> bool:
    """Compare two metrics dicts."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if isinstance(tol, int) and tol == BITWISE:
            if a[k] != b[k]:
                return False
        elif not (
            abs(a[k] - b[k])
            <= tol.get("atol", 1e-5) + tol.get("rtol", 1e-5) * abs(b[k])
        ):
            return False
    return True


# ----------------------------------------------------------------------
# Multi-step Determinism Test
# ----------------------------------------------------------------------


@pytest.mark.parametrize("coordinate", VALID_6D_COORDINATES[:5], ids=_coordinate_id)
def test_l5_determinism_multi_step_6d(coordinate):
    """Test determinism over multiple training steps for 6-D coordinates."""
    device = select_device()

    def _run_multi_step(seed: int):
        with seeded(seed):
            system = _make_6d_system(coordinate, device=device)
            if hasattr(system.geometry, "to"):
                system.geometry.to(device)
            if hasattr(system.substrate, "to"):
                system.substrate.to(device)

            all_metrics = []
            for i in range(STEPS):
                x, y = tiny_batch(seed + i)
                metrics, _ = _run_train_step(system, x, y)
                all_metrics.append(metrics)
            final_params = {k: v.clone() for k, v in system.geometry.params.items()}
            return all_metrics, final_params

    metrics1, params1 = _run_multi_step(42)
    metrics2, params2 = _run_multi_step(42)

    # Check each step's metrics
    for i, (m1, m2) in enumerate(zip(metrics1, metrics2)):
        assert _metrics_equal(m1, m2, BITWISE), (
            f"Step {i} metrics differ for {_coordinate_id(coordinate)}"
        )

    assert _params_equal(params1, params2, BITWISE), (
        f"Final params differ for {_coordinate_id(coordinate)}"
    )

    # GPU test
    if torch.cuda.is_available():

        def _run_gpu_multi():
            enable_deterministic_cuda()
            return _run_multi_step(42)

        try:
            metrics3, params3 = _run_gpu_multi()
            for i, (m1, m3) in enumerate(zip(metrics1, metrics3)):
                assert _metrics_equal(m1, m3, BITWISE), (
                    f"GPU step {i} metrics differ for {_coordinate_id(coordinate)}"
                )
            assert _params_equal(params1, params3, BITWISE), (
                f"GPU final params differ for {_coordinate_id(coordinate)}"
            )
        except RuntimeError as e:
            if "deterministic" in str(e).lower():
                pytest.skip(f"GPU deterministic op not available: {e}")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
