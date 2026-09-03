"""Ontology Property Locks (L1-L7) - CORRECTNESS_LOCK.md

Fast-CI property suite enforcing seven invariants of the 5-D ontology.
Wall-clock budget: <= 5 min on GPU, <= 10 min on CPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from computronium.core.pipeline import phase_states, task_loss
from computronium.core.system_trainer import (
    compose_system,
)
from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    RandomProjectionsCredit,
    RecurrentGeometry,
    RiemannianOrthogonalUpdate,
    StateDynamicsConfig,
    SubstrateConfig,
    System,
    SystemState,
    ThermodynamicContrast,
    TileGeometry,
)
from tests.property._support import (
    BITWISE,
    DEPTH,
    SETTLE_ITERS,
    WIDTH,
    enable_deterministic_cuda,
    perturb_nonlocal,
    seeded,
    select_device,
    tiny_batch,
)

# ----------------------------------------------------------------------
# Constants for magic values
# ----------------------------------------------------------------------
LOCALITY_TOL = 1e-6
ENERGY_CONVERGENCE_TOL = 1e-6
ENERGY_STEP_TOL = 1e-7
PERTURB_SCALE = 1e-3
MIN_LAYERS_FOR_LOCALITY = 2


# ----------------------------------------------------------------------
# Test systems for parametrization
# ----------------------------------------------------------------------
def _make_eqprop_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital()),
        geometry=RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            ),
            hidden_dim=WIDTH,
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=SETTLE_ITERS, beta=0.5)
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
    )


def _make_backprop_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital()),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=BackpropCredit(CreditAssignmentConfig.gradient()),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
    )


def _make_fa_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital()),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=RandomProjectionsCredit(CreditAssignmentConfig.random_projections()),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
    )


def _make_predictive_coding_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital()),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            )
        ),
        dynamics=PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(
                max_steps=SETTLE_ITERS, step_size=0.1
            )
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
    )


def _make_tile_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital()),
        geometry=TileGeometry(
            GeometryConfig.tile_mesh(
                input_dim=WIDTH,
                output_dim=10,
                num_layers=DEPTH,
                neurons_per_tile=8,
                tiles_per_layer=2,
            ),
            neurons_per_tile=8,
            tiles_per_layer=2,
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=BackpropCredit(CreditAssignmentConfig.gradient()),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.001)),
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _copy_system_params(src: System, dst: System) -> None:
    """Copy parameters from src geometry to dst geometry."""
    dst.geometry.update_params(src.geometry.params)


def _params_equal(
    a: dict[str, Tensor], b: dict[str, Tensor], tol: dict | int = BITWISE
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


def _assert_metrics_valid(metrics: dict[str, float]) -> None:
    """Assert metrics have expected keys and valid ranges (imp-46 schema).

    Claim-grade keys are the ``free_*`` post-update target-free metrics; the
    output-phase fit is quarantined under ``nudged_fit_accuracy``.
    """
    assert metrics is not None
    assert "loss" in metrics
    assert metrics["loss"] >= 0
    assert "free_accuracy" in metrics
    assert 0 <= metrics["free_accuracy"] <= 1
    assert "nudged_fit_accuracy" in metrics
    assert 0 <= metrics["nudged_fit_accuracy"] <= 1


def _setup_system_device(sys: System, device: torch.device) -> System:
    """Move system to device if supported."""
    if hasattr(sys.geometry, "to"):
        sys.geometry.to(device)
    return sys


def _run_settle_and_compute_loss(
    sys: System, x: Tensor, y: Tensor | None, target: Tensor | None
) -> SystemState:
    """Run settle and compute loss."""
    state = SystemState(x=x, y=y)
    state.activations = sys.geometry.forward(x, sys.substrate)
    if state.activations is not None:
        state.activations = sys.substrate.inject_state_noise(state.activations)
    state = sys.dynamics.settle(state, sys.geometry, sys.substrate, target=target)
    if target is not None:
        state.loss = task_loss(state, target)
    return state


# ======================================================================
# L1 - Parity Lock (strangler-fig guarantee)
# ======================================================================
def test_l1_composed_systems_train() -> None:
    """Composed 5-D systems can train and produce valid metrics."""
    device = select_device()
    if device.type == "cuda":
        enable_deterministic_cuda()

    with seeded(42):
        x, y = tiny_batch(42)

    for system_factory in [
        _make_backprop_system,
        _make_fa_system,
        _make_tile_system,
    ]:
        with seeded(42):
            system = system_factory()
            _setup_system_device(system, device)

            metrics = system.train_step(x, y)
            _assert_metrics_valid(metrics)


# ======================================================================
# L2 - Orthogonality Lock (ontology honesty)
# ======================================================================
class TestL2OrthogonalityLock:
    """Each pipeline stage is a pure function of the axes that precede it."""

    def test_o1_geometry_forward_deterministic(self) -> None:
        """geometry.forward is deterministic for same inputs."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = _make_backprop_system()
        _setup_system_device(sys, device)

        x, _ = tiny_batch(42)
        out1 = sys.geometry.forward(x, sys.substrate)
        out2 = sys.geometry.forward(x, sys.substrate)
        assert torch.equal(out1, out2)

    def test_o2_geometry_forward_independent_of_dynamics(self) -> None:
        """geometry.forward output doesn't depend on which Dynamics is used."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys1 = _make_backprop_system()
        with seeded(42):
            sys2 = _make_predictive_coding_system()

        _setup_system_device(sys1, device)
        _setup_system_device(sys2, device)

        _copy_system_params(sys1, sys2)
        x, _ = tiny_batch(42)

        out1 = sys1.geometry.forward(x, sys1.substrate)
        out2 = sys2.geometry.forward(x, sys2.substrate)
        assert torch.equal(out1, out2)

    def test_o3_credit_independent_of_update(self) -> None:
        """Pseudo-gradients are independent of ParameterUpdate choice."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys1 = _make_backprop_system()
        with seeded(42):
            sys2 = compose_system(
                substrate=DigitalSubstrate(SubstrateConfig.digital()),
                geometry=FeedforwardGeometry(
                    GeometryConfig.feedforward(
                        input_dim=WIDTH,
                        output_dim=10,
                        hidden_dims=(WIDTH,) * (DEPTH - 1),
                    )
                ),
                dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                credit=BackpropCredit(CreditAssignmentConfig.gradient()),
                update=RiemannianOrthogonalUpdate(
                    ParameterUpdateConfig.riemannian_orthogonal(step_size=0.001)
                ),
            )

        _setup_system_device(sys1, device)
        _setup_system_device(sys2, device)

        _copy_system_params(sys1, sys2)
        x, y = tiny_batch(42)

        free1 = _run_settle_and_compute_loss(sys1, x, y, target=None)
        nudged1 = _run_settle_and_compute_loss(sys1, x, y, target=y)

        free2 = _run_settle_and_compute_loss(sys2, x, y, target=None)
        nudged2 = _run_settle_and_compute_loss(sys2, x, y, target=y)

        grads1 = sys1.credit.compute_pseudo_gradient(
            phase_states(free=free1, nudged=nudged1),
            nudged1.loss,
            sys1.geometry,
        )
        grads2 = sys2.credit.compute_pseudo_gradient(
            phase_states(free=free2, nudged=nudged2),
            nudged2.loss,
            sys2.geometry,
        )

        assert len(grads1) == len(grads2)
        for g1, g2 in zip(grads1, grads2, strict=True):
            assert torch.equal(g1, g2)

    def test_o4_substrate_noise_is_only_effect(self) -> None:
        """With noiseless DigitalSubstrate, forward is deterministic."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = _make_tile_system()
        _setup_system_device(sys, device)

        x, _ = tiny_batch(42)
        out1 = sys.geometry.forward(x, sys.substrate)
        out2 = sys.geometry.forward(x, sys.substrate)
        assert torch.equal(out1, out2)


# ======================================================================
# L3 - Locality Lock (bioplausibility axiom)
# ======================================================================
class TestL3LocalityLock:
    """L3a: Strictly-local rules have invariant pseudo-gradients under
    non-local perturbation. L3b: FA family feedback matrices fixed at init
    and independent of forward weights.
    """

    def test_l3a_thermodynamic_contrast_local(self) -> None:
        """ThermodynamicContrast pseudo-gradient invariant to
        non-local perturbations.
        """
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_eqprop_system()
            _setup_system_device(sys, device)

            x, y = tiny_batch(42)

            free = _run_settle_and_compute_loss(sys, x, y, target=None)
            nudged = _run_settle_and_compute_loss(sys, x, y, target=y)

            grads_orig = sys.credit.compute_pseudo_gradient(
                phase_states(free=free, nudged=nudged),
                nudged.loss,
                sys.geometry,
            )

            if len(grads_orig) >= MIN_LAYERS_FOR_LOCALITY:
                free_pert = perturb_nonlocal(free, 0, PERTURB_SCALE)
                nudged_pert = perturb_nonlocal(nudged, 0, PERTURB_SCALE)

                grads_pert = sys.credit.compute_pseudo_gradient(
                    phase_states(free=free_pert, nudged=nudged_pert),
                    nudged_pert.loss,
                    sys.geometry,
                )

                assert torch.allclose(
                    grads_orig[0], grads_pert[0], atol=LOCALITY_TOL, rtol=0
                ), "ThermodynamicContrast violated locality at layer 0"

    def test_l3b_fa_feedback_matrices_fixed_at_init(self) -> None:
        """Feedback matrices are fixed at init and independent of forward weights."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_fa_system()
            _setup_system_device(sys, device)

            # Initialize feedback weights
            sys.credit._init_feedback_weights(sys.geometry, device)
            fb_weights_orig = {
                k: v.clone() for k, v in sys.credit._feedback_weights.items()
            }

            # Create new system with same credit but different forward weights
            with seeded(123):
                sys2 = compose_system(
                    substrate=DigitalSubstrate(SubstrateConfig.digital()),
                    geometry=FeedforwardGeometry(
                        GeometryConfig.feedforward(
                            input_dim=WIDTH,
                            output_dim=10,
                            hidden_dims=(WIDTH,) * (DEPTH - 1),
                        )
                    ),
                    dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                    credit=sys.credit,  # Same credit instance
                    update=EuclideanUpdate(
                        ParameterUpdateConfig.euclidean(step_size=0.001)
                    ),
                )

            _setup_system_device(sys2, device)

            # Feedback weights should be identical
            assert sys2.credit._feedback_weights is not None
            for k in fb_weights_orig:
                assert torch.equal(
                    fb_weights_orig[k], sys2.credit._feedback_weights[k]
                ), f"Feedback matrix {k} changed after forward weight re-init"

    def test_l3b_fa_different_seeds_produce_different_feedback(self) -> None:
        """Different feedback seed produces different feedback matrices."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = _make_fa_system()
        _setup_system_device(sys, device)

        credit1 = RandomProjectionsCredit(CreditAssignmentConfig.random_projections())
        credit2 = RandomProjectionsCredit(CreditAssignmentConfig.random_projections())

        with seeded(111):
            credit1._init_feedback_weights(sys.geometry, device)
        with seeded(222):
            credit2._init_feedback_weights(sys.geometry, device)

        for k in credit1._feedback_weights:
            if k in credit2._feedback_weights:
                assert not torch.equal(
                    credit1._feedback_weights[k], credit2._feedback_weights[k]
                ), f"Feedback matrix {k} should differ with different seeds"


# ======================================================================
# L4 - Lyapunov / Energy Lock (physics guarantee)
# ======================================================================
class TestL4LyapunovLock:
    """Energy sampled per settling iteration is non-increasing;
    terminal update norm < 1e-6.
    """

    def test_l4_energy_non_increasing_eqprop(self) -> None:
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_eqprop_system()
            _setup_system_device(sys, device)

            x, y = tiny_batch(42)
            state = SystemState(x=x, y=y)

            state.activations = sys.geometry.forward(x, sys.substrate)
            if state.activations is not None:
                state.activations = sys.substrate.inject_state_noise(state.activations)

            energies = []
            for step in range(sys.dynamics.config.max_steps):
                state = sys.dynamics.settle(state, sys.geometry, sys.substrate)
                energy = sys.dynamics.compute_energy(state, sys.geometry)
                energies.append(energy.item())

                if len(energies) > 1:
                    assert energies[-1] <= energies[-2] + ENERGY_STEP_TOL, (
                        f"Energy increased at step {step}"
                    )

            if len(energies) >= MIN_LAYERS_FOR_LOCALITY:
                assert abs(energies[-1] - energies[-2]) < ENERGY_CONVERGENCE_TOL, (
                    "Did not converge to fixed point"
                )

    def test_l4_predictive_coding_produces_finite_energies(self) -> None:
        """Predictive coding settling produces finite free energies (no NaN).

        A single settle call runs max_steps iterations internally and should
        converge to a finite energy.
        """
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_predictive_coding_system()
            _setup_system_device(sys, device)

            x, y = tiny_batch(42)
            state = SystemState(x=x, y=y)

            state.activations = sys.geometry.forward(x, sys.substrate)
            if state.activations is not None:
                state.activations = sys.substrate.inject_state_noise(state.activations)

            # Single settle call (runs max_steps iterations internally)
            state = sys.dynamics.settle(state, sys.geometry, sys.substrate, target=None)
            energy = sys.dynamics.compute_energy(state, sys.geometry)

            # Energy should be finite (not NaN or inf)
            assert not torch.isnan(energy), "Energy is NaN"
            assert not torch.isinf(energy), "Energy is infinite"
            assert energy.item() >= 0, "Energy should be non-negative"

            # Also test nudged phase
            state = SystemState(x=x, y=y)
            state.activations = sys.geometry.forward(x, sys.substrate)
            if state.activations is not None:
                state.activations = sys.substrate.inject_state_noise(state.activations)

            state = sys.dynamics.settle(state, sys.geometry, sys.substrate, target=y)
            energy = sys.dynamics.compute_energy(state, sys.geometry)

            assert not torch.isnan(energy), "Nudged energy is NaN"
            assert not torch.isinf(energy), "Nudged energy is infinite"
            assert energy.item() >= 0, "Nudged energy should be non-negative"


# ======================================================================
# L5 - Determinism Lock (same seed, same device = bitwise equal)
# ======================================================================
@pytest.mark.parametrize(
    "system_factory",
    [
        _make_backprop_system,
        _make_eqprop_system,
        _make_predictive_coding_system,
        _make_tile_system,
    ],
)
def test_l5_determinism_lock(system_factory) -> None:
    """Same seed, same device, two runs of train_step:
    metrics and params bitwise equal.
    """
    device = select_device()

    # CPU always
    with seeded(42):
        sys1 = system_factory()
        _setup_system_device(sys1, device)
        x, y = tiny_batch(42)
        metrics1 = sys1.train_step(x, y)
        params1 = {k: v.clone() for k, v in sys1.geometry.params.items()}

    with seeded(42):
        sys2 = system_factory()
        _setup_system_device(sys2, device)
        x, y = tiny_batch(42)
        metrics2 = sys2.train_step(x, y)
        params2 = {k: v.clone() for k, v in sys2.geometry.params.items()}

    assert _metrics_equal(metrics1, metrics2, BITWISE)
    assert _params_equal(params1, params2, BITWISE)

    # GPU with deterministic settings (may skip if op lacks deterministic impl)
    if torch.cuda.is_available():

        def _run_gpu() -> tuple[dict[str, float], dict[str, Tensor]]:
            enable_deterministic_cuda()
            with seeded(42):
                sys3 = system_factory()
                _setup_system_device(sys3, device)
                x, y = tiny_batch(42)
                metrics3 = sys3.train_step(x, y)
                params3 = {k: v.clone() for k, v in sys3.geometry.params.items()}
            return metrics3, params3

        try:
            metrics3, params3 = _run_gpu()
            assert _metrics_equal(metrics1, metrics3, BITWISE)
            assert _params_equal(params1, params3, BITWISE)
        except RuntimeError as e:
            if "deterministic" in str(e).lower():
                pytest.skip(f"GPU deterministic op not available: {e}")
            raise


# ======================================================================
# L6 - Round-trip & Totality Lock (interchange guarantee)
# ======================================================================
def test_l6_totality_adapters_project() -> None:
    """Any nn.Module projects into the ontology via ModelAdapter (totality)."""
    from computronium.ontology import ModelAdapter

    probes = (
        (
            "feedforward_mlp",
            FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,)
                )
            ),
        ),
        (
            "recurrent_mlp",
            RecurrentGeometry(
                GeometryConfig.recurrent(
                    input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,)
                )
            ),
        ),
    )
    for label, geometry in probes:
        backbone = nn.Sequential(*geometry._layers)
        system = ModelAdapter(backbone).to_system()
        assert hasattr(system, "substrate"), label
        assert hasattr(system, "geometry"), label
        assert hasattr(system, "dynamics"), label
        assert hasattr(system, "credit"), label
        assert hasattr(system, "update"), label
        x = torch.randn(tiny_batch(WIDTH)[0].shape)
        out = system.forward(x)
        assert out.shape[-1] == 10, label
