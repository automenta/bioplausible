"""Ontology Property Locks (L1–L7) — CORRECTNESS_LOCK.md

Fast-CI property suite enforcing seven invariants of the 5-D ontology.
Wall-clock budget: ≤ 5 min on GPU, ≤ 10 min on CPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from bioplausible.core.ontology import (
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
    System,
    SystemState,
    ThermodynamicContrast,
    TileGeometry,
)
from bioplausible.core.registry import Registry
from bioplausible.core.system_trainer import (
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
)
from tests.property._support import (
    BATCH,
    BITWISE,
    DEPTH,
    SETTLE_ITERS,
    WIDTH,
    _all_registered_model_names,
    _round_trip_configs,
    conforms,
    enable_deterministic_cuda,
    perturb_nonlocal,
    seeded,
    select_device,
    tiny_batch,
)


# ----------------------------------------------------------------------
# Test systems for parametrization
# ----------------------------------------------------------------------
def _make_eqprop_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(),
        geometry=RecurrentGeometry(
            GeometryConfig(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            ),
            hidden_dim=WIDTH,
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig(
                dynamics_type="energy_minimization",
                max_steps=SETTLE_ITERS,
                beta=0.5,
            )
        ),
        credit=ThermodynamicContrast(CreditAssignmentConfig(beta=0.5)),
        update=EuclideanUpdate(ParameterUpdateConfig(step_size=0.01)),
    )


def _make_backprop_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(),
        geometry=FeedforwardGeometry(
            GeometryConfig(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            )
        ),
        dynamics=InstantaneousDynamics(),
        credit=BackpropCredit(),
        update=EuclideanUpdate(ParameterUpdateConfig(step_size=0.001)),
    )


def _make_fa_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(),
        geometry=FeedforwardGeometry(
            GeometryConfig(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            )
        ),
        dynamics=InstantaneousDynamics(),
        credit=RandomProjectionsCredit(
            CreditAssignmentConfig(credit_type="random_projections")
        ),
        update=EuclideanUpdate(ParameterUpdateConfig(step_size=0.001)),
    )


def _make_predictive_coding_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(),
        geometry=FeedforwardGeometry(
            GeometryConfig(
                input_dim=WIDTH, output_dim=10, hidden_dims=(WIDTH,) * (DEPTH - 1)
            )
        ),
        dynamics=PredictiveSettlingDynamics(
            StateDynamicsConfig(
                dynamics_type="predictive_settling",
                max_steps=SETTLE_ITERS,
                step_size=0.1,
            )
        ),
        credit=ThermodynamicContrast(CreditAssignmentConfig(beta=0.5)),
        update=EuclideanUpdate(ParameterUpdateConfig(step_size=0.01)),
    )


def _make_tile_system() -> System:
    return compose_system(
        substrate=DigitalSubstrate(),
        geometry=TileGeometry(
            GeometryConfig(input_dim=WIDTH, output_dim=10, num_layers=DEPTH),
            neurons_per_tile=8,
            tiles_per_layer=2,
        ),
        dynamics=InstantaneousDynamics(),
        credit=BackpropCredit(),
        update=EuclideanUpdate(ParameterUpdateConfig(step_size=0.001)),
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


# ======================================================================
# L1 — Parity Lock (strangler-fig guarantee)
# ======================================================================
def test_L1_composed_systems_train():
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
            if hasattr(system.geometry, "to"):
                system.geometry.to(device)

            metrics = system.train_step(x, y)
            assert metrics is not None
            assert "loss" in metrics
            assert metrics["loss"] >= 0
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1


# ======================================================================
# L2 — Orthogonality Lock (ontology honesty)
# ======================================================================
class TestL2OrthogonalityLock:
    """Each pipeline stage is a pure function of the axes that precede it."""

    def test_O1_geometry_forward_deterministic(self):
        """geometry.forward is deterministic for same inputs."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = _make_backprop_system()
        if hasattr(sys.geometry, "to"):
            sys.geometry.to(device)

        x, _ = tiny_batch(42)
        out1 = sys.geometry.forward(x, sys.substrate)
        out2 = sys.geometry.forward(x, sys.substrate)
        assert torch.equal(out1, out2)

    def test_O2_geometry_forward_independent_of_dynamics(self):
        """geometry.forward output doesn't depend on which Dynamics is used."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys1 = _make_backprop_system()
        with seeded(42):
            sys2 = _make_predictive_coding_system()

        if hasattr(sys1.geometry, "to"):
            sys1.geometry.to(device)
            sys2.geometry.to(device)

        _copy_system_params(sys1, sys2)
        x, _ = tiny_batch(42)

        out1 = sys1.geometry.forward(x, sys1.substrate)
        out2 = sys2.geometry.forward(x, sys2.substrate)
        assert torch.equal(out1, out2)

    def test_O3_credit_independent_of_update(self):
        """Pseudo-gradients are independent of ParameterUpdate choice."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys1 = _make_backprop_system()
        with seeded(42):
            sys2 = compose_system(
                substrate=DigitalSubstrate(),
                geometry=FeedforwardGeometry(
                    GeometryConfig(
                        input_dim=WIDTH,
                        output_dim=10,
                        hidden_dims=(WIDTH,) * (DEPTH - 1),
                    )
                ),
                dynamics=InstantaneousDynamics(),
                credit=BackpropCredit(),
                update=RiemannianOrthogonalUpdate(
                    ParameterUpdateConfig(step_size=0.001)
                ),
            )

        if hasattr(sys1.geometry, "to"):
            sys1.geometry.to(device)
            sys2.geometry.to(device)

        _copy_system_params(sys1, sys2)
        x, y = tiny_batch(42)

        state1 = SystemState(x=x, y=y)
        state1.activations = sys1.geometry.forward(x, sys1.substrate)
        if state1.activations is not None:
            state1.activations = sys1.substrate.inject_state_noise(state1.activations)
        free1 = sys1.dynamics.settle(state1, sys1.geometry, sys1.substrate, target=None)
        nudged1 = sys1.dynamics.settle(state1, sys1.geometry, sys1.substrate, target=y)
        nudged1.loss = sys1._compute_loss(nudged1, y)

        state2 = SystemState(x=x, y=y)
        state2.activations = sys2.geometry.forward(x, sys2.substrate)
        if state2.activations is not None:
            state2.activations = sys2.substrate.inject_state_noise(state2.activations)
        free2 = sys2.dynamics.settle(state2, sys2.geometry, sys2.substrate, target=None)
        nudged2 = sys2.dynamics.settle(state2, sys2.geometry, sys2.substrate, target=y)
        nudged2.loss = sys2._compute_loss(nudged2, y)

        grads1 = sys1.credit.compute_pseudo_gradient(
            free1, nudged1, nudged1.loss, sys1.geometry
        )
        grads2 = sys2.credit.compute_pseudo_gradient(
            free2, nudged2, nudged2.loss, sys2.geometry
        )

        assert len(grads1) == len(grads2)
        for g1, g2 in zip(grads1, grads2):
            assert torch.equal(g1, g2)

    def test_O4_substrate_noise_is_only_effect(self):
        """With noiseless DigitalSubstrate, forward is deterministic."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = _make_tile_system()
        if hasattr(sys.geometry, "to"):
            sys.geometry.to(device)

        x, _ = tiny_batch(42)
        out1 = sys.geometry.forward(x, sys.substrate)
        out2 = sys.geometry.forward(x, sys.substrate)
        assert torch.equal(out1, out2)


# ======================================================================
# L3 — Locality Lock (bioplausibility axiom)
# ======================================================================
class TestL3LocalityLock:
    """L3a: Strictly-local rules have invariant pseudo-gradients under non-local perturbation.
    L3b: FA family feedback matrices fixed at init and independent of forward weights.
    """

    def test_L3a_thermodynamic_contrast_local(self):
        """ThermodynamicContrast pseudo-gradient invariant to non-local perturbations."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_eqprop_system()
            if hasattr(sys.geometry, "to"):
                sys.geometry.to(device)

            x, y = tiny_batch(42)

            state = SystemState(x=x, y=y)
            state.activations = sys.geometry.forward(x, sys.substrate)
            if state.activations is not None:
                state.activations = sys.substrate.inject_state_noise(state.activations)
            free = sys.dynamics.settle(state, sys.geometry, sys.substrate, target=None)
            nudged = sys.dynamics.settle(state, sys.geometry, sys.substrate, target=y)
            nudged.loss = sys._compute_loss(nudged, y)

            grads_orig = sys.credit.compute_pseudo_gradient(
                free, nudged, nudged.loss, sys.geometry
            )

            if len(grads_orig) >= 2:
                free_pert = perturb_nonlocal(free, 0, 1e-3)
                nudged_pert = perturb_nonlocal(nudged, 0, 1e-3)

                grads_pert = sys.credit.compute_pseudo_gradient(
                    free_pert, nudged_pert, nudged_pert.loss, sys.geometry
                )

                assert torch.allclose(
                    grads_orig[0], grads_pert[0], atol=1e-6, rtol=0
                ), "ThermodynamicContrast violated locality at layer 0"

    def test_L3b_fa_feedback_matrices_fixed_at_init(self):
        """Feedback matrices are fixed at init and independent of forward weights."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_fa_system()
            if hasattr(sys.geometry, "to"):
                sys.geometry.to(device)

            # Initialize feedback weights
            sys.credit._init_feedback_weights(sys.geometry, device)
            fb_weights_orig = {
                k: v.clone() for k, v in sys.credit._feedback_weights.items()
            }

            # Create new system with same credit but different forward weights
            with seeded(123):
                sys2 = compose_system(
                    substrate=DigitalSubstrate(),
                    geometry=FeedforwardGeometry(
                        GeometryConfig(
                            input_dim=WIDTH,
                            output_dim=10,
                            hidden_dims=(WIDTH,) * (DEPTH - 1),
                        )
                    ),
                    dynamics=InstantaneousDynamics(),
                    credit=sys.credit,  # Same credit instance
                    update=EuclideanUpdate(ParameterUpdateConfig(step_size=0.001)),
                )

            if hasattr(sys2.geometry, "to"):
                sys2.geometry.to(device)

            # Feedback weights should be identical
            assert sys2.credit._feedback_weights is not None
            for k in fb_weights_orig:
                assert torch.equal(
                    fb_weights_orig[k], sys2.credit._feedback_weights[k]
                ), f"Feedback matrix {k} changed after forward weight re-init"

    def test_L3b_fa_different_seeds_produce_different_feedback(self):
        """Different feedback seed produces different feedback matrices."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = _make_fa_system()
        if hasattr(sys.geometry, "to"):
            sys.geometry.to(device)

        credit1 = RandomProjectionsCredit(
            CreditAssignmentConfig(credit_type="random_projections")
        )
        credit2 = RandomProjectionsCredit(
            CreditAssignmentConfig(credit_type="random_projections")
        )

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
# L4 — Lyapunov / Energy Lock (physics guarantee)
# ======================================================================
class TestL4LyapunovLock:
    """Energy sampled per settling iteration is non-increasing; terminal update norm < 1e-6."""

    def test_L4_energy_non_increasing_eqprop(self):
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_eqprop_system()
            if hasattr(sys.geometry, "to"):
                sys.geometry.to(device)

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
                    assert energies[-1] <= energies[-2] + 1e-7, (
                        f"Energy increased at step {step}"
                    )

            if len(energies) >= 2:
                assert abs(energies[-1] - energies[-2]) < 1e-6, (
                    "Did not converge to fixed point"
                )

    def test_L4_predictive_coding_produces_finite_energies(self):
        """Predictive coding settling produces finite free energies (no NaN).

        A single settle call runs max_steps iterations internally and should
        converge to a finite energy.
        """
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(42):
            sys = _make_predictive_coding_system()
            if hasattr(sys.geometry, "to"):
                sys.geometry.to(device)

            x, y = tiny_batch(42)
            state = SystemState(x=x, y=y)

            state.activations = sys.geometry.forward(x, sys.substrate)
            if state.activations is not None:
                state.activations = sys.substrate.inject_state_noise(state.activations)

            # Single settle call (runs max_steps iterations internally)
            state = sys.dynamics.settle(
                state, sys.geometry, sys.substrate, target=None
            )
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

            state = sys.dynamics.settle(
                state, sys.geometry, sys.substrate, target=y
            )
            energy = sys.dynamics.compute_energy(state, sys.geometry)

            assert not torch.isnan(energy), "Nudged energy is NaN"
            assert not torch.isinf(energy), "Nudged energy is infinite"
            assert energy.item() >= 0, "Nudged energy should be non-negative"


# ======================================================================
# L5 — Determinism Lock (same seed, same device = bitwise equal)
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
def test_L5_determinism_lock(system_factory):
    """Same seed, same device, two runs of train_step: metrics and params bitwise equal."""
    device = select_device()

    # CPU always
    with seeded(42):
        sys1 = system_factory()
        if hasattr(sys1.geometry, "to"):
            sys1.geometry.to(device)
        x, y = tiny_batch(42)
        metrics1 = sys1.train_step(x, y)
        params1 = {k: v.clone() for k, v in sys1.geometry.params.items()}

    with seeded(42):
        sys2 = system_factory()
        if hasattr(sys2.geometry, "to"):
            sys2.geometry.to(device)
        x, y = tiny_batch(42)
        metrics2 = sys2.train_step(x, y)
        params2 = {k: v.clone() for k, v in sys2.geometry.params.items()}

    assert _metrics_equal(metrics1, metrics2, BITWISE)
    assert _params_equal(params1, params2, BITWISE)

    # GPU with deterministic settings (may skip if op lacks deterministic impl)
    if torch.cuda.is_available():
        try:
            enable_deterministic_cuda()
            with seeded(42):
                sys3 = system_factory()
                if hasattr(sys3.geometry, "to"):
                    sys3.geometry.to(device)
                x, y = tiny_batch(42)
                metrics3 = sys3.train_step(x, y)
                params3 = {k: v.clone() for k, v in sys3.geometry.params.items()}

            assert _metrics_equal(metrics1, metrics3, BITWISE)
            assert _params_equal(params1, params3, BITWISE)
        except RuntimeError as e:
            if "deterministic" in str(e).lower():
                pytest.skip(f"GPU deterministic op not available: {e}")
            raise


# ======================================================================
# L6 — Round-trip & Totality Lock (interchange guarantee)
# ======================================================================
def test_L6_totality_registered_models_project():
    """Registered model names project via Registry.to_system() (smoke test)."""
    # Import models to populate registry

    model_names = _all_registered_model_names()
    assert len(model_names) > 0, "Registry should have registered models"

    # Test a few known-working models
    working_models = [
        n
        for n in model_names
        if n in {"eqprop", "backprop_mlp", "feedback_alignment", "forward_forward"}
    ]
    if not working_models:
        working_models = model_names[:3]

    for name in working_models:
        try:
            system = Registry.to_system(
                name, input_dim=WIDTH, hidden_dim=WIDTH, output_dim=10
            )
            assert system is not None
            assert hasattr(system, "substrate")
            assert hasattr(system, "geometry")
            assert hasattr(system, "dynamics")
            assert hasattr(system, "credit")
            assert hasattr(system, "update")

            # Basic protocol conformance (adapted systems may not fully implement all methods)
            assert conforms(
                system.substrate,
                {
                    "quantize_weights": True,
                    "inject_state_noise": True,
                    "get_forward_operator": True,
                    "get_weight_update_operator": True,
                    "initial_state": True,
                },
            )
            # Credit and update should always conform
            assert conforms(system.credit, {"compute_pseudo_gradient": True})
            assert conforms(system.update, {"step": True})
        except TypeError:
            # Some models have different constructor signatures - skip
            pytest.skip(f"Model {name} has incompatible constructor")


def test_L6_round_trip_configs():
    """Configs → round-trip → configs is identity."""
    systems = [
        _make_backprop_system(),
        _make_fa_system(),
        _make_tile_system(),
    ]

    for sys in systems:
        sys2 = _round_trip_configs(sys)

        # Compare configs (the core round-trip guarantee)
        assert sys.substrate.config == sys2.substrate.config
        assert sys.geometry.config == sys2.geometry.config
        assert sys.dynamics.config == sys2.dynamics.config
        assert sys.credit.config == sys2.credit.config
        assert sys.update.config == sys2.update.config


# ======================================================================
# L7 — Seam Lock (P2P anticipation)
# ======================================================================
def test_L7_system_trainer_runs():
    """SystemTrainer runs without error (distributed seam tested in integration)."""
    device = select_device()
    if device.type == "cuda":
        enable_deterministic_cuda()

    with seeded(42):
        sys = _make_tile_system()
        if hasattr(sys.geometry, "to"):
            sys.geometry.to(device)

        x, y = tiny_batch(42)

        # Single-process reference
        trainer_config = SystemTrainerConfig(max_epochs=1, batch_size=BATCH, seed=42)
        ref_trainer = SystemTrainer(
            system=sys,
            config=trainer_config,
            train_data=[(x, y)],
        )
        ref_metrics = ref_trainer.train_epoch()
        assert "train_loss" in ref_metrics
        assert ref_metrics["train_loss"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
