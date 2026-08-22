"""System Specification Interchange Format Tests.

Tests for the .system interchange format round-trip:
System -> to_spec() -> json -> from_spec() -> System
"""

from __future__ import annotations

import json

import pytest
import torch

from bioplausible.core.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    ElasticConsolidationUpdate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    NaturalGradientUpdate,
    ParameterUpdateConfig,
    RecurrentGeometry,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
    StateDynamicsConfig,
    SubstrateConfig,
    TargetInversionCredit,
    ThermodynamicContrast,
)
from bioplausible.core.system_trainer import compose_system
from tests.property._support import (
    DEPTH,
    WIDTH,
    enable_deterministic_cuda,
    seeded,
    select_device,
    tiny_batch,
)


def _setup_system_device(sys, device: torch.device):
    """Move system to device if supported."""
    if hasattr(sys.geometry, "to"):
        sys.geometry.to(device)
    return sys


def _run_system_forward(system, x: torch.Tensor) -> torch.Tensor:
    """Run a single forward pass and return output."""
    return system.forward(x)


def _make_random_system(device: torch.device) -> tuple:
    """Create a random valid System composition using deterministic components only."""
    import random

    substrate = DigitalSubstrate(SubstrateConfig(
        precision="float32",
        noise_level=0.0,
        weight_bounds=None,
        sparsity=0.0,
        device="cpu",
    ))

    # Use deterministic geometry types only
    geo_type = random.choice(["feedforward", "recurrent"])
    if geo_type == "recurrent":
        geometry = RecurrentGeometry(
            GeometryConfig(
                input_dim=WIDTH,
                output_dim=10,
                hidden_dims=(WIDTH,) * (DEPTH - 1),
                num_layers=DEPTH,
                topology_type="recurrent",
                connectivity=None,
                recurrent_weight=None,
            ),
            hidden_dim=WIDTH,
        )
    else:
        geometry = FeedforwardGeometry(
            GeometryConfig(
                input_dim=WIDTH,
                output_dim=10,
                hidden_dims=(WIDTH,) * (DEPTH - 1),
                num_layers=DEPTH,
                topology_type="feedforward",
                connectivity=None,
                recurrent_weight=None,
            )
        )

    # Use deterministic dynamics only
    dynamics = InstantaneousDynamics(StateDynamicsConfig(
        dynamics_type="instantaneous",
        max_steps=1,
        convergence_threshold=1e-4,
        convergence_start=1,
        step_size=0.1,
        beta=0.1,
        track_free_energy_per_iter=False,
    ))

    # Use deterministic credit types only (no random projections, temporal trace)
    credit_type = random.choice([
        "thermodynamic_contrast",
        "local_goodness",
        "target_inversion",
    ])
    credit_map = {
        "thermodynamic_contrast": ThermodynamicContrast,
        "local_goodness": LocalGoodnessCredit,
        "target_inversion": TargetInversionCredit,
    }
    credit = credit_map[credit_type](CreditAssignmentConfig(
        credit_type=credit_type,
        beta=0.5,
        feedback_matrix=None,
        local_objective="mse",
        orthogonal_init=False,
        feedback_scale=0.01,
    ))

    # Use deterministic update types only
    update_type = random.choice([
        "euclidean",
        "riemannian_orthogonal",
        "spectral_constrained",
        "natural_gradient",
        "elastic_consolidation",
    ])
    update_map = {
        "euclidean": lambda: EuclideanUpdate(ParameterUpdateConfig(
            update_type="euclidean",
            step_size=0.01,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )),
        "riemannian_orthogonal": lambda: RiemannianOrthogonalUpdate(
            ParameterUpdateConfig(
                update_type="riemannian_orthogonal",
                step_size=0.01,
                momentum=0.9,
                ortho_steps=5,
                spectral_norm=1.0,
                fisher_damping=1e-3,
                ewc_lambda=1000.0,
            )
        ),
        "spectral_constrained": lambda: SpectralConstrainedUpdate(
            ParameterUpdateConfig(
                update_type="spectral_constrained",
                step_size=0.01,
                momentum=0.9,
                ortho_steps=5,
                spectral_norm=1.0,
                fisher_damping=1e-3,
                ewc_lambda=1000.0,
            )
        ),
        "natural_gradient": lambda: NaturalGradientUpdate(
            ParameterUpdateConfig(
                update_type="natural_gradient",
                step_size=0.01,
                momentum=0.9,
                ortho_steps=5,
                spectral_norm=1.0,
                fisher_damping=1e-3,
                ewc_lambda=1000.0,
            )
        ),
        "elastic_consolidation": lambda: ElasticConsolidationUpdate(
            ParameterUpdateConfig(
                update_type="elastic_consolidation",
                step_size=0.01,
                momentum=0.9,
                ortho_steps=5,
                spectral_norm=1.0,
                fisher_damping=1e-3,
                ewc_lambda=1000.0,
            )
        ),
    }
    update = update_map[update_type]()

    sys = compose_system(substrate, geometry, dynamics, credit, update)
    _setup_system_device(sys, device)
    return sys, credit_type, update_type


class TestSystemSpecRoundTrip:
    """Test System spec serialization and deserialization."""

    @pytest.mark.parametrize("seed", range(10))
    def test_spec_round_trip(self, seed: int) -> None:
        """Generate 10 random valid Systems, round-trip through spec, verify identity."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        with seeded(seed):
            sys, credit_type, update_type = _make_random_system(device)

        # Serialize to spec
        spec = sys.to_spec()

        # Verify schema version
        assert spec["schema_version"] == "1.0"

        # Verify all 5 axes present
        assert "substrate" in spec
        assert "geometry" in spec
        assert "dynamics" in spec
        assert "credit" in spec
        assert "update" in spec

        # Round-trip through JSON
        json_str = json.dumps(spec)
        spec_loaded = json.loads(json_str)

        # Deserialize from spec (within seed context for deterministic recurrent weight init)
        with seeded(42):
            sys_reconstructed = sys.from_spec(spec_loaded)
        _setup_system_device(sys_reconstructed, device)

        # Test that reconstructed system produces bitwise-identical outputs on forward pass
        with seeded(42):
            x, _ = tiny_batch(42)

        # Run forward pass on both systems (should be bitwise identical for deterministic systems)
        out_orig = _run_system_forward(sys, x)
        out_recon = _run_system_forward(sys_reconstructed, x)

        # Compare outputs (should be bitwise identical)
        assert torch.equal(out_orig, out_recon), (
            f"Forward pass mismatch: {out_orig} != {out_recon} "
            f"(credit={credit_type}, update={update_type})"
        )

    def test_spec_contains_all_configs(self) -> None:
        """Verify spec contains all config fields for each axis."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = compose_system(
            substrate=DigitalSubstrate(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.01,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            ),
            geometry=FeedforwardGeometry(
                GeometryConfig(
                    input_dim=WIDTH,
                    output_dim=10,
                    hidden_dims=(WIDTH,),
                    num_layers=1,
                    topology_type="feedforward",
                    connectivity=None,
                    recurrent_weight=None,
                )
            ),
            dynamics=EnergyMinimizationDynamics(
                StateDynamicsConfig(
                    dynamics_type="energy_minimization",
                    max_steps=30,
                    convergence_threshold=1e-4,
                    convergence_start=5,
                    step_size=0.1,
                    beta=0.5,
                    track_free_energy_per_iter=False,
                )
            ),
            credit=ThermodynamicContrast(
                CreditAssignmentConfig(
                    credit_type="thermodynamic_contrast",
                    beta=0.5,
                    feedback_matrix=None,
                    local_objective="mse",
                    orthogonal_init=False,
                    feedback_scale=0.01,
                )
            ),
            update=RiemannianOrthogonalUpdate(
                ParameterUpdateConfig(
                    update_type="riemannian_orthogonal",
                    step_size=0.01,
                    momentum=0.9,
                    ortho_steps=5,
                    spectral_norm=1.0,
                    fisher_damping=1e-3,
                    ewc_lambda=1000.0,
                )
            ),
        )
        _setup_system_device(sys, device)

        spec = sys.to_spec()

        # Check substrate config
        assert spec["substrate"]["precision"] == "float32"
        assert spec["substrate"]["noise_level"] == 0.01

        # Check geometry config
        assert spec["geometry"]["input_dim"] == WIDTH
        assert spec["geometry"]["output_dim"] == 10
        assert spec["geometry"]["topology_type"] == "feedforward"

        # Check dynamics config
        assert spec["dynamics"]["dynamics_type"] == "energy_minimization"
        assert spec["dynamics"]["max_steps"] == 30
        assert spec["dynamics"]["beta"] == 0.5

        # Check credit config
        assert spec["credit"]["credit_type"] == "thermodynamic_contrast"
        assert spec["credit"]["beta"] == 0.5

        # Check update config
        assert spec["update"]["update_type"] == "riemannian_orthogonal"
        assert spec["update"]["step_size"] == 0.01
        assert spec["update"]["ortho_steps"] == 5

    def test_spec_rejects_wrong_version(self) -> None:
        """from_spec should reject unsupported schema versions."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = compose_system(
            substrate=DigitalSubstrate(SubstrateConfig(
                precision="float32",
                noise_level=0.0,
                weight_bounds=None,
                sparsity=0.0,
                device="cpu",
            )),
            geometry=FeedforwardGeometry(
                GeometryConfig(
                    input_dim=WIDTH,
                    output_dim=10,
                    hidden_dims=(),
                    num_layers=0,
                    topology_type="feedforward",
                    connectivity=None,
                    recurrent_weight=None,
                )
            ),
            dynamics=InstantaneousDynamics(StateDynamicsConfig(
                dynamics_type="instantaneous",
                max_steps=1,
                convergence_threshold=1e-4,
                convergence_start=1,
                step_size=0.1,
                beta=0.1,
                track_free_energy_per_iter=False,
            )),
            credit=ThermodynamicContrast(CreditAssignmentConfig(
                credit_type="thermodynamic_contrast",
                beta=0.5,
                feedback_matrix=None,
                local_objective="mse",
                orthogonal_init=False,
                feedback_scale=0.01,
            )),
            update=EuclideanUpdate(ParameterUpdateConfig(
                update_type="euclidean",
                step_size=0.01,
                momentum=0.9,
                ortho_steps=5,
                spectral_norm=1.0,
                fisher_damping=1e-3,
                ewc_lambda=1000.0,
            )),
        )
        _setup_system_device(sys, device)

        spec = sys.to_spec()
        spec["schema_version"] = "2.0"

        with pytest.raises(ValueError, match="Unsupported schema version"):
            sys.from_spec(spec)

    def test_spec_preserves_configs(self) -> None:
        """After round-trip, all configs should be identical."""
        device = select_device()
        if device.type == "cuda":
            enable_deterministic_cuda()

        sys = compose_system(
            substrate=DigitalSubstrate(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.01,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            ),
            geometry=FeedforwardGeometry(
                GeometryConfig(
                    input_dim=WIDTH,
                    output_dim=10,
                    hidden_dims=(WIDTH,) * (DEPTH - 1),
                    num_layers=DEPTH,
                    topology_type="feedforward",
                    connectivity=None,
                    recurrent_weight=None,
                )
            ),
            dynamics=InstantaneousDynamics(StateDynamicsConfig(
                dynamics_type="instantaneous",
                max_steps=1,
                convergence_threshold=1e-4,
                convergence_start=1,
                step_size=0.1,
                beta=0.1,
                track_free_energy_per_iter=False,
            )),
            credit=ThermodynamicContrast(CreditAssignmentConfig(
                credit_type="thermodynamic_contrast",
                beta=0.5,
                feedback_matrix=None,
                local_objective="mse",
                orthogonal_init=False,
                feedback_scale=0.01,
            )),
            update=EuclideanUpdate(ParameterUpdateConfig(
                update_type="euclidean",
                step_size=0.01,
                momentum=0.9,
                ortho_steps=5,
                spectral_norm=1.0,
                fisher_damping=1e-3,
                ewc_lambda=1000.0,
            )),
        )
        _setup_system_device(sys, device)

        # Capture original configs
        orig_substrate_cfg = sys.substrate.config
        orig_geometry_cfg = sys.geometry.config
        orig_dynamics_cfg = sys.dynamics.config
        orig_credit_cfg = sys.credit.config
        orig_update_cfg = sys.update.config

        # Serialize and round-trip
        spec = sys.to_spec()
        json_str = json.dumps(spec)
        spec_loaded = json.loads(json_str)
        sys_recon = sys.from_spec(spec_loaded)
        _setup_system_device(sys_recon, device)

        # Configs should be identical
        assert sys_recon.substrate.config == orig_substrate_cfg
        assert sys_recon.geometry.config == orig_geometry_cfg
        assert sys_recon.dynamics.config == orig_dynamics_cfg
        assert sys_recon.credit.config == orig_credit_cfg
        assert sys_recon.update.config == orig_update_cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
