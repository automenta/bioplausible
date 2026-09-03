"""R5b-0 implementation-fidelity gate: per-axis probes over the campaign grid.

Pins the gate's verdicts on the ``joint_grid`` coordinates. A verdict flip —
e.g. after wiring plasticity into the episode pipeline or implementing a
stubbed credit — is the R5b-D lock-④ regression signal, not test breakage.
"""

from __future__ import annotations

import pytest

from computronium.cli.campaign import _get_search_space
from computronium.core.campaign import space_grid
from computronium.core.campaign.evaluation import (
    IncompatibleCoordinateError,
    build_coordinate_system,
)
from computronium.core.campaign.fidelity import (
    CoordinateFidelity,
    check_coordinate_fidelity,
    defect_filtered_attribution,
    fidelity_manifest,
)
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.resources import ResourceUsage

GRID = space_grid(_get_search_space("joint_grid"))


def make_record(
    coordinate: str, *, task: str = "synthetic", seed: int = 0
) -> FrontierRecord:
    return FrontierRecord(
        coordinate=coordinate,
        task_name=task,
        task_loss=1.0,
        task_accuracy=0.5,
        adaptation_time=1,
        rho_jacobian=0.9,
        lyapunov_local=0.0,
        settling_time=1.0,
        basin_stability=1.0,
        resources=ResourceUsage(),
        seed=seed,
    )


def _check(verdict, axis: str):
    return next(c for c in verdict.checks if c.axis == axis)


@pytest.fixture(scope="module")
def manifest() -> dict[str, CoordinateFidelity]:
    return fidelity_manifest(GRID)


class TestDynamicsFidelity:
    def test_energy_minimization_settles(self) -> None:
        verdict = check_coordinate_fidelity(
            "digital/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean"
        )
        d = [c for c in verdict.checks if c.axis == "dynamics"]
        assert all(c.status == "pass" for c in d)
        assert "energy" in d[0].detail
        assert d[2].status == "pass"  # nudged ≠ free (target-responsive)

    def test_instantaneous_is_single_pass_and_nudges(self) -> None:
        """Instantaneous dynamics is a single pass but now nudges the output
        when a target is provided (for contrastive credits)."""
        verdict = check_coordinate_fidelity(
            "digital/recurrent/instantaneous/null/random_projections/euclidean"
        )
        d = [c for c in verdict.checks if c.axis == "dynamics"]
        assert d[0].status == "pass"
        assert "single pass, idempotent" in d[0].detail
        assert d[1].status == "pass"
        assert "nudged phase differs from free phase" in d[1].detail

    def test_predictive_settling_settles_and_nudges(self) -> None:
        """Predictive settling now descends energy and responds to the nudged target."""
        verdict = check_coordinate_fidelity(
            "digital/feedforward/predictive_settling/null/local_goodness/euclidean"
        )
        d = [c for c in verdict.checks if c.axis == "dynamics"]
        assert d[0].status == "pass"
        assert "prediction-error energy" in d[0].detail
        assert d[1].status == "pass"  # deterministic repeat
        assert d[2].status == "pass"
        assert "nudged phase differs from free phase" in d[2].detail
        # Credit and update should also pass now
        credit_check = _check(verdict, "credit")
        assert credit_check.status == "pass"
        update_check = _check(verdict, "update")
        assert update_check.status == "pass"

    def test_spike_integration_membrane_bounded_and_spikes_tracked(self) -> None:
        verdict = check_coordinate_fidelity(
            "digital/feedforward/spike_integration/null/temporal_trace/euclidean"
        )
        d = [c for c in verdict.checks if c.axis == "dynamics"]
        assert all(c.status == "pass" for c in d)
        assert "membrane bounded" in d[1].detail
        assert "spike counts" in d[2].detail
        assert "target-blind by spec" in d[4].detail

    def test_diffusion_langevin_descent_target_responsive(self) -> None:
        """Langevin descent runs with nudged-Langevin (EP-style) target term:
        seed-locked free vs nudged settles differ, so supervision enters
        the dynamics."""
        verdict = check_coordinate_fidelity(
            "digital/feedforward/diffusion/null/temporal_trace/euclidean"
        )
        d = [c for c in verdict.checks if c.axis == "dynamics"]
        assert d[0].status == "pass"  # finite
        assert d[1].status == "pass"  # energy descends
        assert d[2].status == "pass"  # Langevin noise present
        assert d[3].status == "pass"  # target-responsive nudge
        assert "target-responsive" in d[3].detail


class TestR39ValidityMatrix:
    def test_instantaneous_thermodynamic_contrast_rejected_at_composition(self) -> None:
        """R3.9 fence: a contrastive settling credit paired with a single
        target-blind pass is conceptually dead — rejected at composition,
        not quarantined at attribution."""
        with pytest.raises(IncompatibleCoordinateError) as excinfo:
            build_coordinate_system(
                "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean"
            )
        assert "contrastive" in str(excinfo.value)

    def test_incompatible_coordinate_fenced_in_gate(self) -> None:
        """The gate records the fence as a coordinate-level fail verdict."""
        verdict = check_coordinate_fidelity(
            "digital/recurrent/instantaneous/routing/thermodynamic_contrast/euclidean"
        )
        assert not verdict.passed
        assert len(verdict.checks) == 1
        assert verdict.checks[0].axis == "coordinate"
        assert "invalid pairing" in verdict.checks[0].detail


class TestCreditFidelity:
    @pytest.mark.parametrize("credit", ["random_projections", "local_goodness"])
    def test_phase_contrast_credit_under_instantaneous(self, credit: str) -> None:
        """Both random_projections and local_goodness now work under instantaneous
        dynamics since nudging was implemented (R11.2.10 fix)."""
        verdict = check_coordinate_fidelity(
            f"digital/feedforward/instantaneous/null/{credit}/euclidean"
        )
        credit_check = _check(verdict, "credit")
        assert credit_check.status == "pass"
        assert "pseudo-gradient" in credit_check.detail

    def test_implemented_credits_signal_under_energy(self) -> None:
        for credit in (
            "thermodynamic_contrast",
            "random_projections",
            "local_goodness",
            "temporal_trace",
            "target_inversion",
            "homeostatic",
        ):
            verdict = check_coordinate_fidelity(
                f"digital/feedforward/energy_minimization/null/{credit}/euclidean"
            )
            assert _check(verdict, "credit").status == "pass", (
                f"{credit} credit failed: {_check(verdict, 'credit').detail}"
            )

    def test_implemented_credits_signal_under_energy_recurrent(self) -> None:
        """Recurrent geometry: the EM settle graph reaches θ through the
        kernel, and surplus weights (recurrent self-connections) receive
        their gradient via positional act-transition mapping."""
        for credit in (
            "thermodynamic_contrast",
            "random_projections",
            "local_goodness",
            "temporal_trace",
            "target_inversion",
            "homeostatic",
        ):
            verdict = check_coordinate_fidelity(
                f"digital/recurrent/energy_minimization/null/{credit}/euclidean"
            )
            assert _check(verdict, "credit").status == "pass", (
                f"{credit} credit failed (recurrent): "
                f"{_check(verdict, 'credit').detail}"
            )

    def test_local_goodness_now_implemented(self) -> None:
        """LocalGoodnessCredit is no longer a stub: compute_pseudo_gradient
        produces signal."""
        verdict = check_coordinate_fidelity(
            "digital/feedforward/energy_minimization/null/local_goodness/euclidean"
        )
        credit_check = _check(verdict, "credit")
        assert credit_check.status == "pass"
        assert "pseudo-gradient" in credit_check.detail


class TestUpdateFidelity:
    @pytest.mark.parametrize("update", ["euclidean", "spectral_constrained"])
    def test_update_moves_params_given_signal(self, update: str) -> None:
        verdict = check_coordinate_fidelity(
            f"digital/feedforward/energy_minimization/null/thermodynamic_contrast/{update}"
        )
        assert _check(verdict, "update").status == "pass"

    def test_update_moves_params_local_goodness_instantaneous(self) -> None:
        """Update now moves params for local_goodness under instantaneous since
        nudging provides the phase contrast signal."""
        verdict = check_coordinate_fidelity(
            "digital/feedforward/instantaneous/null/local_goodness/euclidean"
        )
        update_check = _check(verdict, "update")
        assert update_check.status == "pass"
        assert "parameters moved" in update_check.detail


class TestPlasticityFidelity:
    def test_null_plasticity_keeps_psi_const(self) -> None:
        verdict = check_coordinate_fidelity(
            "digital/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean"
        )
        m = _check(verdict, "plasticity")
        assert m.status == "pass"
        assert "ψ const" in m.detail

    @pytest.mark.parametrize("plasticity", ["routing", "fast_weights"])
    def test_nonnull_plasticity_steps_and_modulates(self, plasticity: str) -> None:
        """M-axis lock (upgraded, TODO8 Execution Order 9.v): plasticity.step
        is invoked by the episode pipeline AND ψ actually changes across the
        step; primitives with a modulate hook must produce ψ-sensitive
        activity (zeroed-ψ ≠ stepped-ψ)."""
        verdict = check_coordinate_fidelity(
            f"digital/feedforward/energy_minimization/{plasticity}"
            f"/thermodynamic_contrast/euclidean"
        )
        m = _check(verdict, "plasticity")
        assert m.status == "pass"
        assert "ψ stepped" in m.detail and "changed" in m.detail
        assert "ψ-sensitive" in m.detail

    def test_fast_weights_psi_receives_target_activity(self) -> None:
        """Regression pin: the pipeline's M-axis step must expose x AND y in
        z.activity — the FastWeight Hebbian update was silent-dead while only
        x was threaded (caught by the ψ non-const assertion)."""
        verdict = check_coordinate_fidelity(
            "digital/recurrent/energy_minimization/fast_weights/random_projections"
            "/euclidean"
        )
        assert _check(verdict, "plasticity").status == "pass"


class TestCapabilityManifest:
    def test_every_grid_coordinate_has_verdicts(self, manifest) -> None:
        assert set(manifest) == set(GRID)
        for verdict in manifest.values():
            axes = {c.axis for c in verdict.checks}
            assert axes in (
                {"dynamics", "credit", "update", "plasticity"},
                {"coordinate"},  # fenced invalid pairing (R3.9)
            )

    def test_passing_set_is_the_validated_subspace(self, manifest) -> None:
        """The capability manifest of the current grid: all energy_minimization
        coordinates (contrastive/autograd/trace credits now work on both
        geometries incl. surplus recurrent weights) plus instantaneous x
        random_projections and instantaneous x local_goodness (nudging now works).
        Instantaneous x thermodynamic_contrast is fenced invalid."""
        passing = {c for c, v in manifest.items() if v.passed}
        expected = {
            c
            for c in GRID
            if "/energy_minimization/" in c
            or ("/instantaneous/" in c and ("/random_projections/" in c or "/local_goodness/" in c))
        }
        assert passing == expected, (
            f"Passing: {sorted(passing)}, Expected: {sorted(expected)}"
        )
        # Count: 36 energy_minimization (3 geometries × 4 credits × 3 plasticities) +
        # 12 instantaneous×random_projections + 12 instantaneous×local_goodness = 60
        assert len(passing) == 60

    def test_defect_filtered_attribution_excludes_and_lists(self, manifest) -> None:
        records = [
            make_record(c, task=task, seed=seed)
            for c in GRID
            for task in ("synthetic", "parity")
            for seed in range(2)
        ]
        result = defect_filtered_attribution(records, manifest)
        assert result.n_records_total == len(records)
        assert result.n_records_passing == len(result.passing_coordinates) * 4
        assert set(result.excluded_coordinates) == set(GRID) - set(
            result.passing_coordinates
        )
        assert all(c in manifest for c in result.excluded_coordinates)
        # The passing subspace varies every axis (incl. dynamics), so all
        # five are attributable after the filter.
        attributed_axes = {a.axis for a in result.attributions}
        assert attributed_axes <= {
            "dynamics",
            "geometry",
            "credit",
            "update",
            "plasticity",
        }
        assert "dynamics" in attributed_axes
        assert "plasticity" in attributed_axes


def test_manifest_report_renders() -> None:
    """Sanity: the manifest serializes for the campaign report script."""
    import json

    manifest = fidelity_manifest(GRID[:2])
    payload = {
        c: {
            "passed": v.passed,
            "failures": v.failures,
            "checks": [
                {
                    "axis": k.axis,
                    "value": k.value,
                    "status": k.status,
                    "detail": k.detail,
                }
                for k in v.checks
            ],
        }
        for c, v in manifest.items()
    }
    text = json.dumps(payload)
    assert "passed" in text


class TestMetricHonesty:
    """imp-20: free_accuracy must not carry supervision leakage.

    An inert-credit coordinate (Δθ = 0) must show free_accuracy ≈ chance
    while its nudged accuracy may be high due to settle-to-target leakage.
    """

    def test_free_accuracy_is_not_supervision_leaked(self) -> None:
        """RandomProjectionsCredit with feedback_scale=0 produces zero Δθ.
        Under energy_minimization, nudged settle reaches target (high acc),
        but free settle accuracy should be at chance.
        """
        from computronium.core.system_trainer.joint import compose_joint_system
        from computronium.ontology import (
            CreditAssignmentConfig,
            DigitalSubstrate,
            EnergyMinimizationDynamics,
            EuclideanUpdate,
            FeedforwardGeometry,
            GeometryConfig,
            NullPlasticity,
            ParameterUpdateConfig,
            RandomProjectionsCredit,
            StateDynamicsConfig,
            SubstrateConfig,
        )

        substrate = DigitalSubstrate(SubstrateConfig.digital())
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(input_dim=8, output_dim=8, hidden_dims=(16,))
        )
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=3, step_size=0.1)
        )
        credit = RandomProjectionsCredit(
            CreditAssignmentConfig.random_projections(feedback_scale=0.0)
        )
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))

        system = compose_joint_system(
            substrate=substrate,
            geometry=geometry,
            dynamics=dynamics,
            plasticity=NullPlasticity(),
            credit=credit,
            update=update,
        )

        import torch

        x = torch.randn(16, 8)
        y = torch.randint(0, 8, (16,))
        metrics = system.train_step(x, y)

        # Inert credit: Δθ = 0 → free_accuracy ≈ 1/8 = 0.125
        free_acc = metrics["free_accuracy"]
        chance = 1.0 / 8

        # free_accuracy at chance (within 3x tolerance for small sample)
        assert abs(free_acc - chance) < 3 * chance, (
            f"free_accuracy {free_acc:.3f} leaked; should be ≈ chance {chance:.3f}"
        )
        # nudged accuracy can be high (leakage) - that's the bug being fixed
        # but we don't assert on it; the point is the CONTRAST
