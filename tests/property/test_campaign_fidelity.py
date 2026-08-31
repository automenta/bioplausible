"""R5b-0 implementation-fidelity gate: per-axis probes over the campaign grid.

Pins the gate's verdicts on the ``joint_grid`` coordinates. A verdict flip —
e.g. after wiring plasticity into the episode pipeline or implementing a
stubbed credit — is the R5b-D lock-④ regression signal, not test breakage.
"""

from __future__ import annotations

import pytest

from computronium.cli.campaign import _get_search_space
from computronium.core.campaign import space_grid
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

    def test_instantaneous_is_single_pass_and_target_blind(self) -> None:
        verdict = check_coordinate_fidelity(
            "digital/recurrent/instantaneous/null/thermodynamic_contrast/euclidean"
        )
        d = [c for c in verdict.checks if c.axis == "dynamics"]
        assert d[0].status == "pass"
        assert "single pass" in d[0].detail
        assert "free ≡ nudged" in d[1].detail


class TestCreditFidelity:
    @pytest.mark.parametrize(
        "credit",
        ["thermodynamic_contrast", "random_projections", "local_goodness"],
    )
    def test_contrastive_credits_dead_under_instantaneous(self, credit: str) -> None:
        """R3.2-class defect, pinned as a gate verdict: a dynamics that
        ignores the target leaves contrastive credits structurally zero."""
        verdict = check_coordinate_fidelity(
            f"digital/feedforward/instantaneous/null/{credit}/euclidean"
        )
        credit_check = _check(verdict, "credit")
        assert credit_check.status == "fail"
        assert "pseudo-gradient" in credit_check.detail

    def test_implemented_credits_signal_under_energy(self) -> None:
        for credit in ("thermodynamic_contrast", "random_projections"):
            verdict = check_coordinate_fidelity(
                f"digital/feedforward/energy_minimization/null/{credit}/euclidean"
            )
            assert _check(verdict, "credit").status == "pass"

    def test_local_goodness_is_an_unimplemented_stub(self) -> None:
        """R3.3 root cause: compute_pseudo_gradient returns [] — the credit
        path is not implemented, not mis-wired."""
        verdict = check_coordinate_fidelity(
            "digital/feedforward/energy_minimization/null/local_goodness/euclidean"
        )
        credit_check = _check(verdict, "credit")
        assert credit_check.status == "fail"
        assert "0 tensors" in credit_check.detail


class TestUpdateFidelity:
    @pytest.mark.parametrize("update", ["euclidean", "spectral_constrained"])
    def test_update_moves_params_given_signal(self, update: str) -> None:
        verdict = check_coordinate_fidelity(
            f"digital/feedforward/energy_minimization/null/thermodynamic_contrast/{update}"
        )
        assert _check(verdict, "update").status == "pass"

    def test_update_blocked_without_signal(self) -> None:
        verdict = check_coordinate_fidelity(
            "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean"
        )
        update_check = _check(verdict, "update")
        assert update_check.status == "blocked"
        assert "no signal" in update_check.detail


class TestPlasticityFidelity:
    def test_null_plasticity_keeps_psi_const(self) -> None:
        verdict = check_coordinate_fidelity(
            "digital/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean"
        )
        m = _check(verdict, "plasticity")
        assert m.status == "pass"
        assert "ψ const" in m.detail

    @pytest.mark.parametrize("plasticity", ["routing", "fast_weights"])
    def test_nonnull_plasticity_is_inert_in_episode_pipeline(
        self, plasticity: str
    ) -> None:
        """M-axis lock: plasticity.step is never invoked by run_train_step,
        so ψ cannot modulate activity or credit in campaign episodes.
        Flipping this verdict = plasticity wired into the pipeline."""
        verdict = check_coordinate_fidelity(
            f"digital/feedforward/energy_minimization/{plasticity}"
            f"/thermodynamic_contrast/euclidean"
        )
        m = _check(verdict, "plasticity")
        assert m.status == "fail"
        assert "never invoked" in m.detail


class TestCapabilityManifest:
    def test_every_grid_coordinate_has_verdicts(self, manifest) -> None:
        assert set(manifest) == set(GRID)
        for verdict in manifest.values():
            axes = {c.axis for c in verdict.checks}
            assert axes == {"dynamics", "credit", "update", "plasticity"}

    def test_passing_set_is_the_validated_subspace(self, manifest) -> None:
        """The capability manifest of the current grid: only the
        energy_minimization x null x {thermodynamic_contrast,
        random_projections} subspace passes all fidelity probes."""
        passing = {c for c, v in manifest.items() if v.passed}
        expected = {
            c
            for c in GRID
            if "/energy_minimization/" in c
            and "/null/" in c
            and ("/thermodynamic_contrast/" in c or "/random_projections/" in c)
        }
        assert passing == expected
        assert len(passing) == 8

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
        # With only one value per dynamics/plasticity axis in the passing
        # subspace, no D/M minimal pairs can exist in the filtered output.
        attributed_axes = {a.axis for a in result.attributions}
        assert "dynamics" not in attributed_axes
        assert "plasticity" not in attributed_axes
        assert attributed_axes <= {"geometry", "credit", "update"}


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
