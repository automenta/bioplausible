"""Z3 suite-level ψ engagement gate (R8.1) + planted-ψ positive control (R8.2).

Upgrades the R7 pipeline-level lock (test_psi_engagement.py) to the Z3 suite
path itself: every gate item is asserted on ``evaluate_z3`` output at the
registered scale. The ψ-disabled run (``MetaRecipe(feedback=False)``) doubles
as the engineered-broken variant — the gate must flag it (imp-50 class) — and
the engaged-vs-disabled contrast is the planted-ψ instrument self-test: with
identical θ (forced-selection warm-up is ψ-independent; phase 2 freezes θ),
task stream, seed, and budget, the only causal difference is ψ stepping, so
any metric difference is a ψ-mediated effect the instrument must see.
"""

import pytest
import torch

from computronium.experiments.joint.z3_fixed_weights import (
    TASK_CHANCE,
    MetaRecipe,
    evaluate_z3,
)

COORDINATE = (
    "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean"
)

_REGISTERED_SCALE: dict = {
    "meta_train_epochs": 50,
    "eval_epochs_per_task": 20,
    "batch_size": 64,
    "seq_len": 10,
    "input_dim": 32,
    "probe_batches": 4,
    "device": "cpu",
    "seed": 0,
}


@pytest.fixture(scope="module")
def engaged() -> dict:
    return evaluate_z3(COORDINATE, with_baselines=False, **_REGISTERED_SCALE)


@pytest.fixture(scope="module")
def psi_disabled() -> dict:
    return evaluate_z3(
        COORDINATE,
        with_baselines=False,
        recipe=MetaRecipe(feedback=False),
        **_REGISTERED_SCALE,
    )


class TestZ3SuiteEngagementGate:
    """R8.1: the suite fails unless every engagement item holds."""

    def test_theta_exact_invariance(self, engaged: dict) -> None:
        assert not engaged["theta_change"]  # exact zero: bitwise invariance
        assert engaged["theta_invariant"] is True

    def test_psi_non_constant_in_every_task(self, engaged: dict) -> None:
        for name, row in engaged["tasks"].items():
            assert max(row["psi_history"]["delta_norm"]) > 0.0, name

    def test_psi_task_conditioned(self, engaged: dict) -> None:
        finals = [torch.tensor(row["final_psi"]) for row in engaged["tasks"].values()]
        pairwise = [
            float((a - b).abs().max())
            for i, a in enumerate(finals)
            for b in finals[i + 1 :]
        ]
        assert min(pairwise) > 0.0

    def test_gates_respond_to_psi(self, engaged: dict) -> None:
        assert engaged["psi_gate_response"] > 0.0

    def test_frozen_psi_control_changes_metrics(self, engaged: dict) -> None:
        gate = engaged["psi_gate"]
        assert gate["items"]["frozen_psi_control_changes_metrics"] is True
        assert not gate["detail"]["control_theta_change"]  # exact zero

    def test_probe_accuracy_above_chance_all_tasks(self, engaged: dict) -> None:
        assert engaged["psi_gate"]["items"]["probe_above_chance"] == dict.fromkeys(
            TASK_CHANCE, True
        )

    def test_gate_passes(self, engaged: dict) -> None:
        assert engaged["psi_gate"]["passed"] is True, engaged["psi_gate"]["failed"]


class TestZ3PositiveControl:
    """R8.2: the planted ψ-mediated effect is detected through the suite path."""

    def test_psi_disabled_arm_is_flagged_by_the_gate(self, psi_disabled: dict) -> None:
        items = psi_disabled["psi_gate"]["items"]
        assert items["psi_non_constant"] is False
        assert items["psi_task_conditioned"] is False
        assert items["frozen_psi_control_changes_metrics"] is False
        assert psi_disabled["psi_gate"]["passed"] is False

    def test_psi_disabled_arm_moves_no_psi(self, psi_disabled: dict) -> None:
        for row in psi_disabled["tasks"].values():
            # ψ stays bitwise zero: no norm ever becomes nonzero.
            assert not any(row["psi_history"]["norm"])
            assert not any(row["psi_history"]["delta_norm"])

    def test_planted_effect_detected_in_metrics(
        self, engaged: dict, psi_disabled: dict
    ) -> None:
        engaged_acc = {n: r["accuracy"] for n, r in engaged["tasks"].items()}
        disabled_acc = {n: r["accuracy"] for n, r in psi_disabled["tasks"].items()}
        curves_differ = any(
            e["accuracy_curve"] != p["accuracy_curve"]
            for e, p in zip(engaged["tasks"].values(), psi_disabled["tasks"].values())
        )
        assert engaged_acc != disabled_acc or curves_differ

    def test_planted_effect_direction_favors_psi(
        self, engaged: dict, psi_disabled: dict
    ) -> None:
        for name, row in engaged["tasks"].items():
            assert row["accuracy"] >= psi_disabled["tasks"][name]["accuracy"], name
        best_gap = max(
            row["accuracy"] - psi_disabled["tasks"][name]["accuracy"]
            for name, row in engaged["tasks"].items()
        )
        assert best_gap > 0.2

    def test_arms_share_theta(self, engaged: dict, psi_disabled: dict) -> None:
        assert engaged["theta_sha256"] == psi_disabled["theta_sha256"]


class TestZ3RetentionPivot:
    """R9.1: A→B→A with θ frozen and ψ snapshot/restored — switching
    prevents forgetting. Instrument items (exact θ invariance, lossless ψ
    restore) must hold at any scale, including on the ψ-disabled variant;
    capability items (acquisition, above-chance retention, floor contrast)
    hold at registered scale and are the ψ-carrier evidence."""

    def test_retention_theta_exact_invariance(self, engaged: dict) -> None:
        items = engaged["retention_gate"]["items"]
        assert items["theta_exact_invariant"] is True
        assert not engaged["retention"]["restored"]["theta_change"]

    def test_psi_restore_is_lossless_on_both_arms(
        self, engaged: dict, psi_disabled: dict
    ) -> None:
        for run in (engaged, psi_disabled):
            assert run["retention_gate"]["items"]["psi_restore_reproduces_stage_a"]

    def test_retention_gate_passes(self, engaged: dict) -> None:
        assert engaged["retention_gate"]["passed"] is True, engaged["retention_gate"][
            "failed"
        ]

    def test_restored_psi_beats_floor_and_b_psi(self, engaged: dict) -> None:
        retention = engaged["retention"]
        assert (
            retention["restored"]["task_a_accuracy"]
            > retention["stage_b"]["task_a_accuracy_under_psi_b"]
        )
        assert retention["forgetting_via_psi_switch"] > 0.0

    def test_disabled_arm_flags_the_carrier_item(self, psi_disabled: dict) -> None:
        gate = psi_disabled["retention_gate"]
        # ψ never moves on the disabled arm, so ψ_A == ψ_B: the ψ-state
        # task-conditioning item must fail (probe-the-probe), while the
        # lossless-restore instrument item still holds.
        assert gate["items"]["psi_state_task_conditioned"] is False
        assert gate["passed"] is False

    def test_enabled_arm_carries_tasks_in_psi(self, engaged: dict) -> None:
        assert engaged["retention_gate"]["items"]["psi_state_task_conditioned"]

    def test_stage_b_interferes_with_task_a(self, engaged: dict) -> None:
        retention = engaged["retention"]
        under_b = retention["stage_b"]["task_a_accuracy_under_psi_b"]
        assert under_b < retention["stage_a"]["accuracy"]
