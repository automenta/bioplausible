"""R5b-D discovery locks (TODO8): pre-registered claims enforced as code.

Locks ① (winner replication) and ② (attribution rank) are exercised at
data level with engineered ground truth — deterministic, no device
dependence, no flake surface. Lock ③ runs a real GPU replay of an episode
from its (seed, campaign_id, iteration) seed when CUDA is available. Lock
④ re-runs fidelity probes over representative grid coordinates. The live
R5b-B artifacts are checked against the registered null claim when
present; if an instrument repair or a re-commissioned campaign flips that
lock, the discovery state must be re-evaluated — not silently preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from computronium.core.campaign.discovery import (
    DiscoverySpec,
    verify_attribution_rank,
    verify_fidelity_standing,
    verify_replay,
    verify_winner_replication,
)
from computronium.core.campaign.fidelity import fidelity_manifest
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.resources import ResourceUsage

R5B_B_EPISODES = (
    Path(__file__).resolve().parents[2]
    / "autoscientist_campaigns/r5b_b/records/episodes.json"
)

NULL_COORD = (
    "digital/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean"
)
FAST_COORD = NULL_COORD.replace("null", "fast_weights")
CREDIT_COORD = NULL_COORD.replace("thermodynamic_contrast", "local_goodness")

FAST_SPEC = DiscoverySpec(
    name="engineered-fast-weights",
    axis="plasticity",
    from_value="null",
    to_value="fast_weights",
    min_delta=0.15,
)
CREDIT_SPEC = DiscoverySpec(
    name="engineered-local-goodness",
    axis="credit",
    from_value="thermodynamic_contrast",
    to_value="local_goodness",
    min_delta=0.15,
)
NULL_SPEC = DiscoverySpec(name="registered-null", claim="null", min_delta=0.15)


def record(
    coordinate: str, *, task: str = "synthetic", seed: int = 0, accuracy: float = 0.5
) -> FrontierRecord:
    return FrontierRecord(
        coordinate=coordinate,
        task_name=task,
        task_loss=1.0 - accuracy,
        task_accuracy=accuracy,
        adaptation_time=1,
        rho_jacobian=0.9,
        lyapunov_local=0.0,
        settling_time=1.0,
        basin_stability=1.0,
        resources=ResourceUsage(),
        seed=seed,
    )


def strata_records(
    fast_accs: tuple[float, float] = (0.5, 0.5),
    credit_accs: tuple[float, float] = (0.4, 0.4),
    null_acc: float = 0.3,
) -> list[FrontierRecord]:
    """Two seeds x two families with engineered per-stratum accuracies.

    Null→fast_weights (plasticity) and contrast→local_goodness (credit)
    are the only minimal pairs; their deltas per stratum are set by
    ``fast_accs``/``credit_accs`` (indexed by seed; both families share a
    stratum's value so seeds carry the signal).
    """
    records: list[FrontierRecord] = []
    for seed in range(2):
        for task in ("synthetic", "parity"):
            records.append(record(NULL_COORD, task=task, seed=seed, accuracy=null_acc))
            records.append(
                record(FAST_COORD, task=task, seed=seed, accuracy=fast_accs[seed])
            )
            records.append(
                record(CREDIT_COORD, task=task, seed=seed, accuracy=credit_accs[seed])
            )
    return records


class TestWinnerReplication:
    def test_effect_replicates_across_strata(self) -> None:
        verdict = verify_winner_replication(strata_records(), FAST_SPEC)
        assert verdict.holds, verdict.detail

    def test_effect_fails_when_one_stratum_flips(self) -> None:
        records = strata_records(fast_accs=(0.5, 0.35))
        verdict = verify_winner_replication(records, FAST_SPEC)
        assert not verdict.holds
        assert "min_delta" in verdict.detail

    def test_effect_fails_when_transition_absent(self) -> None:
        records = [r for r in strata_records() if r.coordinate != FAST_COORD]
        verdict = verify_winner_replication(records, FAST_SPEC)
        assert not verdict.holds
        assert "absent" in verdict.detail

    def test_null_holds_while_unstable(self) -> None:
        """Seed 0's top is plasticity, seed 1's is credit — nothing stable."""
        records = strata_records(fast_accs=(0.5, 0.42), credit_accs=(0.4, 0.55))
        verdict = verify_winner_replication(records, NULL_SPEC)
        assert verdict.holds, verdict.detail

    def test_null_fails_when_effect_is_stable(self) -> None:
        verdict = verify_winner_replication(strata_records(), NULL_SPEC)
        assert not verdict.holds
        assert "plasticity" in verdict.detail


class TestAttributionRank:
    def test_claim_ranks_first_pooled_and_per_stratum(self) -> None:
        verdict = verify_attribution_rank(strata_records(), FAST_SPEC)
        assert verdict.holds, verdict.detail

    def test_claim_fails_when_not_pooled_top(self) -> None:
        verdict = verify_attribution_rank(strata_records(), CREDIT_SPEC)
        assert not verdict.holds
        assert "pooled top" in verdict.detail

    def test_null_holds_when_strata_disagree(self) -> None:
        records = strata_records(fast_accs=(0.5, 0.42), credit_accs=(0.4, 0.55))
        verdict = verify_attribution_rank(records, NULL_SPEC)
        assert verdict.holds, verdict.detail

    def test_null_fails_when_strata_agree(self) -> None:
        verdict = verify_attribution_rank(strata_records(), NULL_SPEC)
        assert not verdict.holds
        assert "register an effect claim" in verdict.detail

    def test_manifest_filters_defective_records(self) -> None:
        """Lock ② honors the fidelity filter: a quarantined fast_weights arm
        vanishes from attribution; the remaining top (credit, +0.10) is
        stable but below the claimable threshold, so the null claim holds."""

        class _Failing:
            passed = False

        class _Passing:
            passed = True

        manifest = {
            FAST_COORD: _Failing(),
            NULL_COORD: _Passing(),
            CREDIT_COORD: _Passing(),
        }
        verdict = verify_attribution_rank(
            strata_records(),
            NULL_SPEC,
            manifest=manifest,  # type: ignore[arg-type]
        )
        assert verdict.holds, verdict.detail


class TestReplayLock:
    COORDS = (
        NULL_COORD,
        "digital/recurrent/instantaneous/null/random_projections/euclidean",
    )

    @pytest.mark.parametrize("coordinate", COORDS)
    def test_episode_replays_from_seed_on_gpu(self, coordinate: str) -> None:
        pytest.importorskip("torch.cuda")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        verdict = verify_replay(
            coordinate,
            seed=7,
            campaign_id="lock_replay",
            iteration=3,
            device="cuda",
        )
        assert verdict.holds, verdict.detail


class TestFidelityStanding:
    COORDS = (
        NULL_COORD,
        "digital/recurrent/energy_minimization/fast_weights/random_projections"
        "/euclidean",
        "digital/feedforward/instantaneous/null/random_projections/euclidean",
        "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
    )

    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        return fidelity_manifest(self.COORDS)

    def test_passing_coordinates_hold(self, manifest: dict) -> None:
        verdict = verify_fidelity_standing(manifest, self.COORDS[:3])
        assert verdict.holds, verdict.detail

    def test_quarantined_coordinate_fails(self, manifest: dict) -> None:
        """The R3.9-fenced pairing fails the winner lock with it."""
        verdict = verify_fidelity_standing(manifest, self.COORDS)
        assert not verdict.holds
        assert "thermodynamic_contrast" in verdict.detail

    def test_missing_verdict_fails(self, manifest: dict) -> None:
        verdict = verify_fidelity_standing(manifest, (*self.COORDS[:2], "digital/x"))
        assert not verdict.holds
        assert "digital/x" in verdict.detail


class TestRegisteredNullResult:
    """The R5b-B/C registered state, locked against the live artifacts.

    Claim: at smoke scale no axis transition is stratified-stable at the
    0.05 accuracy floor (the claimable-effect threshold for toy campaigns).
    A flip here is the R5b-D regression signal: either a discovery landed
    (register an effect spec) or an instrument repair changed behavior
    (re-run the evidence chain) — never silently preserve the claim.
    """

    MIN_CLAIMABLE_DELTA = 0.05

    @pytest.fixture(scope="class")
    def records(self) -> list[FrontierRecord]:
        if not R5B_B_EPISODES.exists():
            pytest.skip(
                f"{R5B_B_EPISODES} not found — commission the R5b-B campaign first"
            )
        payload = json.loads(R5B_B_EPISODES.read_text())
        return [FrontierRecord.from_dict(entry) for entry in payload]

    def test_winner_replication_null_holds(self, records) -> None:
        spec = DiscoverySpec(
            name="r5b-b-registered-null",
            claim="null",
            min_delta=self.MIN_CLAIMABLE_DELTA,
        )
        verdict = verify_winner_replication(records, spec)
        assert verdict.holds, verdict.detail

    def test_attribution_rank_null_holds(self, records) -> None:
        spec = DiscoverySpec(
            name="r5b-b-registered-null",
            claim="null",
            min_delta=self.MIN_CLAIMABLE_DELTA,
        )
        verdict = verify_attribution_rank(records, spec)
        assert verdict.holds, verdict.detail
