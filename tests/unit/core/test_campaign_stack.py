"""P5 campaign infrastructure: schema freeze, CampaignStack, replication,
counterfactual attribution, and Pareto frontier semantics."""

from __future__ import annotations

import random
import sqlite3
import subprocess
from pathlib import Path

import pytest

from computronium.analysis.counterfactual import (
    attribute_axis_effects,
    counterfactual_pairs,
    what_if,
)
from computronium.core.campaign import (
    SCHEMA_VERSION,
    CampaignStack,
    CampaignStore,
    FrontierRecord,
    ParetoFrontier,
    ResourceUsage,
    SchemaVersionError,
    episode_batch,
    pareto_frontier,
    replication_manifest,
    task_family,
    verify_replication,
)
from computronium.core.campaign.replication import unreplicated

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_record(  # ruff: ignore[too-many-arguments] - record builder with independent defaults
    coordinate: str,
    *,
    task: str = "mnist",
    seed: int = 0,
    accuracy: float = 0.5,
    loss: float = 1.0,
    energy: float = 1.0,
    rho: float = 0.9,
) -> FrontierRecord:
    return FrontierRecord(
        coordinate=coordinate,
        task_name=task,
        task_loss=loss,
        task_accuracy=accuracy,
        adaptation_time=1,
        rho_jacobian=rho,
        lyapunov_local=0.0,
        settling_time=1.0,
        basin_stability=1.0,
        resources=ResourceUsage(energy=energy),
        seed=seed,
    )


def _bad_axis_sampler(_rng, _iteration: int, _experiment: int) -> str:
    return "digital/bad_axis/instantaneous/null/thermodynamic_contrast/euclidean"


# --- Schema freeze -----------------------------------------------------------


class TestSchemaFreeze:
    def test_fresh_db_stamped_with_current_version(self, tmp_path: Path) -> None:
        store = CampaignStore(tmp_path / "campaign.db")
        assert store.schema_version == SCHEMA_VERSION

    def test_future_schema_rejected(self, tmp_path: Path) -> None:
        db = tmp_path / "campaign.db"
        db.write_text("")
        with sqlite3.connect(db) as conn:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(SchemaVersionError):
            CampaignStore(db)

    def test_legacy_pre_freeze_db_grandfathered(self, tmp_path: Path) -> None:
        """A pre-freeze DB has v1 tables but user_version=0; opening stamps it."""
        db = tmp_path / "campaign.db"
        legacy = CampaignStore(db)
        legacy.create_campaign()
        with sqlite3.connect(db) as conn:
            conn.execute("PRAGMA user_version = 0")

        reopened = CampaignStore(db)
        assert reopened.schema_version == SCHEMA_VERSION
        assert reopened.list_campaigns()[0].branch_name == "main"

    def test_migration_hook_appends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A v1 DB upgrades forward-only through appended migrations (v2 sim)."""
        db = tmp_path / "campaign.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE campaigns (campaign_id TEXT PRIMARY KEY)")
            conn.execute("PRAGMA user_version = 1")
            conn.execute("CREATE TABLE legacy_flags (k TEXT)")

        from computronium.core.campaign import campaign_store

        monkeypatch.setattr(campaign_store, "SCHEMA_VERSION", 2)
        monkeypatch.setattr(
            campaign_store,
            "MIGRATIONS",
            (lambda conn: None, lambda conn: conn.execute("DROP TABLE legacy_flags")),
        )
        store = CampaignStore(db)
        assert store.schema_version == 2
        with sqlite3.connect(db) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "legacy_flags" not in tables
        assert "campaigns" in tables  # pre-existing v1 tables preserved


# --- CampaignStack -----------------------------------------------------------


class TestCampaignStack:
    def test_run_campaign_records_episodes(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=7)
        result = stack.run_campaign(
            iterations=2, experiments_per_iter=2, campaign_id="camp_test"
        )
        assert result.records
        assert all(o.status == "recorded" for o in result.outcomes)
        assert result.iterations_run == (1, 2)
        episodes = stack.store.get_episodes("camp_test")
        assert len(episodes) == len(result.outcomes)
        campaign_state = stack.store.get_campaign("camp_test")
        assert campaign_state is not None
        assert campaign_state.iteration == 2

    def test_coordinate_stream_deterministic(self, tmp_path: Path) -> None:
        coords = []
        for run_dir in (tmp_path / "a", tmp_path / "b"):
            stack = CampaignStack(run_dir, seed=3)
            result = stack.run_campaign(
                iterations=1,
                experiments_per_iter=3,
                campaign_id="camp_fixed",
            )
            coords.append([o.coordinate for o in result.outcomes])
        assert coords[0] == coords[1]

    def test_resume_continues_iteration(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=5)
        first = stack.run_campaign(iterations=2, experiments_per_iter=2)
        resumed = stack.run_campaign(iterations=1, experiments_per_iter=2, resume=True)
        assert resumed.iterations_run == (3,)
        assert resumed.campaign_id == first.campaign_id
        total = len(stack.store.get_episodes(first.campaign_id))
        assert total == len(first.outcomes) + len(resumed.outcomes)

    def test_resume_redo_replays_lost_episode(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=5, checkpoint_interval=1)
        result = stack.run_campaign(iterations=2, experiments_per_iter=2)
        campaign_id = result.campaign_id

        # Simulate a crash after the iteration counter advanced but before the
        # last episode row survived: delete it and resume.
        episodes = stack.store.get_episodes(campaign_id)
        lost = episodes[-1]
        with sqlite3.connect(stack.store.db_path) as conn:
            conn.execute(
                "DELETE FROM episodes WHERE campaign_id = ? AND iteration = ?",
                (campaign_id, lost.iteration),
            )

        stack.run_campaign(iterations=1, experiments_per_iter=1, resume=True)
        iterations = {ep.iteration for ep in stack.store.get_episodes(campaign_id)}
        assert lost.iteration in iterations

    def test_yaml_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=1)
        result = stack.run_campaign(iterations=1, experiments_per_iter=1)
        assert result.yaml_checkpoint is not None
        state, history = stack.load_yaml_checkpoint(result.yaml_checkpoint)
        assert state.campaign_id == result.campaign_id
        assert len(history) == 1

    def test_unsupported_coordinates_skipped(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=0)
        result = stack.run_campaign(
            iterations=1,
            experiments_per_iter=2,
            sampler=_bad_axis_sampler,
        )
        assert all(o.status == "unsupported" for o in result.outcomes)
        assert result.records == ()

    def test_dry_run_proposes_without_executing(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=0)
        result = stack.run_campaign(iterations=1, experiments_per_iter=3, dry_run=True)
        assert len(result.outcomes) == 3
        assert all(o.status == "dry_run" for o in result.outcomes)
        assert result.yaml_checkpoint is not None


# --- Deterministic replay (improvement-10/11 locks) ----------------------------


class TestDeterministicReplay:
    """Resume skips recorded episodes; reconstruction replays metrics."""

    COORDS = (
        "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
        "digital/recurrent/instantaneous/null/thermodynamic_contrast/euclidean",
    )

    @staticmethod
    def _slot_sampler(coordinates: tuple[str, ...]):
        def sample(_rng, _iteration: int, experiment: int) -> str:
            return coordinates[experiment % len(coordinates)]

        return sample

    def test_resume_skips_already_recorded_episodes(self, tmp_path: Path) -> None:
        """Crash mid-iteration: durable episodes are skipped, not duplicated."""
        stack = CampaignStack(tmp_path, seed=5, checkpoint_interval=1)
        stack.run_campaign(
            iterations=1,
            experiments_per_iter=2,
            campaign_id="camp_dedup",
            sampler=self._slot_sampler(self.COORDS),
        )
        # Simulate a crash after one more durable episode whose iteration
        # counter never advanced (killed between add_episode and update_iteration).
        stack.store.add_episode(
            campaign_id="camp_dedup",
            branch_name="main",
            iteration=2,
            coordinate=self.COORDS[0],
            task_name="synthetic",
            frontier_record=make_record(self.COORDS[0], seed=5),
        )
        resumed = stack.run_campaign(
            iterations=2,
            experiments_per_iter=2,
            resume=True,
            sampler=self._slot_sampler(self.COORDS),
        )
        assert any(o.status == "already_recorded" for o in resumed.outcomes)
        episodes = stack.store.get_episodes("camp_dedup")
        keys = [(ep.iteration, ep.coordinate, ep.task_name) for ep in episodes]
        assert len(keys) == len(set(keys))

    def test_rebuild_replays_identical_metrics(self, tmp_path: Path) -> None:
        """Same (seed, campaign, iteration, coordinate) ⇒ same θ ⇒ same metrics."""
        runs = []
        for run_dir in (tmp_path / "a", tmp_path / "b"):
            stack = CampaignStack(run_dir, seed=3)
            result = stack.run_campaign(
                iterations=1,
                experiments_per_iter=2,
                campaign_id="camp_replay",
            )
            runs.append([(r.task_loss, r.task_accuracy) for r in result.records])
        assert runs[0] == runs[1]

    def test_records_carry_campaign_seed(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=9)
        result = stack.run_campaign(iterations=1, experiments_per_iter=1)
        assert result.records
        assert all(r.seed == 9 for r in result.records)

    def test_unknown_task_family_rejected(self, tmp_path: Path) -> None:
        """Real-dataset labels must not masquerade as smoke-batch families."""
        stack = CampaignStack(tmp_path, seed=0)
        with pytest.raises(ValueError, match="task family"):
            stack.run_campaign(iterations=1, experiments_per_iter=1, tasks=("mnist",))

    def test_task_families_produce_distinct_batches(self) -> None:
        import torch

        x_syn, y_syn = episode_batch(0, task_name="synthetic")
        x_par, y_par = episode_batch(0, task_name="parity")
        assert not torch.equal(x_syn, x_par)
        assert y_par.shape == y_syn.shape
        assert y_syn.shape[0] == x_syn.shape[0]

    def test_grid_sampler_is_slot_deterministic(self) -> None:
        from computronium.core.campaign import grid_sampler, space_grid

        space = {
            "substrates": ["digital"],
            "geometries": ["feedforward", "recurrent"],
            "dynamics": ["instantaneous"],
            "plasticity": ["null"],
            "credits": ["thermodynamic_contrast"],
            "updates": ["euclidean"],
        }
        grid = space_grid(space)
        assert len(grid) == 2
        sample = grid_sampler(grid, experiments_per_iter=2)
        rng_a, rng_b = random.Random(0), random.Random(1)
        first = [sample(rng_a, 1, e) for e in range(2)]
        again = [sample(rng_b, 1, e) for e in range(2)]
        assert first == again
        assert set(first) == set(grid)

    def test_grid_layout_replicates_both_families(self, tmp_path: Path) -> None:
        """R5b-B parity bug pinned: an even grid cycle under the old
        slot-parity task rotation left every coordinate on a single family
        per seed (replication gate 0/48). Visit-count alternation must give
        each coordinate both families per seed and certify replication."""
        from computronium.core.campaign import grid_sampler, space_grid

        space = {
            "substrates": ["digital"],
            "geometries": ["feedforward", "recurrent"],
            "dynamics": ["instantaneous"],
            "plasticity": ["null"],
            "credits": ["random_projections"],
            "updates": ["euclidean"],
        }
        sampler = grid_sampler(space_grid(space), experiments_per_iter=2)
        records = []
        for seed in (0, 1):
            stack = CampaignStack(tmp_path / f"s{seed}", seed=seed)
            stack.run_campaign(
                iterations=2,
                experiments_per_iter=2,
                tasks=("synthetic", "parity"),
                campaign_id=f"camp_families_s{seed}",
                sampler=sampler,
            )
            records.extend(stack.frontier_records())
        by_coordinate: dict[str, set[str]] = {}
        for record in records:
            by_coordinate.setdefault(record.coordinate, set()).add(record.task_name)
        assert by_coordinate
        assert all(
            families == {"synthetic", "parity"} for families in by_coordinate.values()
        )
        assert all(
            r.replicated for r in replication_manifest(records, min_seeds=2).values()
        )


# --- Pareto over loss, stability, resources -----------------------------------


class TestParetoSemantics:
    def test_lower_loss_and_higher_stability_dominate(self) -> None:
        better = make_record(
            "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
            loss=0.5,
            rho=0.5,
            energy=0.5,
        )
        worse = make_record(
            "digital/recurrent/instantaneous/null/thermodynamic_contrast/euclidean",
            loss=1.5,
            rho=0.9,
            energy=2.0,
        )
        pf = pareto_frontier(
            [better, worse],
            ("task_loss", "stability_score", "energy"),
            (True, True, True),
        )
        assert [r.coordinate for r in pf.frontier] == [better.coordinate]
        assert worse in pf.dominated

    def test_tradeoff_keeps_both_on_frontier(self) -> None:
        stable = make_record(
            "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
            loss=0.4,
            rho=0.1,
            energy=5.0,
        )
        cheap = make_record(
            "digital/recurrent/instantaneous/null/thermodynamic_contrast/euclidean",
            loss=1.2,
            rho=0.9,
            energy=0.1,
        )
        pf = pareto_frontier(
            [stable, cheap],
            ("task_loss", "stability_score", "energy"),
            (True, True, True),
        )
        assert len(pf.frontier) == 2

    def test_hypervolume_positive_for_dominated_reference(self) -> None:
        records = [
            make_record(
                "digital/feedforward/instantaneous/null/"
                "thermodynamic_contrast/euclidean",
                loss=0.3,
                rho=0.2,
                energy=0.3,
            ),
            make_record(
                "digital/recurrent/instantaneous/null/thermodynamic_contrast/euclidean",
                loss=0.8,
                rho=0.6,
                energy=0.8,
            ),
        ]
        pf = pareto_frontier(
            records,
            ("task_loss", "stability_score"),
            (True, True),
            reference_point=(-2.0, -2.0),
        )
        assert isinstance(pf, ParetoFrontier)
        assert pf.hypervolume > 0


# --- Replication gate ---------------------------------------------------------


class TestReplicationGate:
    COORD = "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean"

    def test_replicated_with_five_seeds_two_families(self) -> None:
        records = [
            make_record(self.COORD, seed=s, task=task)
            for s in range(5)
            for task in ("mnist", "cartpole")
        ]
        report = verify_replication(records, coordinate=self.COORD)
        assert report.replicated
        assert report.seeds == (0, 1, 2, 3, 4)
        assert report.task_families == ("rl", "vision")

    def test_unmet_requirements_reported(self) -> None:
        records = [make_record(self.COORD, seed=s) for s in range(3)]
        report = verify_replication(records, coordinate=self.COORD)
        assert not report.replicated
        assert any("task families" in u for u in report.unmet())
        assert any("seeds" in u for u in report.unmet())

    def test_task_family_mapping(self) -> None:
        assert task_family("cartpole") == "rl"
        assert task_family("tiny_shakespeare") == "lm"
        assert task_family("custom_task") == "custom_task"

    def test_manifest_covers_all_coordinates(self) -> None:
        other = self.COORD.replace("feedforward", "recurrent")
        records = [make_record(self.COORD), make_record(other)]
        manifest = replication_manifest(records)
        assert set(manifest) == {self.COORD, other}

    def test_unreplicated_worst_first(self) -> None:
        good = [
            make_record(self.COORD, seed=s, task=task)
            for s in range(5)
            for task in ("mnist", "cartpole")
        ]
        bad = [make_record(self.COORD.replace("euclidean", "natural_gradient"))]
        failing = unreplicated(good + bad)
        assert len(failing) == 1
        assert "natural_gradient" in failing[0].coordinate


# --- Counterfactual attribution -----------------------------------------------


A = "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean"
B = "digital/feedforward/instantaneous/routing/thermodynamic_contrast/euclidean"
C = "digital/recurrent/instantaneous/null/thermodynamic_contrast/euclidean"


class TestCounterfactualAttribution:
    def test_minimal_pair_attribution(self) -> None:
        records = [
            make_record(A, accuracy=0.5),
            make_record(B, accuracy=0.7),
        ]
        pairs = counterfactual_pairs(records)
        assert len(pairs) == 1
        assert pairs[0].axis == "plasticity"
        assert pairs[0].from_value == "null"
        assert pairs[0].to_value == "routing"
        assert pairs[0].delta == pytest.approx(0.2)

    def test_multi_axis_differences_ignored(self) -> None:
        records = [make_record(A), make_record(B), make_record(C)]
        pairs = counterfactual_pairs(records)
        # A/B differ only in plasticity; A/C only in geometry; B/C in two axes.
        assert {(p.axis, p.from_value, p.to_value) for p in pairs} == {
            ("plasticity", "null", "routing"),
            ("geometry", "feedforward", "recurrent"),
        }

    def test_effects_sorted_by_magnitude(self) -> None:
        records = [
            make_record(A, accuracy=0.5),
            make_record(B, accuracy=0.8),  # plasticity effect +0.3
            make_record(C, accuracy=0.55),
        ]
        effects = attribute_axis_effects(records)
        assert effects[0].axis == "plasticity"
        assert effects[0].mean_delta == pytest.approx(0.3)
        assert effects[0].to_dict()["n_pairs"] == 1

    def test_what_if_forward_and_reverse(self) -> None:
        records = [
            make_record(A, accuracy=0.5),
            make_record(B, accuracy=0.7),
        ]
        forward = what_if(records, A, "plasticity", "routing")
        reverse = what_if(records, B, "plasticity", "null")
        assert forward == pytest.approx(0.2)
        assert reverse == pytest.approx(-0.2)

    def test_what_if_without_data_returns_none(self) -> None:
        records = [make_record(A)]
        assert what_if(records, A, "plasticity", "fast_weights") is None

    def test_invalid_inputs_rejected(self) -> None:
        records = [make_record(A)]
        with pytest.raises(ValueError, match="axis"):
            what_if(records, A, "not_an_axis", "routing")
        with pytest.raises(ValueError, match="coordinate"):
            what_if(records, "bad/coordinate", "plasticity", "routing")


# --- CLI end-to-end ------------------------------------------------------------


class TestCampaignCLI:
    def test_campaign_run_end_to_end(self, tmp_path: Path) -> None:
        """`comp campaign run` produces a queryable store via CampaignStack."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "comp",
                "campaign",
                "run",
                "--space",
                "joint_smoke",
                "--objective",
                "accuracy",
                "--iterations",
                "1",
                "--experiments-per-iter",
                "2",
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "completed" in result.stdout

        db = tmp_path / "campaign.db"
        store = CampaignStore(db)
        assert store.schema_version == SCHEMA_VERSION
        campaigns = store.list_campaigns()
        assert len(campaigns) == 1
        assert store.get_episodes(campaigns[0].campaign_id)
