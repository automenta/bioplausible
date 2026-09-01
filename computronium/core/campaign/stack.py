"""CampaignStack: composable facade over the campaign infrastructure.

Consolidates persistence (CampaignStore), fault tolerance (CheckpointManager),
episode evaluation, Pareto frontier analysis, counterfactual attribution, and
the replication gate behind one ``run_campaign`` entry point. The CLI and
research runners share this engine; ``run_campaign`` is deterministic per
(campaign_id, iteration): coordinate proposals, episode batches, and θ
construction all replay identically across crash/resume, and resume skips
already-recorded (iteration, coordinate, task) rows instead of duplicating
them.
"""

from __future__ import annotations

import itertools
import random
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from computronium.analysis.counterfactual import AxisAttribution, attribute_axis_effects
from computronium.core.campaign.campaign_store import (
    CampaignState,
    CampaignStore,
    EpisodeRecord,
)
from computronium.core.campaign.checkpoint import CheckpointManager, JointCheckpoint
from computronium.core.campaign.evaluation import (
    DEFAULT_GUARD_TAU,
    DEFAULT_INPUT_DIM,
    DEFAULT_NUM_CLASSES,
    GuardKillError,
    UnsupportedCoordinateError,
    build_coordinate_system,
    episode_batch,
    episode_seed,
    evaluate_episode,
)
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.core.campaign.pareto import ParetoFrontier, pareto_frontier
from computronium.core.campaign.replication import (
    DEFAULT_MIN_FAMILIES,
    DEFAULT_MIN_SEEDS,
    ReplicationReport,
    replication_manifest,
)
from computronium.core.logging import get_logger

if TYPE_CHECKING:
    import torch

    from computronium.core.campaign.report import DiscoveryReport
    from computronium.core.system_trainer import JointSystem

logger = get_logger()


class CoordinateSampler(Protocol):
    """Stateless coordinate proposal for one episode slot.

    Receives the per-iteration RNG and the slot's (iteration, experiment)
    coordinates; implementations must derive proposals only from these so a
    resumed campaign replays identical proposals without persistent state.
    Slots are positional-only: implementations may rename them freely.
    """

    def __call__(
        self, rng: random.Random, iteration: int, experiment: int, /
    ) -> str: ...


def _space_sampler(space: dict) -> CoordinateSampler:
    """Bind a search-space table into a seeded coordinate sampler."""

    axes = ("substrates", "geometries", "dynamics", "plasticity", "credits", "updates")

    def sample(rng: random.Random, _iteration: int, _experiment: int) -> str:
        return "/".join(rng.choice(space[axis]) for axis in axes)

    return sample


def space_grid(space: dict) -> list[str]:
    """Enumerate a search-space table into its full coordinate grid."""
    # Custom grid for R5b-B fidelity-passing coordinates
    if "_custom_grid" in space:
        return list(space["_custom_grid"])
    axes = ("substrates", "geometries", "dynamics", "plasticity", "credits", "updates")
    return [
        "/".join(parts) for parts in itertools.product(*(space[axis] for axis in axes))
    ]


def grid_sampler(grid: Sequence[str], experiments_per_iter: int) -> CoordinateSampler:
    """Round-robin traversal of a fixed coordinate grid.

    Stateless in (iteration, experiment): every seed-campaign sharing the
    grid evaluates the same coordinate at the same slot, and resume replays
    identical proposals — the shape a controlled multi-seed ablation needs.
    """

    coords = tuple(dict.fromkeys(grid))
    if not coords:
        raise ValueError(  # ruff: ignore[raise-vanilla-args] - one-off validation message
            "empty coordinate grid"
        )

    def sample(_rng: random.Random, iteration: int, experiment: int) -> str:
        return coords[(iteration * experiments_per_iter + experiment) % len(coords)]

    return sample


EventHook = Callable[[str], None]

# Canonical campaign frontier: lower loss, higher stability, lower energy.
# (pareto.pareto_frontier pre-negates cost objectives; maximize=True on each
# entry therefore means "the stated objective improves".)
DEFAULT_PARETO_OBJECTIVES: tuple[str, ...] = ("task_loss", "stability_score", "energy")
DEFAULT_PARETO_MAXIMIZE: tuple[bool, ...] = (True, True, True)


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Result of one coordinate evaluation attempt."""

    coordinate: str
    status: Literal[
        "recorded", "already_recorded", "guard_killed", "unsupported", "dry_run"
    ]
    record: FrontierRecord | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CampaignRunResult:
    """Summary of a ``CampaignStack.run_campaign`` invocation."""

    campaign_id: str
    branch: str
    db_path: Path
    iterations_run: tuple[int, ...]
    outcomes: tuple[EpisodeOutcome, ...]
    checkpoint_path: Path | None = None
    yaml_checkpoint: Path | None = None

    @property
    def records(self) -> tuple[FrontierRecord, ...]:
        return tuple(o.record for o in self.outcomes if o.record is not None)

    @property
    def unreplicated(self) -> tuple[ReplicationReport, ...]:
        from computronium.core.campaign.replication import unreplicated

        return unreplicated(self.records)

    def counterfactuals(
        self, *, metric: str = "task_accuracy"
    ) -> list[AxisAttribution]:
        return attribute_axis_effects(self.records, metric=metric)


class NoCampaignToResumeError(ValueError):
    """Resume requested but the branch has no campaign."""


class NoCampaignToCheckpointError(ValueError):
    """Checkpoint requested but no campaign exists on the branch."""


class CampaignStack:
    """Facade composing store, checkpoints, evaluation, and analysis.

    Args:
        root: Directory for the SQLite DB and checkpoints.
        branch: Campaign branch to run on.
        checkpoint_interval: Episodes between fault-tolerance checkpoints.
        guard_threshold: Stability-guard kill threshold (``None`` = observe only).
        seed: Base seed for coordinate sampling and episode streams.
        db_path: Optional explicit DB path (default ``root/campaign.db``).
        checkpoint_dir: Optional explicit checkpoint directory.
        on_event: Optional sink for human-readable progress events.
        device: Episode execution device (``"auto"`` = best available
            backend, ``None`` = build in place); θ restore copies honor it.
    """

    def __init__(  # ruff: ignore[too-many-arguments] - facade surface, all independently settable
        self,
        root: str | Path,
        *,
        branch: str = "main",
        checkpoint_interval: int = 5,
        guard_threshold: float | None = DEFAULT_GUARD_TAU,
        seed: int = 0,
        db_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        on_event: EventHook | None = None,
        device: str | torch.device | None = "auto",
    ):
        from computronium.core.utils.device import get_device

        self.root = Path(root)
        self.branch = branch
        self.guard_threshold = guard_threshold
        self.seed = seed
        self.device = None if device is None else get_device(device)
        self.store = CampaignStore(
            db_path or (self.root / "campaign.db"),
            checkpoint_dir or (self.root / "checkpoints"),
        )
        self.manager = CheckpointManager(
            checkpoint_dir or (self.root / "checkpoints"),
            checkpoint_interval=checkpoint_interval,
        )
        self._on_event = on_event

    # -- orchestration -----------------------------------------------------

    def run_campaign(  # ruff: ignore[too-many-arguments] - orchestration knobs, all defaulted
        self,
        *,
        iterations: int = 1,
        experiments_per_iter: int = 4,
        tasks: Sequence[str] = ("synthetic",),
        sampler: CoordinateSampler | None = None,
        campaign_id: str | None = None,
        resume: bool = False,
        dry_run: bool = False,
        build_kwargs: dict | None = None,
        objective: str | None = None,
    ) -> CampaignRunResult:
        """Run (or resume) a campaign of real composed-system episodes.

        Args:
            iterations: Number of campaign iterations to run.
            experiments_per_iter: Coordinates proposed per iteration.
            tasks: Task-family labels rotated across episodes (drives the
                replication gate; batches are family-aware — synthetic, parity).
            sampler: Coordinate sampler receiving the per-iteration seeded RNG
                plus the (iteration, experiment) slot; defaults to the built-in
                joint smoke space.
            campaign_id: Explicit campaign ID (auto-generated when creating).
            resume: Continue the latest campaign on the branch, replaying
                episodes lost to a crash since the last checkpoint and skipping
                episodes already durably recorded.
            dry_run: Propose coordinates without executing them.
            build_kwargs: Overrides for coordinate composition (input_dim,
                output_dim, hidden_dims).
            objective: Recorded in campaign config metadata for audit.

        Returns:
            CampaignRunResult with per-episode outcomes and analysis handles.
        """
        build_kwargs = {
            "input_dim": DEFAULT_INPUT_DIM,
            "output_dim": DEFAULT_NUM_CLASSES,
            "hidden_dims": (16,),
            **(build_kwargs or {}),
        }
        task_cycle = tuple(tasks) or ("synthetic",)

        if resume:
            state = self.store.get_latest_on_branch(self.branch)
            if state is None:
                raise NoCampaignToResumeError(self.branch)
            campaign_id = state.campaign_id
            self._event(
                f"Resuming campaign {campaign_id} on branch {self.branch!r} "
                f"at iteration {state.iteration}"
            )
            self._redo_unrecorded_episodes(
                campaign_id=campaign_id,
                last_complete=state.iteration,
                build_kwargs=build_kwargs,
            )
            # Crash mid-iteration leaves durable episodes behind an unadvanced
            # iteration counter; re-proposing those slots must skip, not
            # duplicate, the recorded (iteration, coordinate) rows. The task
            # family is a deterministic function of the slot's visit state, so
            # keying the skip on the task would re-execute rows written under
            # a different rotation rule.
            recorded = {
                (ep.iteration, ep.coordinate)
                for ep in self.store.get_episodes(campaign_id)
            }
            start_iteration = state.iteration
        else:
            state = self.store.create_campaign(
                campaign_id=campaign_id or f"camp_{uuid.uuid4().hex[:8]}",
                branch_name=self.branch,
                config={"objective": objective},
                metadata={"created_by": "CampaignStack", "seed": self.seed},
            )
            campaign_id = state.campaign_id
            self._event(f"Created campaign {campaign_id} on branch {self.branch!r}")
            recorded = set()
            start_iteration = 0

        sample = sampler or _space_sampler(_SMOKE_SPACE)
        # Per-coordinate visit counts drive task-family alternation: every
        # repeat visit of a coordinate flips the family regardless of grid
        # cycle parity (the slot-parity rotation left each coordinate on a
        # single family whenever the cycle was even — the R5b-B replication
        # gate read 0/48 for exactly that reason). Counts derive from durable
        # rows plus in-run recordings, so resume replays the same assignment.
        visits: dict[str, int] = Counter(c for _, c in recorded)
        outcomes: list[EpisodeOutcome] = []
        for iteration in range(start_iteration + 1, start_iteration + iterations + 1):
            # Coordinate stream is derived from (seed, campaign, iteration):
            # resume replays identical proposals without persistent sampler state.
            rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage] - sampling, not security-sensitive
                f"{self.seed}:{campaign_id}:{iteration}"
            )
            self._event(f"Iteration {iteration}: {experiments_per_iter} proposals")
            # One ENTERING-episode snapshot per interval iteration, taken on
            # that iteration's first recorded experiment.
            checkpointed = False
            for experiment in range(experiments_per_iter):
                coordinate = sample(rng, iteration, experiment)
                task_name = task_cycle[visits.get(coordinate, 0) % len(task_cycle)]
                if (iteration, coordinate) in recorded:
                    self._event(f"  skipped (already recorded): [{coordinate}]")
                    outcomes.append(
                        EpisodeOutcome(coordinate=coordinate, status="already_recorded")
                    )
                    continue
                outcome = self._evaluate(
                    campaign_id=campaign_id,
                    iteration=iteration,
                    coordinate=coordinate,
                    task_name=task_name,
                    dry_run=dry_run,
                    build_kwargs=build_kwargs,
                    checkpoint_joint=not checkpointed,
                )
                checkpointed = checkpointed or outcome.status == "recorded"
                if outcome.status == "recorded":
                    visits[coordinate] = visits.get(coordinate, 0) + 1
                outcomes.append(outcome)
            self.store.update_iteration(campaign_id, iteration)

        yaml_path = self.save_yaml_checkpoint(campaign_id=campaign_id)
        self._event(f"Campaign {campaign_id}: {len(outcomes)} episode attempt(s)")
        return CampaignRunResult(
            campaign_id=campaign_id,
            branch=self.branch,
            db_path=self.store.db_path,
            iterations_run=tuple(
                range(start_iteration + 1, start_iteration + iterations + 1)
            ),
            outcomes=tuple(outcomes),
            yaml_checkpoint=yaml_path,
        )

    def _evaluate(  # ruff: ignore[too-many-arguments] - episode context travels as one bundle
        self,
        *,
        campaign_id: str,
        iteration: int,
        coordinate: str,
        task_name: str,
        dry_run: bool,
        build_kwargs: dict,
        checkpoint_joint: bool,
    ) -> EpisodeOutcome:
        """Compose, checkpoint, evaluate, and persist one episode."""
        if dry_run:
            return EpisodeOutcome(coordinate=coordinate, status="dry_run")
        try:
            import torch

            # θ init draws otherwise ride the ambient RNG; per-episode
            # construction seeding makes rebuilds replay-safe across resume.
            torch.manual_seed(
                episode_seed(self.seed, campaign_id, iteration, coordinate)
            )
            joint: JointSystem = build_coordinate_system(
                coordinate, device=self.device, **build_kwargs
            )
        except UnsupportedCoordinateError as exc:
            self._event(f"  skipped (unsupported): {exc}")
            return EpisodeOutcome(
                coordinate=coordinate, status="unsupported", detail=str(exc)
            )

        if checkpoint_joint and self.manager.should_checkpoint(iteration):
            self._checkpoint_entering(
                campaign_id=campaign_id,
                iteration=iteration,
                coordinate=coordinate,
                task_name=task_name,
                joint=joint,
            )

        try:
            record, _metrics = evaluate_episode(
                joint,
                coordinate=coordinate,
                task_name=task_name,
                campaign_id=campaign_id,
                episode=iteration,
                guard_threshold=self.guard_threshold,
                seed=self.seed,
            )
        except GuardKillError as exc:
            self._event(f"  skipped (guard kill): {exc}")
            return EpisodeOutcome(
                coordinate=coordinate, status="guard_killed", detail=str(exc)
            )

        episode_id = self.store.add_episode(
            campaign_id=campaign_id,
            branch_name=self.branch,
            iteration=iteration,
            coordinate=coordinate,
            task_name=task_name,
            frontier_record=record,
        )
        self.store.add_registry_snapshot(
            campaign_id=campaign_id,
            episode_id=episode_id,
            registry_signature=record.registry_signature,
            composite_state_shape=record.composite_state_shape,
            plasticity_primitive=record.plasticity_primitive,
            plasticity_config=record.plasticity_config,
        )
        self._event(
            f"  recorded [{coordinate}]: loss={record.task_loss:.4f} "
            f"acc={record.task_accuracy:.4f}"
        )
        return EpisodeOutcome(coordinate=coordinate, status="recorded", record=record)

    def _checkpoint_entering(
        self,
        *,
        campaign_id: str,
        iteration: int,
        coordinate: str,
        task_name: str,
        joint: JointSystem,
    ) -> Path | None:
        """Snapshot the system ENTERING an episode so resume replays bit-exact."""
        from computronium.state import CompositeState

        snapshot = self.store.get_campaign(campaign_id)
        if snapshot is None:
            return None
        path = self.manager.create_checkpoint(
            campaign_state=snapshot,
            episode_index=iteration,
            composite_state=CompositeState(
                activity={"x": episode_batch(iteration)[0]},
                plastic={},
                substrate={},
            ),
            context=joint.context,
            coordinate=coordinate,
            task_name=task_name,
        )
        self._event(f"  checkpoint -> {path.name}")
        return path

    def _redo_unrecorded_episodes(
        self,
        *,
        campaign_id: str,
        last_complete: int,
        build_kwargs: dict,
    ) -> None:
        """Replay episodes lost to a crash between the latest checkpoint and now."""
        ckpt_path = self.manager.get_latest_checkpoint(campaign_id)
        if ckpt_path is None:
            return
        checkpoint = self.manager.load_checkpoint(ckpt_path)
        if not self.manager.validate_checkpoint(checkpoint):
            self._event(
                f"  warning: checkpoint {ckpt_path.name} failed validation; "
                "skipping redo"
            )
            return

        task_name = checkpoint.task_name
        recorded = {
            (ep.iteration, ep.coordinate) for ep in self.store.get_episodes(campaign_id)
        }
        joint = self._restore_checkpointed_joint(checkpoint, build_kwargs=build_kwargs)
        for episode in range(checkpoint.episode_index, last_complete + 1):
            if (episode, checkpoint.coordinate) in recorded:
                continue
            try:
                record, _metrics = evaluate_episode(
                    joint,
                    coordinate=checkpoint.coordinate,
                    task_name=task_name,
                    campaign_id=campaign_id,
                    episode=episode,
                    guard_threshold=self.guard_threshold,
                    seed=self.seed,
                )
            except GuardKillError as exc:
                self._event(f"  episode {episode} redone but guard-killed: {exc}")
                continue
            episode_id = self.store.add_episode(
                campaign_id=campaign_id,
                branch_name=checkpoint.branch_name or self.branch,
                iteration=episode,
                coordinate=record.coordinate,
                task_name=task_name,
                frontier_record=record,
            )
            self.store.add_registry_snapshot(
                campaign_id=campaign_id,
                episode_id=episode_id,
                registry_signature=record.registry_signature,
                composite_state_shape=record.composite_state_shape,
                plasticity_primitive=record.plasticity_primitive,
                plasticity_config=record.plasticity_config,
            )
            self.store.update_iteration(campaign_id, episode)
            self._event(f"  redid episode {episode}: loss={record.task_loss:.4f}")

    def _restore_checkpointed_joint(
        self, checkpoint: JointCheckpoint, *, build_kwargs: dict
    ) -> JointSystem:
        """Rebuild the checkpointed coordinate's joint and reload θ + RNG."""
        import torch

        fallback = (
            "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean"
        )
        joint = build_coordinate_system(
            checkpoint.coordinate or fallback,
            device=self.device,
            **build_kwargs,
        )
        with torch.no_grad():
            for name, array in checkpoint.theta.items():
                joint.geometry.params[name].copy_(torch.from_numpy(array))
        self.manager.restore_rng_states(checkpoint)
        return joint

    # -- analysis ----------------------------------------------------------

    def episodes(self, campaign_id: str | None = None) -> list[EpisodeRecord]:
        """Episode records for a campaign (latest on the branch by default)."""
        cid = campaign_id or self._latest_campaign_id()
        if cid is None:
            return []
        return self.store.get_episodes(cid)

    def frontier_records(self, campaign_id: str | None = None) -> list[FrontierRecord]:
        """Frontier records reconstructed from persisted episodes."""
        return [
            FrontierRecord.from_dict(ep.frontier_record)
            for ep in self.episodes(campaign_id)
        ]

    def pareto(
        self,
        *,
        objectives: tuple[str, ...] = DEFAULT_PARETO_OBJECTIVES,
        maximize: tuple[bool, ...] = DEFAULT_PARETO_MAXIMIZE,
        reference_point: tuple[float, ...] | None = None,
        task_name: str | None = None,
        campaign_id: str | None = None,
    ) -> ParetoFrontier:
        """Pareto frontier over loss, stability, and resources for a branch."""
        cid = campaign_id or self._latest_campaign_id()
        branch = self.branch
        records: list[FrontierRecord] = []
        if cid is not None:
            records = [
                FrontierRecord.from_dict(ep.frontier_record)
                for ep in self.store.get_episodes(cid)
            ]
        else:
            records = [
                FrontierRecord.from_dict(r)
                for r in self.store.export_pareto_frontier(branch, task_name)
            ]
        if task_name is not None:
            records = [r for r in records if r.task_name == task_name]
        return pareto_frontier(records, objectives, maximize, reference_point)

    def replication(
        self,
        *,
        min_seeds: int = DEFAULT_MIN_SEEDS,
        min_families: int = DEFAULT_MIN_FAMILIES,
        campaign_id: str | None = None,
    ) -> dict[str, ReplicationReport]:
        """Replication-gate manifest over all coordinates in a campaign."""
        return replication_manifest(
            self.frontier_records(campaign_id),
            min_seeds=min_seeds,
            min_families=min_families,
        )

    def counterfactuals(
        self,
        *,
        metric: str = "task_accuracy",
        campaign_id: str | None = None,
    ) -> list[AxisAttribution]:
        """Data-grounded axis attribution over the campaign's records."""
        return attribute_axis_effects(self.frontier_records(campaign_id), metric=metric)

    def discovery_report(
        self,
        campaign_id: str | None = None,
        *,
        metric: str = "task_accuracy",
        fidelity: Mapping[str, object] | None = None,
    ) -> DiscoveryReport:
        """Static discovery report (R5b-F Stage 1) over persisted records."""
        from computronium.core.campaign.report import build_discovery_report

        return build_discovery_report(
            self.frontier_records(campaign_id),
            metric=metric,
            fidelity=fidelity,
        )

    # -- checkpoints ---------------------------------------------------------

    def save_yaml_checkpoint(
        self, *, campaign_id: str | None = None, filename: str | None = None
    ) -> Path:
        """Persist campaign state + episode history as human-readable YAML."""
        cid = campaign_id or self._latest_campaign_id()
        if cid is None:
            raise NoCampaignToCheckpointError
        state = self.store.get_campaign(cid)
        if state is None:
            raise NoCampaignToCheckpointError(cid)
        return self.store.save_checkpoint(state, self.store.get_episodes(cid), filename)

    def load_yaml_checkpoint(
        self, filepath: str | Path
    ) -> tuple[CampaignState, list[EpisodeRecord]]:
        """Load a YAML checkpoint written by ``save_yaml_checkpoint``."""
        return self.store.load_checkpoint(filepath)

    # -- internals -----------------------------------------------------------

    def _latest_campaign_id(self) -> str | None:
        state = self.store.get_latest_on_branch(self.branch)
        return state.campaign_id if state else None

    def _event(self, message: str) -> None:
        logger.info("%s", message)
        if self._on_event is not None:
            self._on_event(message)


_SMOKE_SPACE: dict[str, list[str]] = {
    "substrates": ["digital"],
    "geometries": ["feedforward", "recurrent"],
    "dynamics": ["energy_minimization", "instantaneous"],
    "plasticity": ["null", "routing", "fast_weights"],
    # thermodynamic_contrast is excluded: contrastive settling credit ×
    # instantaneous is R3.9-fenced, so cross-product proposals would emit
    # unsupported outcomes. local_goodness composes on both dynamics (its
    # instantaneous pairings are quarantined by the fidelity gate, not here).
    "credits": ["random_projections", "local_goodness"],
    "updates": ["euclidean"],
}
