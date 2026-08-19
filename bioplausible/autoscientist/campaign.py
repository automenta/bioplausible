"""
AutoScientistCampaign: Orchestrates multi-day autonomous campaigns.

Combines the Scientist (execution) with AutoScientist (reasoning/proposal)
into a continuous discovery loop with:
  - Hypothesis generation
  - Experiment proposal
  - Execution via Scientist or CoreTrainer
  - Result analysis
  - KnowledgeBase update

Features:
  - YAML + SQLite persistence for campaign state
  - Git-like branching for experimental lineages
  - Resume from checkpoints
"""

import json
import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from bioplausible.knowledge import KnowledgeBase

from bioplausible.autoscientist.proposer import ExperimentProposer
from bioplausible.autoscientist.reasoner import HypothesisReasoner
from bioplausible.core.exceptions import KnowledgeBaseError
from bioplausible.core.logging import get_logger
from bioplausible.core.trainer import CoreTrainer, TrainerConfig

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Immutable campaign state snapshot."""

    campaign_id: str
    branch_name: str
    parent_branch: str | None
    iteration: int
    created_at: str
    updated_at: str
    config: dict[str, object]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Single iteration record."""

    iteration: int
    timestamp: str
    branch_name: str
    n_proposals: int
    n_completed: int
    n_failed: int
    proposals: list[dict[str, object]]
    results: list[dict[str, object]]
    insights: list[str]


class CampaignDatabase:
    """SQLite-backed campaign persistence with branch support."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    branch_name TEXT NOT NULL,
                    parent_branch TEXT,
                    iteration INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config TEXT NOT NULL,  -- JSON
                    metadata TEXT NOT NULL  -- JSON
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_campaign_branch ON campaigns(branch_name)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    n_proposals INTEGER,
                    n_completed INTEGER,
                    n_failed INTEGER,
                    proposals TEXT NOT NULL,  -- JSON
                    results TEXT NOT NULL,    -- JSON
                    insights TEXT NOT NULL,   -- JSON
                    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_iter_campaign ON iterations(campaign_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_iter_branch ON iterations(branch_name)
            """)
            conn.commit()

    def create_campaign(
        self,
        campaign_id: str,
        branch_name: str,
        parent_branch: str | None,
        config: dict[str, object],
        metadata: dict[str, object] | None = None,
    ) -> CampaignState:
        """Create a new campaign or branch."""
        now = datetime.now().isoformat()
        state = CampaignState(
            campaign_id=campaign_id,
            branch_name=branch_name,
            parent_branch=parent_branch,
            iteration=0,
            created_at=now,
            updated_at=now,
            config=config,
            metadata=metadata or {},
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO campaigns
                (campaign_id, branch_name, parent_branch, iteration, created_at, updated_at, config, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.campaign_id,
                    state.branch_name,
                    state.parent_branch,
                    state.iteration,
                    state.created_at,
                    state.updated_at,
                    json.dumps(state.config),
                    json.dumps(state.metadata),
                ),
            )
            conn.commit()

        return state

    def get_campaign(self, campaign_id: str) -> CampaignState | None:
        """Get campaign by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return CampaignState(
            campaign_id=row["campaign_id"],
            branch_name=row["branch_name"],
            parent_branch=row["parent_branch"],
            iteration=row["iteration"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            config=json.loads(row["config"]),
            metadata=json.loads(row["metadata"]),
        )

    def get_latest_on_branch(self, branch_name: str) -> CampaignState | None:
        """Get the most recent campaign state on a branch."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM campaigns
                WHERE branch_name = ?
                ORDER BY iteration DESC, updated_at DESC
                LIMIT 1
                """,
                (branch_name,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return CampaignState(
            campaign_id=row["campaign_id"],
            branch_name=row["branch_name"],
            parent_branch=row["parent_branch"],
            iteration=row["iteration"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            config=json.loads(row["config"]),
            metadata=json.loads(row["metadata"]),
        )

    def update_iteration(
        self,
        campaign_id: str,
        iteration: int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Update campaign iteration counter."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            if metadata:
                conn.execute(
                    "UPDATE campaigns SET iteration = ?, updated_at = ?, metadata = ? WHERE campaign_id = ?",
                    (iteration, now, json.dumps(metadata), campaign_id),
                )
            else:
                conn.execute(
                    "UPDATE campaigns SET iteration = ?, updated_at = ? WHERE campaign_id = ?",
                    (iteration, now, campaign_id),
                )
            conn.commit()

    def add_iteration_record(self, record: IterationRecord) -> int:
        """Add an iteration record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO iterations
                (campaign_id, branch_name, iteration, timestamp, n_proposals, n_completed, n_failed,
                 proposals, results, insights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.iteration,  # campaign_id reused as iteration for now
                    record.branch_name,
                    record.iteration,
                    record.timestamp,
                    record.n_proposals,
                    record.n_completed,
                    record.n_failed,
                    json.dumps(record.proposals),
                    json.dumps(record.results),
                    json.dumps(record.insights),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_iteration_history(self, branch_name: str) -> list[IterationRecord]:
        """Get all iterations for a branch."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM iterations
                WHERE branch_name = ?
                ORDER BY iteration ASC
                """,
                (branch_name,),
            )
            rows = cursor.fetchall()

        return [
            IterationRecord(
                iteration=row["iteration"],
                timestamp=row["timestamp"],
                branch_name=row["branch_name"],
                n_proposals=row["n_proposals"],
                n_completed=row["n_completed"],
                n_failed=row["n_failed"],
                proposals=json.loads(row["proposals"]),
                results=json.loads(row["results"]),
                insights=json.loads(row["insights"]),
            )
            for row in rows
        ]

    def list_branches(self) -> list[str]:
        """List all branch names."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT branch_name FROM campaigns ORDER BY branch_name"
            )
            return [row[0] for row in cursor.fetchall()]

    def list_campaigns(self, branch_name: str | None = None) -> list[CampaignState]:
        """List campaigns, optionally filtered by branch."""
        query = "SELECT * FROM campaigns"
        params: tuple = ()
        if branch_name:
            query += " WHERE branch_name = ?"
            params = (branch_name,)
        query += " ORDER BY updated_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            CampaignState(
                campaign_id=row["campaign_id"],
                branch_name=row["branch_name"],
                parent_branch=row["parent_branch"],
                iteration=row["iteration"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                config=json.loads(row["config"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]


class CampaignCheckpointer:
    """Handles YAML checkpoint serialization for human-readable campaign state."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        campaign_state: CampaignState,
        iteration_history: list[IterationRecord],
        filename: str | None = None,
    ) -> Path:
        """Save campaign state as human-readable YAML."""
        if filename is None:
            filename = (
                f"{campaign_state.branch_name}_iter{campaign_state.iteration:04d}.yaml"
            )

        filepath = self.checkpoint_dir / filename

        data = {
            "campaign": asdict(campaign_state),
            "history": [asdict(r) for r in iteration_history],
        }

        with filepath.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Checkpoint saved: %s", filepath)
        return filepath

    def load_checkpoint(
        self, filepath: str | Path
    ) -> tuple[CampaignState, list[IterationRecord]]:
        """Load campaign state from YAML checkpoint."""
        with Path(filepath).open() as f:
            data = yaml.safe_load(f)

        campaign = CampaignState(**data["campaign"])
        history = [IterationRecord(**r) for r in data.get("history", [])]

        return campaign, history

    def list_checkpoints(self, branch_name: str | None = None) -> list[Path]:
        """List available checkpoints."""
        pattern = f"{branch_name}_*.yaml" if branch_name else "*.yaml"
        return sorted(self.checkpoint_dir.glob(pattern))


class AutoScientistCampaign:
    """
    Autonomous research campaign manager with persistence and branching.

    Runs continuous discovery loops:
        1. Reason: Generate hypotheses from KnowledgeBase
        2. Propose: Convert hypotheses to experiment proposals
        3. Execute: Run experiments via CoreTrainer
        4. Learn: Update KnowledgeBase with results

    Features:
        - SQLite + YAML persistence
        - Git-like branching (create_branch, checkout, merge)
        - Resume from any checkpoint
        - Human approval gates
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        output_dir: str = "autoscientist_campaigns",
        db_path: str | Path | None = None,
        branch_name: str = "main",
        parent_branch: str | None = None,
        campaign_id: str | None = None,
        resume: bool = False,
        max_concurrent: int = 1,
        human_approval_gate: bool = False,
    ):
        self.knowledge_base = knowledge_base
        self.proposer = (
            ExperimentProposer(self.knowledge_base) if self.knowledge_base else None
        )
        self.reasoner = (
            HypothesisReasoner(self.knowledge_base) if self.knowledge_base else None
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Persistence layer
        self.db_path = db_path or (self.output_dir / "campaign.db")
        self.db = CampaignDatabase(self.db_path)
        self.checkpointer = CampaignCheckpointer(self.output_dir / "checkpoints")

        # Campaign identity
        self.branch_name = branch_name
        self.parent_branch = parent_branch
        self.campaign_id = campaign_id or f"camp_{uuid.uuid4().hex[:8]}"
        self.max_concurrent = max_concurrent
        self.human_approval_gate = human_approval_gate

        # Runtime state
        self._iteration = 0
        self._campaign_state: CampaignState | None = None
        self._config: dict[str, object] = {}

        # Resume or initialize
        if resume:
            self._resume()
        else:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize a new campaign or branch."""
        self._config = {
            "max_concurrent": self.max_concurrent,
            "human_approval_gate": self.human_approval_gate,
            "knowledge_base_path": str(self.knowledge_base.db_path)
            if self.knowledge_base
            else None,
        }

        # Check if branch exists
        existing = self.db.get_latest_on_branch(self.branch_name)
        if existing:
            # Branch exists, continue from latest
            self._campaign_state = existing
            self.campaign_id = existing.campaign_id
            self._iteration = existing.iteration
            logger.info(
                "Resuming branch '%s' at iteration %d (campaign %s)",
                self.branch_name,
                self._iteration,
                self.campaign_id,
            )
        else:
            # Create new campaign
            self._campaign_state = self.db.create_campaign(
                campaign_id=self.campaign_id,
                branch_name=self.branch_name,
                parent_branch=self.parent_branch,
                config=self._config,
                metadata={"created_by": "AutoScientistCampaign", "version": "1.0"},
            )
            self._iteration = 0
            logger.info(
                "Created new campaign %s on branch '%s'",
                self.campaign_id,
                self.branch_name,
            )

    def _resume(self) -> None:
        """Resume from latest checkpoint on current branch."""
        state = self.db.get_latest_on_branch(self.branch_name)
        if not state:
            raise ValueError(
                f"No campaign found on branch '{self.branch_name}' to resume"
            )

        self._campaign_state = state
        self.campaign_id = state.campaign_id
        self._iteration = state.iteration
        self._config = state.config

        logger.info(
            "Resumed campaign %s on branch '%s' at iteration %d",
            self.campaign_id,
            self.branch_name,
            self._iteration,
        )

    @classmethod
    def create_branch(
        cls,
        source_branch: str,
        new_branch: str,
        db_path: str | Path,
        knowledge_base: KnowledgeBase | None = None,
        output_dir: str = "autoscientist_campaigns",
    ) -> AutoScientistCampaign:
        """
        Create a new branch from an existing branch (git-like).

        The new branch starts from the latest state of the source branch.
        """
        db = CampaignDatabase(db_path)
        source_state = db.get_latest_on_branch(source_branch)
        if not source_state:
            raise ValueError(f"Source branch '{source_branch}' not found")

        # Create new campaign on new branch, inheriting from source
        new_campaign_id = f"camp_{uuid.uuid4().hex[:8]}"
        new_state = db.create_campaign(
            campaign_id=new_campaign_id,
            branch_name=new_branch,
            parent_branch=source_branch,
            config=source_state.config,
            metadata={
                "forked_from": source_state.campaign_id,
                "forked_at_iteration": source_state.iteration,
                "created_by": "branch",
            },
        )

        return cls(
            knowledge_base=knowledge_base,
            output_dir=output_dir,
            db_path=db_path,
            branch_name=new_branch,
            parent_branch=source_branch,
            campaign_id=new_campaign_id,
            resume=True,
        )

    def checkout(self, branch_name: str) -> None:
        """Switch to a different branch (like git checkout)."""
        state = self.db.get_latest_on_branch(branch_name)
        if not state:
            raise ValueError(f"Branch '{branch_name}' not found")

        self.branch_name = branch_name
        self._campaign_state = state
        self.campaign_id = state.campaign_id
        self._iteration = state.iteration
        self._config = state.config

        logger.info(
            "Checked out branch '%s' at iteration %d", branch_name, self._iteration
        )

    def merge_from(self, source_branch: str, strategy: str = "latest") -> None:
        """
        Merge insights from another branch.

        Strategies:
            - 'latest': Take the latest iteration from source branch
            - 'best': Take the best performing iteration from source branch
        """
        source_state = self.db.get_latest_on_branch(source_branch)
        if not source_state:
            raise ValueError(f"Source branch '{source_branch}' not found")

        # Get best iteration from source
        history = self.db.get_iteration_history(source_branch)
        if not history:
            logger.warning(
                "Source branch '%s' has no iterations to merge", source_branch
            )
            return

        if strategy == "best":
            best_iter = max(
                history,
                key=lambda r: max(
                    (res.get("final_accuracy", 0) for res in r.results), default=0
                ),
            )
            logger.info(
                "Merging best iteration %d from branch '%s'",
                best_iter.iteration,
                source_branch,
            )
        else:
            best_iter = history[-1]
            logger.info(
                "Merging latest iteration %d from branch '%s'",
                best_iter.iteration,
                source_branch,
            )

        # Add merged insights to current campaign metadata
        if self._campaign_state:
            merged_meta = dict(self._campaign_state.metadata)
            merged_meta.setdefault("merged_from", []).append({
                "branch": source_branch,
                "iteration": best_iter.iteration,
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
            })
            self.db.update_iteration(self.campaign_id, self._iteration, merged_meta)

    def run_iteration(
        self,
        domain: str | None = None,
        n_experiments: int = 5,
        dry_run: bool = False,
    ) -> list[dict[str, object]]:
        """
        Run one iteration of the discovery loop.

        Args:
            domain: Optional domain filter.
            n_experiments: Number of experiments to propose and run.
            dry_run: If True, only propose without executing.

        Returns:
            List of experiment results.
        """
        self._iteration += 1
        logger.info(
            "=== Campaign %s Iteration %d ===", self.campaign_id, self._iteration
        )

        insights = []
        if self.reasoner:
            insights = self.reasoner.analyze_knowledge_base()
            if insights:
                logger.info("KnowledgeBase insights (%d):", len(insights))
                for insight in insights[:3]:
                    logger.info("  - %s", insight)

        proposals = []
        if self.proposer:
            proposals = self.proposer.propose_batch(
                domain=domain,
                n_proposals=n_experiments,
            )

        if not proposals:
            logger.warning("No proposals generated. Skipping iteration.")
            self._record_iteration(proposals, [], insights)
            return []

        logger.info("Proposed %d experiments", len(proposals))

        # Human approval gate
        if self.human_approval_gate:
            approved = self._human_approval(proposals)
            proposals = [p for i, p in enumerate(proposals) if i in approved]
            if not proposals:
                logger.info("No proposals approved. Skipping.")
                self._record_iteration(proposals, [], insights)
                return []

        # Execute experiments
        results = []
        if not dry_run:
            for i, proposal in enumerate(proposals):
                logger.info(
                    "Executing proposal %d/%d: %s on %s",
                    i + 1,
                    len(proposals),
                    proposal.model,
                    proposal.task,
                )
                try:
                    result = self._execute_proposal(proposal)
                    results.append(result)

                    if self.knowledge_base:
                        self._update_knowledge_base(proposal, result)
                except (
                    Exception
                ) as e:  # broad: a failing trial must not stop the campaign
                    logger.error("Proposal %d failed: %s", i, e, exc_info=True)
                    results.append({
                        "proposal": {
                            "model": proposal.model,
                            "task": proposal.task,
                            "hypothesis": proposal.hypothesis,
                        },
                        "status": "failed",
                        "error": str(e),
                    })

        # Record and persist
        self._record_iteration(proposals, results, insights)

        # Update campaign state in DB
        if self._campaign_state:
            self.db.update_iteration(self.campaign_id, self._iteration)

        return results

    def _execute_proposal(self, proposal) -> dict[str, object]:
        """Execute a single experiment proposal via CoreTrainer."""
        config = TrainerConfig(
            model=proposal.model,
            task=proposal.task,
            optimizer=proposal.optimizer,
            epochs=5,
            batch_size=64,
            track_energy=True,
            tags={
                "hypothesis": proposal.hypothesis,
                "autoscientist": True,
                "iteration": self._iteration,
                "campaign_id": self.campaign_id,
                "branch": self.branch_name,
            },
        )

        if proposal.propagator:
            config.propagator = proposal.propagator

        for k, v in proposal.hyperparams.items():
            if hasattr(config, k):
                setattr(config, k, v)

        trainer = CoreTrainer(config)
        history = trainer.fit()

        return {
            "proposal": {
                "hypothesis": proposal.hypothesis,
                "model": proposal.model,
                "task": proposal.task,
                "propagator": proposal.propagator,
                "optimizer": proposal.optimizer,
                "justification": proposal.justification,
            },
            "status": "completed",
            "metrics": [m.to_dict() for m in history],
            "final_accuracy": history[-1].val_acc if history else 0.0,
            "final_loss": history[-1].val_loss if history else 0.0,
            "train_accuracy": history[-1].train_acc if history else 0.0,
            "epochs_completed": len(history),
        }

    def _update_knowledge_base(self, proposal, result: dict[str, object]) -> None:
        """Store experiment result in KnowledgeBase with schema validation."""
        try:
            from bioplausible.knowledge import KnowledgeEntry

            entry = KnowledgeEntry(
                id=f"campaign_{self.campaign_id}_iter{self._iteration}_{int(time.time())}",
                topic=f"experiment:{proposal.task}",
                model_family=proposal.model,
                finding=(
                    f"Accuracy: {result.get('final_accuracy', 'N/A'):.4f}, "
                    f"Loss: {result.get('final_loss', 'N/A'):.4f}"
                ),
                details=(
                    f"Hypothesis: {proposal.hypothesis}\n"
                    f"Propagator: {proposal.propagator}\n"
                    f"Optimizer: {proposal.optimizer}\n"
                    f"Config: {proposal.hyperparams}\n"
                    f"Epochs: {result.get('epochs_completed', 'N/A')}"
                ),
                confidence=float(result.get("final_accuracy", 0.0) or 0.0),
                tags=[
                    "experiment",
                    proposal.task,
                    proposal.model,
                    f"campaign:{self.campaign_id}",
                ],
                source="experiment",
                metrics={
                    k: v
                    for k, v in result.items()
                    if isinstance(v, (int, float)) and v is not None
                },
                hyperparameters=(
                    proposal.hyperparams
                    if isinstance(proposal.hyperparams, dict)
                    else {"raw": str(proposal.hyperparams)}
                ),
                extra={
                    "campaign_id": self.campaign_id,
                    "campaign_iteration": self._iteration,
                    "branch": self.branch_name,
                },
            )
            self.knowledge_base.add_entry(entry)
        except (KnowledgeBaseError, OSError, ValueError) as e:
            logger.warning("Failed to update KnowledgeBase: %s", e)

    def _record_iteration(
        self,
        proposals: list,
        results: list[dict[str, object]],
        insights: list[str],
    ) -> None:
        """Record iteration to database and checkpoint."""
        record = IterationRecord(
            iteration=self._iteration,
            timestamp=datetime.now().isoformat(),
            branch_name=self.branch_name,
            n_proposals=len(proposals),
            n_completed=sum(1 for r in results if r.get("status") == "completed"),
            n_failed=sum(1 for r in results if r.get("status") == "failed"),
            proposals=[
                {
                    "model": p.model,
                    "task": p.task,
                    "hypothesis": p.hypothesis[:100],
                    "propagator": p.propagator,
                    "priority": p.priority,
                }
                for p in proposals
            ],
            results=[
                {
                    "status": r.get("status"),
                    "final_accuracy": r.get("final_accuracy"),
                    "final_loss": r.get("final_loss"),
                }
                for r in results
            ],
            insights=insights,
        )

        self.db.add_iteration_record(record)

        # Save checkpoint every 5 iterations
        if self._iteration % 5 == 0 and self._campaign_state:
            updated_state = CampaignState(
                campaign_id=self._campaign_state.campaign_id,
                branch_name=self._campaign_state.branch_name,
                parent_branch=self._campaign_state.parent_branch,
                iteration=self._iteration,
                created_at=self._campaign_state.created_at,
                updated_at=datetime.now().isoformat(),
                config=self._campaign_state.config,
                metadata=self._campaign_state.metadata,
            )
            self.checkpointer.save_checkpoint(
                updated_state, self.db.get_iteration_history(self.branch_name)
            )

        # Also save JSON log (backward compatible)
        log_path = self.output_dir / f"iteration_{self._iteration:04d}.json"
        with log_path.open("w") as f:
            json.dump(asdict(record), f, indent=2, default=str)

        logger.info(
            "Iteration %d logged (completed: %d, failed: %d)",
            self._iteration,
            record.n_completed,
            record.n_failed,
        )

    def _human_approval(self, proposals: list) -> list[int]:
        """Gate for human approval of expensive runs."""
        logger.info("Human approval gate: %d proposals pending", len(proposals))
        if not proposals:
            return []
        if not sys.stdin.isatty():
            auto_msg = (
                "stdin is not a TTY; auto-approving all proposals. "
                "Provide BIOPL_AUTO_APPROVE=0 to deny."
            )
            if os.environ.get("BIOPL_AUTO_APPROVE", "1") == "0":
                logger.warning(auto_msg + " Denying all (BIOPL_AUTO_APPROVE=0).")
                return []
            logger.info(auto_msg)
            return list(range(len(proposals)))
        approved: list[int] = []
        for idx, proposal in enumerate(proposals):
            label = (
                getattr(proposal, "name", None) or f"{proposal.model}/{proposal.task}"
            )
            try:
                answer = (
                    input(f"Approve proposal {idx} ({label})? [y/N] ").strip().lower()
                )
            except EOFError:
                logger.warning("EOF on stdin; auto-approving remaining proposals")
                approved.extend(range(idx, len(proposals)))
                return approved
            if answer == "y":
                approved.append(idx)
        return approved

    def save_checkpoint(self, filename: str | None = None) -> Path:
        """Manually save a checkpoint."""
        if not self._campaign_state:
            raise RuntimeError("No campaign state to save")

        updated_state = CampaignState(
            campaign_id=self._campaign_state.campaign_id,
            branch_name=self._campaign_state.branch_name,
            parent_branch=self._campaign_state.parent_branch,
            iteration=self._iteration,
            created_at=self._campaign_state.created_at,
            updated_at=datetime.now().isoformat(),
            config=self._campaign_state.config,
            metadata=self._campaign_state.metadata,
        )
        return self.checkpointer.save_checkpoint(
            updated_state,
            self.db.get_iteration_history(self.branch_name),
            filename,
        )

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Load campaign state from a checkpoint file."""
        campaign, history = self.checkpointer.load_checkpoint(filepath)
        self._campaign_state = campaign
        self.campaign_id = campaign.campaign_id
        self.branch_name = campaign.branch_name
        self.parent_branch = campaign.parent_branch
        self._iteration = campaign.iteration
        self._config = campaign.config
        logger.info(
            "Loaded checkpoint: iteration %d, branch %s",
            self._iteration,
            self.branch_name,
        )

    def get_summary(self) -> dict[str, object]:
        """Get campaign summary statistics."""
        history = self.db.get_iteration_history(self.branch_name)

        completed = []
        for entry in history:
            for r in entry.results:
                if r.get("status") == "completed":
                    completed.append(r)

        total_experiments = sum(entry.n_proposals for entry in history)
        best_accuracy = 0.0
        if completed:
            best_accuracy = max(r.get("final_accuracy", 0) for r in completed)

        return {
            "campaign_id": self.campaign_id,
            "branch_name": self.branch_name,
            "parent_branch": self.parent_branch,
            "iterations": self._iteration,
            "total_experiments": total_experiments,
            "completed": len(completed),
            "best_accuracy": best_accuracy,
            "output_dir": str(self.output_dir),
            "db_path": str(self.db_path),
        }

    def get_history(self) -> list[IterationRecord]:
        """Get full iteration history for current branch."""
        return self.db.get_iteration_history(self.branch_name)

    @contextmanager
    def run_campaign(
        self,
        n_iterations: int = 10,
        domain: str | None = None,
        n_experiments_per_iter: int = 5,
        checkpoint_interval: int = 5,
    ):
        """
        Context manager for running a multi-iteration campaign.

        Usage:
            with campaign.run_campaign(n_iterations=20) as results:
                # results is a list of all iteration results
                pass
        """
        all_results = []
        try:
            for i in range(n_iterations):
                results = self.run_iteration(
                    domain=domain,
                    n_experiments=n_experiments_per_iter,
                )
                all_results.extend(results)

                if (i + 1) % checkpoint_interval == 0:
                    self.save_checkpoint()

            yield all_results
        finally:
            # Always save final checkpoint
            self.save_checkpoint()
            logger.info(
                "Campaign %s completed. Final summary: %s",
                self.campaign_id,
                self.get_summary(),
            )


def create_campaign(
    knowledge_base: KnowledgeBase | None = None,
    output_dir: str = "autoscientist_campaigns",
    branch: str = "main",
    resume: bool = False,
    **kwargs,
) -> AutoScientistCampaign:
    """Factory function to create or resume a campaign."""
    return AutoScientistCampaign(
        knowledge_base=knowledge_base,
        output_dir=output_dir,
        branch_name=branch,
        resume=resume,
        **kwargs,
    )


def list_campaigns(db_path: str | Path) -> list[CampaignState]:
    """List all campaigns in a database."""
    db = CampaignDatabase(db_path)
    return db.list_campaigns()


def list_branches(db_path: str | Path) -> list[str]:
    """List all branches in a database."""
    db = CampaignDatabase(db_path)
    return db.list_branches()


__all__ = [
    "AutoScientistCampaign",
    "CampaignCheckpointer",
    "CampaignDatabase",
    "CampaignState",
    "IterationRecord",
    "create_campaign",
    "list_branches",
    "list_campaigns",
    "logger",
]
