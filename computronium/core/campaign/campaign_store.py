"""
Campaign Store for Joint Architecture.

Persists campaign state including 6-D coordinates, StateRegistry metadata,
CompositeState shape signatures, FrontierRecords, ResourceUsage, and
episode consolidation events. Supports SQLite for structured queries and
YAML for human-readable checkpoints.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.core.campaign.frontier_record import FrontierRecord
    from computronium.state import StateRegistry, SystemContext

SCHEMA_VERSION = 1


class SchemaVersionError(RuntimeError):
    """Database schema is newer than this build (frozen schema, forward-only)."""

    def __init__(self, db_version: int, supported: int) -> None:
        super().__init__(
            f"Campaign DB schema v{db_version} > supported v{supported}; "
            "open it with a newer build instead"
        )
        self.db_version = db_version
        self.supported = supported


# MIGRATIONS[v] upgrades a v-old schema to v+1. Schema is frozen: every future
# schema change appends here, never mutates earlier entries.
MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Immutable campaign state snapshot."""

    campaign_id: str
    branch_name: str
    parent_branch: str | None
    iteration: int
    created_at: str
    updated_at: str
    config: dict
    metadata: dict


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Single episode record within a campaign."""

    iteration: int
    timestamp: str
    branch_name: str
    coordinate: str
    task_name: str
    frontier_record: dict  # Serialized FrontierRecord
    consolidation_event: dict | None = None
    rng_state: bytes | None = None


class CampaignStore:
    """
    SQLite + YAML backed campaign persistence.

    Stores:
    - Campaign metadata (branches, iterations)
    - 6-D coordinate evaluations with full FrontierRecords
    - StateRegistry signatures and CompositeState shapes
    - ResourceUsage vectors
    - Episode consolidation events
    - RNG state for reproducibility
    """

    def __init__(self, db_path: str | Path, checkpoint_dir: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = self.db_path.parent / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize or migrate the database schema (frozen, versioned)."""
        with sqlite3.connect(self.db_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise SchemaVersionError(version, SCHEMA_VERSION)
            if version == SCHEMA_VERSION:
                return
            if version == 0 and self._is_empty(conn):
                self._create_schema(conn)
            elif version == 0:
                # Legacy pre-freeze database: identical to the v1 schema.
                pass
            else:
                for v in range(version, SCHEMA_VERSION):
                    MIGRATIONS[v](conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()

    @staticmethod
    def _is_empty(conn: sqlite3.Connection) -> bool:
        """True when no campaigns table exists (fresh database)."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='campaigns'"
        ).fetchone()
        return row is None

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        """Create the v1 schema."""
        # Campaigns table
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_campaign_branch ON campaigns(branch_name)"
        )

        # Episodes table - stores full evaluation records
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                coordinate TEXT NOT NULL,       -- 6-D coordinate string
                task_name TEXT NOT NULL,
                frontier_record TEXT NOT NULL,  -- JSON serialized FrontierRecord
                consolidation_event TEXT,       -- JSON serialized consolidation event
                rng_state BLOB,                 -- Pickled RNG state
                FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
            )
        """)
        for index in (
            "CREATE INDEX IF NOT EXISTS idx_episode_campaign ON episodes(campaign_id)",
            "CREATE INDEX IF NOT EXISTS idx_episode_branch ON episodes(branch_name)",
            "CREATE INDEX IF NOT EXISTS idx_episode_coord ON episodes(coordinate)",
            "CREATE INDEX IF NOT EXISTS idx_episode_task ON episodes(task_name)",
        ):
            conn.execute(index)

        # State registry snapshots
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registry_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                episode_id INTEGER NOT NULL,
                registry_signature TEXT NOT NULL,  -- Hash of StateVariable registrations
                composite_state_shape TEXT NOT NULL,  -- JSON: {activity: {...}, plastic: {...}, substrate: {...}}
                plasticity_primitive TEXT NOT NULL,
                plasticity_config TEXT NOT NULL,  -- JSON
                FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id),
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            )
        """)

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @property
    def schema_version(self) -> int:
        """Schema version stamped in the database (frozen at SCHEMA_VERSION)."""
        with self._conn() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def create_campaign(
        self,
        branch_name: str = "main",
        parent_branch: str | None = None,
        config: dict | None = None,
        metadata: dict | None = None,
        campaign_id: str | None = None,
    ) -> CampaignState:
        """Create a new campaign or branch."""
        campaign_id = campaign_id or f"camp_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        state = CampaignState(
            campaign_id=campaign_id,
            branch_name=branch_name,
            parent_branch=parent_branch,
            iteration=0,
            created_at=now,
            updated_at=now,
            config=config or {},
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
        with self._conn() as conn:
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
        with self._conn() as conn:
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
        metadata: dict | None = None,
    ) -> None:
        """Update campaign iteration counter and metadata."""
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

    def add_episode(
        self,
        campaign_id: str,
        branch_name: str,
        iteration: int,
        coordinate: str,
        task_name: str,
        frontier_record: FrontierRecord,
        consolidation_event: dict | None = None,
        rng_state: bytes | None = None,
    ) -> int:
        """Add an episode evaluation record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO episodes
                (campaign_id, branch_name, iteration, timestamp, coordinate, task_name,
                 frontier_record, consolidation_event, rng_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    branch_name,
                    iteration,
                    datetime.now().isoformat(),
                    coordinate,
                    task_name,
                    json.dumps(frontier_record.to_dict()),
                    json.dumps(consolidation_event) if consolidation_event else None,
                    rng_state,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            assert row_id is not None  # ruff: ignore[assert]
            return row_id

    def add_registry_snapshot(
        self,
        campaign_id: str,
        episode_id: int,
        registry_signature: str,
        composite_state_shape: dict[str, dict[str, tuple[int, ...]]],
        plasticity_primitive: str,
        plasticity_config: dict,
    ) -> int:
        """Store StateRegistry and CompositeState shape signature."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO registry_snapshots
                (campaign_id, episode_id, registry_signature, composite_state_shape,
                 plasticity_primitive, plasticity_config)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    episode_id,
                    registry_signature,
                    json.dumps({
                        k: {sk: list(sv) for sk, sv in v.items()}
                        for k, v in composite_state_shape.items()
                    }),
                    plasticity_primitive,
                    json.dumps(plasticity_config),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            assert row_id is not None  # ruff: ignore[assert]
            return row_id

    def get_episodes(self, campaign_id: str) -> list[EpisodeRecord]:
        """Get all episodes for a campaign."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM episodes
                WHERE campaign_id = ?
                ORDER BY iteration ASC
                """,
                (campaign_id,),
            )
            rows = cursor.fetchall()

        return [
            EpisodeRecord(
                iteration=row["iteration"],
                timestamp=row["timestamp"],
                branch_name=row["branch_name"],
                coordinate=row["coordinate"],
                task_name=row["task_name"],
                frontier_record=json.loads(row["frontier_record"]),
                consolidation_event=json.loads(row["consolidation_event"])
                if row["consolidation_event"]
                else None,
                rng_state=row["rng_state"],
            )
            for row in rows
        ]

    def get_episodes_by_coordinate(
        self, coordinate: str, branch_name: str | None = None
    ) -> list[EpisodeRecord]:
        """Get all episodes matching a 6-D coordinate."""
        query = "SELECT * FROM episodes WHERE coordinate = ?"
        params: list = [coordinate]
        if branch_name:
            query += " AND branch_name = ?"
            params.append(branch_name)
        query += " ORDER BY iteration ASC"

        with self._conn() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            EpisodeRecord(
                iteration=row["iteration"],
                timestamp=row["timestamp"],
                branch_name=row["branch_name"],
                coordinate=row["coordinate"],
                task_name=row["task_name"],
                frontier_record=json.loads(row["frontier_record"]),
                consolidation_event=json.loads(row["consolidation_event"])
                if row["consolidation_event"]
                else None,
                rng_state=row["rng_state"],
            )
            for row in rows
        ]

    def get_best_episode(
        self, campaign_id: str, metric: str = "task_accuracy"
    ) -> EpisodeRecord | None:
        """Get the best episode by a metric."""
        episodes = self.get_episodes(campaign_id)
        if not episodes:
            return None

        best = max(episodes, key=lambda e: e.frontier_record.get(metric, 0))
        return best

    def list_branches(self) -> list[str]:
        """List all branch names."""
        with self._conn() as conn:
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

        with self._conn() as conn:
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

    def create_branch(
        self,
        source_branch: str,
        new_branch: str,
        config: dict | None = None,
        metadata: dict | None = None,
    ) -> CampaignState:
        """Create a new branch from an existing branch (git-like)."""
        source_state = self.get_latest_on_branch(source_branch)
        if not source_state:
            raise ValueError(f"Source branch '{source_branch}' not found")

        new_campaign_id = f"camp_{uuid.uuid4().hex[:8]}"
        new_metadata = metadata or {}
        new_metadata.update({
            "forked_from": source_state.campaign_id,
            "forked_at_iteration": source_state.iteration,
            "created_by": "branch",
        })

        return self.create_campaign(
            campaign_id=new_campaign_id,
            branch_name=new_branch,
            parent_branch=source_branch,
            config=config or source_state.config,
            metadata=new_metadata,
        )

    def save_checkpoint(
        self,
        campaign_state: CampaignState,
        episode_history: list[EpisodeRecord],
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
            "history": [asdict(r) for r in episode_history],
        }

        with filepath.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return filepath

    def load_checkpoint(
        self, filepath: str | Path
    ) -> tuple[CampaignState, list[EpisodeRecord]]:
        """Load campaign state from YAML checkpoint."""
        with Path(filepath).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        campaign = CampaignState(**data["campaign"])
        history = [EpisodeRecord(**r) for r in data.get("history", [])]

        return campaign, history

    def list_checkpoints(self, branch_name: str | None = None) -> list[Path]:
        """List available checkpoints."""
        pattern = f"{branch_name}_*.yaml" if branch_name else "*.yaml"
        return sorted(self.checkpoint_dir.glob(pattern))

    def export_pareto_frontier(
        self,
        branch_name: str,
        task_name: str | None = None,
    ) -> list[dict]:
        """Export Pareto frontier data for a branch/task."""
        query = """
            SELECT frontier_record FROM episodes
            WHERE branch_name = ?
        """
        params: list = [branch_name]
        if task_name:
            query += " AND task_name = ?"
            params.append(task_name)

        with self._conn() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        records = [json.loads(row["frontier_record"]) for row in rows]
        return records


def compute_registry_signature(registry: StateRegistry) -> str:
    """Compute a hash signature of the StateRegistry."""
    import hashlib

    vars_info = []
    for var in registry._variables.values():
        vars_info.append(
            f"{var.name}:{var.persistent}:{var.fast_plastic}:{var.substrate_owned}:{var.consolidatable}"
        )

    content = "|".join(sorted(vars_info))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_composite_state_shape(
    context: SystemContext,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Compute shape signature of CompositeState from SystemContext."""
    shape = {
        "activity": {},
        "plastic": {},
        "substrate": {},
    }

    # Activity shapes from geometry params
    for name, param in context.theta.items():
        shape["activity"][name] = tuple(param.shape)

    # Plastic shapes from plasticity config
    if context.plasticity_config.plastic_state_dims:
        for name, dim in context.plasticity_config.plastic_state_dims.items():
            shape["plastic"][name] = (dim,)  # Batch dim added at runtime

    # Substrate shapes from substrate config
    substrate_config = context.substrate_config
    if hasattr(substrate_config, "shape_hints"):
        shape_hints = getattr(substrate_config, "shape_hints", {})
        for name, shape_hint in shape_hints.items():
            shape["substrate"][name] = shape_hint

    return shape
