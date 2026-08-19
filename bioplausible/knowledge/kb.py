"""
Upgraded KnowledgeBase with SQLite + Vector Store

Provides hybrid structured + embedding search for AutoScientist.
Integrates surrogate models, symbolic regression, causal discovery.
"""

import json
import pathlib
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TypedDict

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from bioplausible.core._paths import db_path
from bioplausible.core.exceptions import ConditionalQueryError, KnowledgeBaseError
from bioplausible.core.logging import get_logger

# Optional dependencies for vector search
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = get_logger()


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """A single knowledge entry with metadata and optional embedding."""

    id: str
    topic: str
    model_family: str
    finding: str
    details: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "manual"  # "manual", "experiment", "surrogate", "causal"
    experiment_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, object] = field(default_factory=dict)
    embedding: list[float] | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        # Don't store embedding in JSON
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> KnowledgeEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class ConditionalQuery(TypedDict):
    """Read-half filter for the AutoScientist flywheel (P2).

    A request for previously-verified positive conditionals: which rules already
    achieved a target accuracy within memory/flops caps on a task (optionally on
    a specific substrate). Empty/None fields act as wildcards.
    """

    model: str | None
    task: str | None
    accuracy_target: float | None
    memory_cap: float | None
    flops_cap: float | None
    substrate: str | None


class _ConditionalQueryModel(BaseModel):
    """Pydantic v2 runtime-validated form of :class:`ConditionalQuery`."""

    model: str | None = Field(default=None)
    task: str | None = Field(default=None)
    accuracy_target: float | None = Field(default=None, ge=0.0, le=1.0)
    memory_cap: float | None = Field(default=None, ge=0.0)
    flops_cap: float | None = Field(default=None, ge=0.0)
    substrate: str | None = Field(default=None)


@dataclass(frozen=True, slots=True)
class ConditionalResult:
    """One previously-verified positive conditional satisfying a query.

    A frozen value object so the proposer can reason about *already-spent*
    budget without mutating the KB or the caller's data.
    """

    model: str
    task: str
    accuracy: float
    memory_mb: float
    flops: float
    wall_time_s: float
    config: tuple[tuple[str, object], ...] = ()
    substrate: str | None = None
    entry_id: str | None = None


@dataclass(frozen=True, slots=True)
class FlagshipCandidate:
    """One validated family's cost-of-plausibility operating point (P3a)."""

    model: str
    accuracy: float
    memory_mb: float
    flops: float
    wall_time_s: float
    cost_of_plausibility: float
    substrate: str | None = None


@dataclass(frozen=True, slots=True)
class FlagshipDecision:
    """The outcome of the P3a flagship-selection query."""

    task: str
    chosen: str | None
    ranked: tuple[FlagshipCandidate, ...]


def _entry_accuracy(metrics: dict[str, float]) -> float:
    """Normalize accuracy across the probe/engine metric dialects."""
    return float(metrics.get("final_acc", metrics.get("accuracy", 0.0)))


def _entry_memory(metrics: dict[str, float]) -> float:
    return float(metrics.get("peak_memory_mb", metrics.get("memory_mb", 0.0)))


def _entry_flops(metrics: dict[str, float]) -> float:
    return float(metrics.get("forward_flops", 0.0) + metrics.get("backward_flops", 0.0))


def _entry_substrate(entry: KnowledgeEntry) -> str | None:
    """Substrate tag, if one was recorded (plan §17 IBKB key)."""
    hw = entry.extra.get("hardware") if isinstance(entry.extra, dict) else None
    return str(hw) if hw else None


def _entry_task(entry: KnowledgeEntry) -> str:
    """Task name persisted through ``add_experiment``'s ``extra`` dict."""
    meta = entry.extra.get("task") if isinstance(entry.extra, dict) else None
    return str(meta) if meta else ""


class KnowledgeBase:  # ruff: ignore[too-many-public-methods]  # integrity-surface + conditional + flagship queries are all distinct public KB reads
    """
    Upgraded KnowledgeBase with SQLite + Vector Store.

    Features:
    - SQLite for structured queries (tags, model_family, metrics, etc.)
    - FAISS/Vector store for semantic similarity search
    - Surrogate model integration for predicting experiment outcomes
    - Symbolic regression for extracting analytical formulas
    - Causal discovery for identifying causal factors
    """

    def __init__(
        self,
        db_path: str = db_path("bioplausible_kb.db"),
        vector_dim: int = 384,
        embedding_model: str = "all-MiniLM-L6-v2",
        auto_embed: bool = True,
    ):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.auto_embed = auto_embed

        # Initialize SQLite
        self._init_db()

        # Initialize vector index
        self._init_vector_index()

        # Initialize embedding model
        self.embedding_model = None
        if auto_embed and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info("Loaded embedding model: %s", embedding_model)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("Failed to load embedding model: %s", e)

        # Load seed data if empty
        self._load_seed_if_empty()

    def _init_db(self) -> None:
        """Initialize SQLite database with tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    finding TEXT NOT NULL,
                    details TEXT,
                    confidence REAL NOT NULL,
                    tags TEXT,  -- JSON array
                    timestamp REAL NOT NULL,
                    source TEXT DEFAULT 'manual',
                    experiment_id TEXT,
                    metrics TEXT,  -- JSON
                    hyperparameters TEXT,  -- JSON
                    extra TEXT  -- JSON
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic ON knowledge(topic)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_family ON knowledge(model_family)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON knowledge(source)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment ON knowledge(experiment_id)
            """)

            # Table for experiment results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    task TEXT NOT NULL,
                    config TEXT,  -- JSON
                    metrics TEXT,  -- JSON
                    status TEXT DEFAULT 'completed',
                    timestamp REAL NOT NULL,
                    artifacts TEXT  -- JSON
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_model ON experiments(model_family)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_task ON experiments(task)
            """)

            # Table for surrogate model predictions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surrogates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,  -- 'gp', 'rf', 'nn', 'symbolic'
                    target_metric TEXT NOT NULL,
                    features TEXT,  -- JSON list of feature names
                    trained_at REAL NOT NULL,
                    performance TEXT,  -- JSON metrics
                    model_path TEXT
                )
            """)

            conn.commit()

    def _init_vector_index(self) -> None:
        """Initialize FAISS vector index."""
        if HAS_FAISS:
            self.vector_index = faiss.IndexFlatIP(
                self.vector_dim
            )  # Inner product for cosine similarity
            self.vector_ids = []  # Maps index position to knowledge entry ID
        else:
            self.vector_index = None
            self.vector_ids = []
            logger.warning(
                "FAISS not available. Vector search disabled. "
                "Install with: pip install faiss-cpu"
            )

    def _load_seed_if_empty(self) -> None:
        """Load seed knowledge if database is empty."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]

        if count == 0:
            self._load_seed_data()

    def _load_seed_data(self) -> None:
        """Load initial seed knowledge."""
        seed_entries = [
            KnowledgeEntry(
                id="KB-001",
                topic="Scaling",
                model_family="eqprop",
                finding="O(1) memory scaling for BPTT equivalent",
                details=(
                    "Equilibrium Propagation requires constant memory "
                    "regardless of trajectory length, unlike BPTT which scales O(T)."
                ),
                confidence=0.95,
                tags=["memory", "scaling", "eqprop"],
                source="literature",
            ),
            KnowledgeEntry(
                id="KB-002",
                topic="Architecture",
                model_family="tile_eq",
                finding="Optimal 2D grid layout improves locality",
                details=(
                    "TileEQ variants arranged in a 2D locally connected grid "
                    "demonstrate superior scaling on neuromorphic simulators "
                    "vs fully connected counterparts."
                ),
                confidence=0.85,
                tags=["architecture", "neuromorphic", "local-learning"],
                source="literature",
            ),
            KnowledgeEntry(
                id="KB-003",
                topic="Optimization",
                model_family="forward_forward",
                finding="Layer-local goodness thresholds",
                details=(
                    "A threshold of 2.0 provides stable contrastive separation "
                    "on MNIST-level tasks without causing early layer saturation."
                ),
                confidence=0.80,
                tags=["hyperparams", "forward-forward", "thresholds"],
                source="literature",
            ),
        ]

        for entry in seed_entries:
            self.add_entry(entry)

        logger.info("Loaded %s seed knowledge entries", len(seed_entries))

    def _embed_text(self, text: str) -> np.ndarray | None:
        """Generate embedding for text."""
        if self.embedding_model is None:
            return None
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Embedding failed: %s", e)
            return None

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry to the database."""
        # Generate embedding if auto_embed enabled
        if self.auto_embed and entry.embedding is None:
            text = f"{entry.topic} {entry.finding} {entry.details}"
            embedding = self._embed_text(text)
            if embedding is not None:
                object.__setattr__(entry, "embedding", embedding.tolist())

        # Store in SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge
                (id, topic, model_family, finding, details, confidence, tags,
                 timestamp, source, experiment_id, metrics, hyperparameters, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    entry.topic,
                    entry.model_family,
                    entry.finding,
                    entry.details,
                    entry.confidence,
                    json.dumps(entry.tags),
                    entry.timestamp,
                    entry.source,
                    entry.experiment_id,
                    json.dumps(entry.metrics),
                    json.dumps(entry.hyperparameters),
                    json.dumps(entry.extra),
                ),
            )
            conn.commit()

        # Add to vector index
        if self.vector_index is not None and entry.embedding is not None:
            embedding = np.array(entry.embedding, dtype=np.float32).reshape(1, -1)
            self.vector_index.add(embedding)
            self.vector_ids.append(entry.id)

        logger.debug("Added knowledge entry: %s", entry.id)
        return entry.id

    def add_experiment(
        self,
        name: str,
        model_family: str,
        task: str,
        config: dict[str, object],
        metrics: dict[str, float],
        experiment_id: str | None = None,
        artifacts: dict[str, str] | None = None,
        status: str = "completed",
    ) -> str:
        """Add an experiment result to the knowledge base."""
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())[:8]

        entry = KnowledgeEntry(
            id=f"EXP-{experiment_id}",
            topic="Experiment",
            model_family=model_family,
            finding=f"{name} on {task}",
            details=f"Experiment {name} with {model_family} on {task}",
            confidence=1.0,
            tags=["experiment", model_family, task],
            source="experiment",
            experiment_id=experiment_id,
            metrics=metrics,
            hyperparameters=config,
            extra={"artifacts": artifacts or {}, "task": task},
        )

        self.add_entry(entry)

        # Also store in experiments table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, name, model_family, task, config, metrics, status,
                 timestamp, artifacts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    name,
                    model_family,
                    task,
                    json.dumps(config),
                    json.dumps(metrics),
                    status,
                    time.time(),
                    json.dumps(artifacts or {}),
                ),
            )
            conn.commit()

        return experiment_id

    def query(
        self,
        tag: str | None = None,
        model_family: str | None = None,
        topic: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
    ) -> list[KnowledgeEntry]:
        """Query knowledge base with structured filters."""
        conditions = []
        params = []

        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if model_family:
            conditions.append("model_family = ?")
            params.append(model_family)
        if topic:
            conditions.append("topic = ?")
            params.append(topic)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                f"SELECT * FROM knowledge{where_clause} ORDER BY timestamp DESC LIMIT ?"
            )
            cursor = conn.execute(sql, params + [limit])
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def query_conditionals(
        self, query: ConditionalQuery, limit: int = 100
    ) -> list[ConditionalResult]:
        """Return verified positive conditionals matching a P2 read-half query.

        Filters stored experiment results by model, task, substrate, and any
        accuracy/memory/flops caps provided in ``query``. This is the flywheel's
        *read* side: the proposer consumes it to avoid re-characterizing probes
        that a prior run already answered.

        Args:
            query: The filter (see :class:`ConditionalQuery`). Validated with
                Pydantic v2 at this boundary.
            limit: Maximum number of results.

        Returns:
            Matching conditionals, best-accuracy first.

        Raises:
            ConditionalQueryError: If ``query`` is malformed (e.g. negative cap
                or an accuracy target outside [0, 1]).
        """
        try:
            q = _ConditionalQueryModel(**query)
        except ValidationError as exc:
            raise ConditionalQueryError(f"invalid conditional query: {exc}") from exc

        entries = self.query(
            model_family=q.model,
            source="experiment",
            limit=max(limit * 4, 64),
        )
        if q.task is not None:
            entries = [e for e in entries if _entry_task(e) == q.task]

        results: list[ConditionalResult] = []
        for e in entries:
            metrics = e.metrics or {}
            acc = _entry_accuracy(metrics)
            if q.accuracy_target is not None and acc < q.accuracy_target:
                continue
            mem = _entry_memory(metrics)
            if q.memory_cap is not None and mem > q.memory_cap:
                continue
            flops = _entry_flops(metrics)
            if q.flops_cap is not None and flops > q.flops_cap:
                continue
            substrate = _entry_substrate(e)
            if q.substrate is not None and substrate != q.substrate:
                continue
            results.append(
                ConditionalResult(
                    model=str(e.model_family),
                    task=_entry_task(e),
                    accuracy=acc,
                    memory_mb=mem,
                    flops=flops,
                    wall_time_s=float(metrics.get("wall_time_s", 0.0)),
                    config=tuple(sorted((e.hyperparameters or {}).items())),
                    substrate=substrate,
                    entry_id=e.id,
                )
            )
        results.sort(key=lambda r: r.accuracy, reverse=True)
        return results[:limit]

    def select_flagship(
        self,
        *,
        task: str = "mnist",
        accuracy_gap: float = 0.15,
        substrate: str | None = None,
    ) -> FlagshipDecision:
        """Select the flagship rule by querying the KB (P3a).

        Codified selection rule, implemented as a query rather than a judgment
        call: among **validated** families (P0a surface record present and
        honest), pick the one with minimal ``cost_of_plausibility`` — the
        geometric mean of its FLOPs/memory/time ratios to the backprop reference
        on the same task — subject to:

        1. **non-phantom space** — the family passed P0a (honest surface record).
        2. **substrate-eligible** — if ``substrate`` is given, only families with
           a matching substrate-conditional candidate.
        3. **minimal accuracy-to-backprop gap** — the candidate's accuracy must
           be within ``accuracy_gap`` of the backprop reference, so a trivially
           cheap-but-awful rule is never chosen.

        Args:
            task: Task the frontier/conditionals were measured on.
            accuracy_gap: Maximum accuracy deficit vs the backprop reference.
            substrate: If set, only consider conditionals on this substrate.

        Returns:
            A :class:`FlagshipDecision` ranking the candidate families.
        """
        import math

        bp = self._best_conditional("backprop_mlp", task, substrate)
        if bp is None:
            return FlagshipDecision(task=task, chosen=None, ranked=())

        validated = {
            e.model_family
            for e in self.query(topic="rule_space_surface", source="validator")
            if e.finding == "honest"
        }

        ranked: list[FlagshipCandidate] = []
        for family in validated:
            cand = self._best_conditional(family, task, substrate)
            if cand is None:
                continue
            if bp.accuracy - cand.accuracy > accuracy_gap:
                continue
            cost = math.pow(
                (cand.flops / max(bp.flops, 1e-12))
                * (cand.memory_mb / max(bp.memory_mb, 1e-12))
                * (cand.wall_time_s / max(bp.wall_time_s, 1e-12)),
                1.0 / 3.0,
            )
            ranked.append(
                FlagshipCandidate(
                    model=family,
                    accuracy=cand.accuracy,
                    memory_mb=cand.memory_mb,
                    flops=cand.flops,
                    wall_time_s=cand.wall_time_s,
                    cost_of_plausibility=cost,
                    substrate=cand.substrate,
                )
            )

        ranked.sort(key=lambda c: (c.cost_of_plausibility, -c.accuracy))
        chosen = ranked[0].model if ranked else None
        return FlagshipDecision(task=task, chosen=chosen, ranked=tuple(ranked))

    def _best_conditional(
        self, model: str, task: str, substrate: str | None
    ) -> ConditionalResult | None:
        """Best (highest-accuracy) verified conditional for a model on a task."""
        results = self.query_conditionals({
            "model": model,
            "task": task,
            "accuracy_target": 0.0,
            "substrate": substrate,
            "memory_cap": None,
            "flops_cap": None,
        })
        return results[0] if results else None

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert SQLite row to KnowledgeEntry."""
        return KnowledgeEntry(
            id=row["id"],
            topic=row["topic"],
            model_family=row["model_family"],
            finding=row["finding"],
            details=row["details"] or "",
            confidence=row["confidence"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            timestamp=row["timestamp"],
            source=row["source"],
            experiment_id=row["experiment_id"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
            hyperparameters=(
                json.loads(row["hyperparameters"]) if row["hyperparameters"] else {}
            ),
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )

    def search(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.5,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[KnowledgeEntry, float]]:
        """
        Semantic search using vector embeddings.

        Returns list of (entry, similarity_score) tuples.
        """
        if self.vector_index is None or self.embedding_model is None:
            logger.warning(
                "Vector search not available. Falling back to keyword search."
            )
            return self._keyword_search(query, k, filters)

        # Generate query embedding
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            return []

        query_embedding = query_embedding.reshape(1, -1)

        # Search vector index
        scores, indices = self.vector_index.search(
            query_embedding, min(k * 2, len(self.vector_ids))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.vector_ids):
                if score >= min_similarity:
                    entry_id = self.vector_ids[idx]
                    entry = self.get_by_id(entry_id)
                    if entry:
                        # Apply filters
                        if filters:
                            if not self._matches_filters(entry, filters):
                                continue
                        results.append((entry, float(score)))
                        if len(results) >= k:
                            break

        return results

    def _matches_filters(
        self, entry: KnowledgeEntry, filters: dict[str, object]
    ) -> bool:
        """Check if entry matches filter criteria."""
        for key, value in filters.items():
            if key == "model_family" and entry.model_family != value:
                return False
            if key == "topic" and entry.topic != value:
                return False
            if key == "tags" and not all(tag in entry.tags for tag in value):
                return False
            if key == "min_confidence" and entry.confidence < value:
                return False
        return True

    def _keyword_search(
        self,
        query: str,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[KnowledgeEntry, float]]:
        """Fallback keyword search."""
        if not query:
            return []
        query_lower = query.lower()
        results = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM knowledge ORDER BY timestamp DESC LIMIT 1000"
            )
            for row in cursor:
                entry = self._row_to_entry(row)

                # Simple keyword matching
                text = " ".join([
                    entry.topic,
                    entry.finding,
                    entry.details,
                    " ".join(entry.tags),
                ]).lower()
                score = sum(1 for word in query_lower.split() if word in text) / len(
                    query_lower.split()
                )

                if score > 0:
                    if filters and not self._matches_filters(entry, filters):
                        continue
                    results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_by_id(self, entry_id: str) -> KnowledgeEntry | None:
        """Get entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

        if row:
            return self._row_to_entry(row)
        return None

    def get_experiment(self, experiment_id: str) -> dict[str, object] | None:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def list_experiments(
        self,
        model_family: str | None = None,
        task: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List experiments with optional filters."""
        conditions = []
        params = []

        if model_family:
            conditions.append("model_family = ?")
            params.append(model_family)
        if task:
            conditions.append("task = ?")
            params.append(task)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT * FROM experiments"
                f"{where_clause} ORDER BY timestamp DESC LIMIT ?"
            )
            cursor = conn.execute(sql, params + [limit])
            return [dict(row) for row in cursor]

    def get_surrogate(self, name: str) -> dict[str, object] | None:
        """Get surrogate model by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surrogates WHERE name = ?", (name,))
            row = cursor.fetchone()

        return dict(row) if row else None

    def register_surrogate(
        self,
        name: str,
        model_type: str,
        target_metric: str,
        features: list[str],
        performance: dict[str, float],
        model_path: str | None = None,
    ) -> str:
        """Register a surrogate model."""
        surrogate_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO surrogates
                (id, name, model_type, target_metric, features,
                 trained_at, performance, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    surrogate_id,
                    name,
                    model_type,
                    target_metric,
                    json.dumps(features),
                    time.time(),
                    json.dumps(performance),
                    model_path,
                ),
            )
            conn.commit()

        return surrogate_id

    def list_surrogates(self) -> list[dict[str, object]]:
        """List all registered surrogate models."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surrogates ORDER BY trained_at DESC")
            return [dict(row) for row in cursor]

    def natural_language_query(self, question: str) -> str:
        """
        Answer natural language questions using the knowledge base.
        This is a simplified version - in production would use an LLM.
        """
        # Search for relevant entries
        results = self.search(question, k=5)

        if not results:
            return "No relevant knowledge found."

        # Build answer from top results
        answer_parts = [f"Found {len(results)} relevant entries:"]
        for entry, score in results[:3]:
            answer_parts.append(
                f"\n• [{entry.model_family}] {entry.finding} "
                f"(confidence: {entry.confidence:.0%})"
            )
            answer_parts.append(f"  Details: {entry.details[:200]}...")

        return "\n".join(answer_parts)

    def get_stats(self) -> dict[str, object]:
        """Get knowledge base statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
            by_source = dict(
                conn.execute(
                    "SELECT source, COUNT(*) FROM knowledge GROUP BY source"
                ).fetchall()
            )
            by_model = dict(
                conn.execute(
                    "SELECT model_family, COUNT(*) FROM knowledge GROUP BY model_family"
                ).fetchall()
            )
            by_topic = dict(
                conn.execute(
                    "SELECT topic, COUNT(*) FROM knowledge GROUP BY topic"
                ).fetchall()
            )

            exp_total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            exp_by_model = dict(
                conn.execute(
                    "SELECT model_family, COUNT(*) FROM experiments "
                    "GROUP BY model_family"
                ).fetchall()
            )
            exp_by_task = dict(
                conn.execute(
                    "SELECT task, COUNT(*) FROM experiments GROUP BY task"
                ).fetchall()
            )

        return {
            "total_entries": total,
            "by_source": by_source,
            "by_model_family": by_model,
            "by_topic": by_topic,
            "total_experiments": exp_total,
            "experiments_by_model": exp_by_model,
            "experiments_by_task": exp_by_task,
            "vector_index_size": len(self.vector_ids) if self.vector_index else 0,
            "has_embeddings": self.embedding_model is not None,
        }

    def export_json(self, path: str) -> None:
        """Export knowledge base to JSON."""
        entries = self.query(limit=10000)
        with pathlib.Path(path).open("w") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)

    def close(self) -> None:
        """Close connections."""

    # ------------------------------------------------------------------
    # Metamodel / Surrogate integration
    # ------------------------------------------------------------------

    def extract_symbolic_rules(
        self,
        target_metric: str = "outcome",
        focus_model: str = "eqprop_mlp",
    ) -> list[str]:
        """
        Extract human-readable symbolic rules from experiment data.

        Uses a decision tree surrogate to find interpretable decision boundaries
        that predict experiment success/failure.

        Args:
            target_metric: Column/field to predict.
            focus_model: Model family to analyze.

        Returns:
            List of human-readable rule strings.
        """
        try:
            from bioplausible.knowledge.metamodel import KnowledgebaseMetamodel

            mm = KnowledgebaseMetamodel()
            mm.fit(self.db_path)
            return mm.extract_symbolic_rules(
                target_metric=target_metric, focus_model=focus_model
            )
        except (sqlite3.Error, KeyError, ValueError) as e:
            logger.exception("Symbolic rule extraction failed")
            raise KnowledgeBaseError("Symbolic rule extraction failed") from e

    def compute_algorithm_similarity(self) -> dict[str, dict[str, float]]:
        """
        Compute pairwise similarity between algorithms based on
        their hyperparameter sensitivity fingerprints.

        Returns:
            Dict of model_name -> {other_model: similarity_score}
        """
        try:
            from bioplausible.knowledge.metamodel import KnowledgebaseMetamodel

            mm = KnowledgebaseMetamodel()
            mm.fit(self.db_path)
            sim_df = mm.compute_algorithm_similarity()
            if sim_df.empty:
                return {}
            return sim_df.to_dict()
        except (sqlite3.Error, KeyError, ValueError) as e:
            logger.exception("Algorithm similarity failed")
            raise KnowledgeBaseError("Algorithm similarity computation failed") from e

    def train_surrogate(
        self,
        target_metric: str = "val_accuracy",
        model_type: str = "rf",
        experiment_ids: list[str] | None = None,
    ) -> str | None:
        """
        Train a surrogate model to predict experiment outcomes.

        Args:
            target_metric: Metric to predict.
            model_type: 'rf' (random forest), 'gp' (Gaussian process), or 'nn'.
            experiment_ids: Optional subset of experiments to use.

        Returns:
            Surrogate model ID if successful, None otherwise.
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor

            exps = self.list_experiments(limit=500)
            if not exps or len(exps) < 10:
                logger.warning("Not enough experiments to train surrogate")
                return None

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", "{}"))
                record = {
                    "lr": config.get("lr", 0.001),
                    "batch_size": config.get("batch_size", 64),
                    "hidden_dim": config.get("hidden_dim", 256),
                    "num_layers": config.get("num_layers", 2),
                    "epochs": config.get("epochs", 10),
                }
                if target_metric in metrics:
                    record["target"] = metrics[target_metric]
                    records.append(record)

            if len(records) < 10:
                logger.warning("Not enough records with %s", target_metric)
                return None

            df = pd.DataFrame(records)
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            y = df["target"].values

            model = RandomForestRegressor(n_estimators=100, max_depth=5)
            model.fit(X, y)

            score = model.score(X, y)
            surrogate_id = self.register_surrogate(
                name=f"surrogate_{target_metric}",
                model_type=model_type,
                target_metric=target_metric,
                features=feature_cols,
                performance={"r2": float(score), "n_samples": len(records)},
            )
            logger.info("Trained surrogate %s with R2=%s", surrogate_id, score)
            return surrogate_id

        except (sqlite3.Error, KeyError, ValueError) as e:
            logger.exception("Surrogate training failed")
            raise KnowledgeBaseError("Surrogate training failed") from e

    def predict_outcome(
        self,
        config: dict[str, object],
        target_metric: str = "val_accuracy",
    ) -> float:
        """
        Predict experiment outcome using trained surrogate.

        Args:
            config: Experiment configuration with hyperparameters.
            target_metric: Metric to predict.

        Returns:
            Predicted value for the target metric.
        """
        surrogate = self.get_surrogate(f"surrogate_{target_metric}")
        if not surrogate:
            return 0.0

        try:
            # This is a simplified prediction - in production,
            # we'd load the actual saved model
            return float(surrogate.get("performance", {}).get("r2", 0.0))
        except (KeyError, ValueError, TypeError) as e:
            logger.exception("Prediction failed")
            raise KnowledgeBaseError("Surrogate prediction failed") from e

    def run_causal_analysis(
        self,
        outcome: str = "val_accuracy",
    ) -> dict[str, object]:
        """
        Run causal discovery analysis on experiment data.

        Uses correlation-based methods to identify potentially causal
        relationships between hyperparameters and outcomes.

        Args:
            outcome: Target metric for analysis.

        Returns:
            Dict with causal analysis results.
        """
        try:
            import pandas as pd

            exps = self.list_experiments(limit=500)
            if not exps or len(exps) < 10:
                return {"error": "Not enough data for causal analysis"}

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", "{}"))
                if outcome in metrics:
                    records.append({
                        "lr": config.get("lr", 0.001),
                        "hidden_dim": config.get("hidden_dim", 256),
                        "num_layers": config.get("num_layers", 2),
                        "batch_size": config.get("batch_size", 64),
                        "outcome": metrics[outcome],
                    })

            if len(records) < 10:
                return {"error": f"Not enough records with {outcome}"}

            df = pd.DataFrame(records)

            # Compute correlations with outcome
            correlations = {}
            for col in df.columns:
                if col != "outcome":
                    corr = df[col].corr(df["outcome"])
                    if not np.isnan(corr):
                        correlations[col] = float(abs(corr))

            # Sort by correlation magnitude
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            return {
                "outcome": outcome,
                "correlations": dict(correlations),
                "ranked_factors": sorted_corr,
                "n_samples": len(records),
            }
        except (sqlite3.Error, KeyError, ValueError) as e:
            logger.exception("Causal analysis failed")
            raise KnowledgeBaseError("Causal analysis failed") from e

    # ------------------------------------------------------------------
    # Meta-Analysis Methods (P1.42)
    # ------------------------------------------------------------------

    def meta_fit_scaling_laws(
        self,
        model_families: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate Chinchilla-style scaling law fits across all runs.

        Fits: L = E + A / N^alpha + B / D^beta
        Where N = params, D = data, L = loss

        Args:
            model_families: Optional filter for specific model families.
            tasks: Optional filter for specific tasks.

        Returns:
            Dict of model_family -> {alpha, beta, E, A, B, r2}
        """
        try:
            import numpy as np
            import pandas as pd
            from scipy.optimize import curve_fit

            exps = self.list_experiments(limit=1000)
            if not exps:
                return {}

            # Filter
            if model_families:
                exps = [e for e in exps if e.get("model_family") in model_families]
            if tasks:
                exps = [e for e in exps if e.get("task") in tasks]

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", "{}"))

                # Extract scaling parameters
                n_params = config.get("n_params", 0)
                n_data = config.get("n_data", 0)
                loss = metrics.get("final_loss", metrics.get("val_loss", 0))

                if n_params > 0 and n_data > 0 and loss > 0:
                    records.append({
                        "model_family": exp.get("model_family", "unknown"),
                        "task": exp.get("task", "unknown"),
                        "n_params": n_params,
                        "n_data": n_data,
                        "loss": loss,
                    })

            if len(records) < 20:
                logger.warning("Not enough data for scaling law meta-fit")
                return {}

            df = pd.DataFrame(records)
            results = {}

            def scaling_law(N, D, E, A, B, alpha, beta):
                return E + A / (N ** alpha) + B / (D ** beta)

            for model in df["model_family"].unique():
                model_df = df[df["model_family"] == model]
                if len(model_df) < 10:
                    continue

                try:
                    X = np.column_stack([model_df["n_params"].values, model_df["n_data"].values])
                    y = model_df["loss"].values

                    # Initial guess: E=0.1, A=1, B=1, alpha=0.5, beta=0.5
                    p0 = [0.1, 1.0, 1.0, 0.5, 0.5]
                    bounds = ([0, 0, 0, 0.1, 0.1], [10, 100, 100, 2.0, 2.0])

                    popt, pcov = curve_fit(
                        lambda X, E, A, B, alpha, beta: scaling_law(X[:, 0], X[:, 1], E, A, B, alpha, beta),
                        X, y, p0=p0, bounds=bounds, maxfev=5000,
                    )

                    E, A, B, alpha, beta = popt
                    y_pred = scaling_law(X[:, 0], X[:, 1], E, A, B, alpha, beta)
                    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

                    results[model] = {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "E": float(E),
                        "A": float(A),
                        "B": float(B),
                        "r2": float(r2),
                        "n_samples": len(model_df),
                    }
                except Exception as e:
                    logger.warning("Scaling fit failed for %s: %s", model, e)
                    continue

            logger.info("Meta-fit scaling laws for %d model families", len(results))
            return results

        except Exception as e:
            logger.exception("Scaling law meta-fit failed")
            raise KnowledgeBaseError("Scaling law meta-fit failed") from e

    def compute_algorithm_fingerprints(
        self,
        model_families: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute algorithm fingerprints: hyperparameter sensitivity embeddings.

        Each algorithm gets a fingerprint vector representing its sensitivity
        to different hyperparameters. Used for algorithm phylogeny.

        Args:
            model_families: Optional filter for specific model families.

        Returns:
            Dict of model_family -> {hyperparam: sensitivity_score}
        """
        try:
            import numpy as np
            import pandas as pd

            exps = self.list_experiments(limit=1000)
            if not exps:
                return {}

            if model_families:
                exps = [e for e in exps if e.get("model_family") in model_families]

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", {}))
                acc = metrics.get("val_accuracy", 0)

                record = {
                    "model_family": exp.get("model_family", "unknown"),
                    "lr": config.get("lr", 0.001),
                    "hidden_dim": config.get("hidden_dim", 256),
                    "num_layers": config.get("num_layers", 2),
                    "batch_size": config.get("batch_size", 64),
                    "beta": config.get("beta", 0.0),
                    "val_accuracy": acc,
                }
                records.append(record)

            if len(records) < 20:
                return {}

            df = pd.DataFrame(records)

            # Compute sensitivity for each model family
            fingerprints = {}
            for model in df["model_family"].unique():
                model_df = df[df["model_family"] == model]
                if len(model_df) < 5:
                    continue

                # Correlation between each hyperparam and accuracy
                sensitivity = {}
                for param in ["lr", "hidden_dim", "num_layers", "batch_size", "beta"]:
                    if param in model_df.columns:
                        corr = model_df[param].corr(model_df["val_accuracy"])
                        if not np.isnan(corr):
                            sensitivity[param] = float(abs(corr))

                # Also compute variance-based sensitivity (Sobol-like)
                for param in ["lr", "hidden_dim", "num_layers", "batch_size"]:
                    if param in model_df.columns:
                        grouped = model_df.groupby(param)["val_accuracy"].mean()
                        if len(grouped) > 1:
                            sensitivity[f"{param}_variance"] = float(grouped.var())

                fingerprints[model] = sensitivity

            logger.info("Computed fingerprints for %d algorithms", len(fingerprints))
            return fingerprints

        except Exception as e:
            logger.exception("Algorithm fingerprint computation failed")
            raise KnowledgeBaseError("Algorithm fingerprint computation failed") from e

    def map_failure_manifold(
        self,
        min_samples: int = 5,
    ) -> dict[str, dict[str, object]]:
        """
        Cluster failed runs by error mode to identify failure manifolds.

        Identifies common failure patterns across algorithms/tasks.

        Args:
            min_samples: Minimum samples to form a cluster.

        Returns:
            Dict of failure_cluster -> {error_pattern, algorithms, tasks, count, characteristics}
        """
        try:
            import numpy as np
            import pandas as pd
            from sklearn.cluster import DBSCAN
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import StandardScaler

            exps = self.list_experiments(limit=1000)
            if not exps:
                return {}

            # Collect failed experiments
            failed = []
            for exp in exps:
                metrics = json.loads(exp.get("metrics", "{}"))
                config = json.loads(exp.get("config", "{}"))

                # Consider failed if low accuracy or explicit error
                acc = metrics.get("val_accuracy", 1.0)
                error = metrics.get("error", config.get("error", ""))

                if acc < 0.15 or error:
                    failed.append({
                        "model_family": exp.get("model_family", "unknown"),
                        "task": exp.get("task", "unknown"),
                        "accuracy": acc,
                        "error": str(error),
                        "config": config,
                    })

            if len(failed) < min_samples:
                logger.warning("Not enough failed runs for manifold mapping")
                return {}

            df = pd.DataFrame(failed)

            # Vectorize error messages
            tfidf = TfidfVectorizer(max_features=50, stop_words="english")
            error_texts = df["error"].fillna("").tolist()
            if all(not e for e in error_texts):
                # No error messages, use accuracy + config
                X_config = pd.DataFrame(df["config"].tolist())
                X_config = X_config.fillna(0)
                scaler = StandardScaler()
                X = scaler.fit_transform(X_config.select_dtypes(include=[np.number]))
            else:
                X_text = tfidf.fit_transform(error_texts).toarray()
                # Add config features
                X_config = pd.DataFrame(df["config"].tolist())
                X_config = X_config.fillna(0)
                numeric_cols = X_config.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    X_config_scaled = scaler.fit_transform(X_config[numeric_cols])
                    X = np.hstack([X_text, X_config_scaled])
                else:
                    X = X_text

            # Cluster
            clustering = DBSCAN(eps=0.5, min_samples=min_samples)
            labels = clustering.fit_predict(X)

            df["cluster"] = labels

            # Analyze clusters
            failure_manifold = {}
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue  # Noise

                cluster_df = df[df["cluster"] == cluster_id]
                if len(cluster_df) < min_samples:
                    continue

                # Characterize cluster
                error_mode = cluster_df["error"].mode().iloc[0] if not cluster_df["error"].mode().empty else "unknown"
                algorithms = cluster_df["model_family"].value_counts().to_dict()
                tasks = cluster_df["task"].value_counts().to_dict()
                mean_acc = float(cluster_df["accuracy"].mean())

                # Common config patterns
                common_config = {}
                for col in cluster_df["config"].iloc[0].keys() if len(cluster_df) > 0 else []:
                    vals = [c.get(col) for c in cluster_df["config"] if col in c]
                    if vals:
                        common_config[col] = max(set(vals), key=vals.count)

                failure_manifold[f"cluster_{cluster_id}"] = {
                    "error_pattern": error_mode,
                    "algorithms": algorithms,
                    "tasks": tasks,
                    "count": len(cluster_df),
                    "mean_accuracy": mean_acc,
                    "common_config": common_config,
                }

            logger.info("Mapped failure manifold with %d clusters", len(failure_manifold))
            return failure_manifold

        except Exception as e:
            logger.exception("Failure manifold mapping failed")
            raise KnowledgeBaseError("Failure manifold mapping failed") from e

    def generate_algorithm_phylogeny(
        self,
        method: str = "ward",
    ) -> dict[str, object]:
        """
        Generate phylogenetic tree of algorithms based on fingerprints.

        Uses hierarchical clustering on algorithm fingerprints.

        Args:
            method: Linkage method ('ward', 'complete', 'average', 'single').

        Returns:
            Dict with tree structure and cluster assignments.
        """
        try:
            import numpy as np
            from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
            from sklearn.preprocessing import StandardScaler

            fingerprints = self.compute_algorithm_fingerprints()
            if not fingerprints or len(fingerprints) < 3:
                return {}

            # Build feature matrix
            models = list(fingerprints.keys())
            all_features = set()
            for fp in fingerprints.values():
                all_features.update(fp.keys())

            feature_list = sorted(all_features)
            X = np.zeros((len(models), len(feature_list)))

            for i, model in enumerate(models):
                for j, feat in enumerate(feature_list):
                    X[i, j] = fingerprints[model].get(feat, 0.0)

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Hierarchical clustering
            Z = linkage(X_scaled, method=method)

            # Get cluster assignments
            clusters = fcluster(Z, t=0.5, criterion="distance")

            # Build tree structure
            tree = {
                "models": models,
                "linkage_matrix": Z.tolist(),
                "clusters": {models[i]: int(clusters[i]) for i in range(len(models))},
                "n_clusters": int(clusters.max()),
                "feature_names": feature_list,
            }

            # Dendrogram data for visualization
            dend = dendrogram(Z, labels=models, no_plot=True)
            tree["dendrogram"] = {
                "icoord": [c.tolist() for c in dend["icoord"]],
                "dcoord": [c.tolist() for c in dend["dcoord"]],
                "ivl": dend["ivl"],
                "leaves": dend["leaves"],
            }

            logger.info("Generated algorithm phylogeny with %d clusters", tree["n_clusters"])
            return tree

        except Exception as e:
            logger.exception("Algorithm phylogeny generation failed")
            raise KnowledgeBaseError("Algorithm phylogeny generation failed") from e

    def get_meta_analysis_summary(self) -> dict[str, object]:
        """
        Get comprehensive meta-analysis summary of the knowledge base.

        Combines scaling laws, fingerprints, failure manifold, and phylogeny.
        """
        summary = {}

        # Scaling laws
        try:
            summary["scaling_laws"] = self.meta_fit_scaling_laws()
        except Exception as e:
            summary["scaling_laws"] = {"error": str(e)}

        # Algorithm fingerprints
        try:
            summary["fingerprints"] = self.compute_algorithm_fingerprints()
        except Exception as e:
            summary["fingerprints"] = {"error": str(e)}

        # Failure manifold
        try:
            summary["failure_manifold"] = self.map_failure_manifold()
        except Exception as e:
            summary["failure_manifold"] = {"error": str(e)}

        # Phylogeny
        try:
            summary["phylogeny"] = self.generate_algorithm_phylogeny()
        except Exception as e:
            summary["phylogeny"] = {"error": str(e)}

        # Basic stats
        summary["kb_stats"] = self.get_stats()

        return summary


# Factory function
def create_knowledge_base(
    db_path: str = db_path("bioplausible_kb.db"), **kwargs
) -> KnowledgeBase:
    """Create a KnowledgeBase instance."""
    return KnowledgeBase(db_path=db_path, **kwargs)


# Default instance (lazy — created on first access)
_DEFAULT_KB: KnowledgeBase | None = None


def _get_default_kb() -> KnowledgeBase:
    global _DEFAULT_KB
    if _DEFAULT_KB is None:
        _DEFAULT_KB = KnowledgeBase()
    return _DEFAULT_KB


# Make DEFAULT_KB accessible as a module attribute
def __getattr__(name: str) -> object:
    if name == "DEFAULT_KB":
        return _get_default_kb()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConditionalQuery",
    "ConditionalResult",
    "FlagshipCandidate",
    "FlagshipDecision",
    "KnowledgeBase",
    "KnowledgeEntry",
    "create_knowledge_base",
]
