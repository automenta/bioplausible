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
from dataclasses import dataclass

from computronium.core._paths import db_path
from computronium.core.exceptions import KnowledgeBaseError
from computronium.core.logging import get_logger
from computronium.knowledge.causal import CausalAnalyzer, CausalConfig
from computronium.knowledge.entries import (
    ConditionalQuery,
    ConditionalResult,
    FlagshipCandidate,
    FlagshipDecision,
    KnowledgeEntry,
)
from computronium.knowledge.query import QueryConfig, QueryEngine
from computronium.knowledge.surrogate import SurrogateConfig, SurrogateManager
from computronium.knowledge.vector_store import VectorStore, VectorStoreConfig

logger = get_logger()


@dataclass(frozen=True, slots=True)
class KnowledgeBaseConfig:
    """Configuration for KnowledgeBase."""

    db_path: str = db_path("computronium_kb.db")
    vector_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    auto_embed: bool = True
    default_limit: int = 100
    min_experiments: int = 10
    min_records: int = 10


class KnowledgeBase:  # ruff: ignore[too-many-public-methods]  # integrity-surface + conditional + flagship queries are all distinct public KB reads
    """
    Upgraded KnowledgeBase with SQLite + Vector Store.

    Features:
    - SQLite for structured queries (tags, model_family, metrics, etc.)
    - FAISS/Vector store for semantic similarity search
    - Surrogate model integration for predicting experiment outcomes
    - Symbolic regression for extracting analytical formulas
    - Causal discovery for identifying causal factors
    - Meta-analysis: scaling laws, fingerprints, failure manifolds, phylogeny
    """

    def __init__(self, config: KnowledgeBaseConfig | None = None):
        self.config = config or KnowledgeBaseConfig()

        # Initialize components
        self.query_engine = QueryEngine(
            QueryConfig(
                db_path=self.config.db_path,
                default_limit=self.config.default_limit,
            )
        )
        self.vector_store = VectorStore(
            db_path=self.config.db_path,
            config=VectorStoreConfig(
                vector_dim=self.config.vector_dim,
                embedding_model=self.config.embedding_model,
                auto_embed=self.config.auto_embed,
            ),
        )
        self.surrogate_manager = SurrogateManager(
            SurrogateConfig(
                db_path=self.config.db_path,
                min_experiments=self.config.min_experiments,
                min_records=self.config.min_records,
            )
        )
        self.causal_analyzer = CausalAnalyzer(
            CausalConfig(
                db_path=self.config.db_path,
                min_experiments=self.config.min_experiments,
                min_records=self.config.min_records,
            )
        )

        # Initialize SQLite
        self._init_db()

        # Load seed data if empty
        self._load_seed_if_empty()

    def _init_db(self) -> None:
        """Initialize SQLite database with tables."""
        with sqlite3.connect(self.config.db_path) as conn:
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

    def _load_seed_if_empty(self) -> None:
        """Load seed knowledge if database is empty."""
        with sqlite3.connect(self.config.db_path) as conn:
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

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry to the database."""
        # Generate embedding if auto_embed enabled
        if self.config.auto_embed and entry.embedding is None:
            text = f"{entry.topic} {entry.finding} {entry.details}"
            embedding = self.vector_store._embed_text(text)
            if embedding is not None:
                object.__setattr__(entry, "embedding", embedding.tolist())

        # Store in SQLite
        with sqlite3.connect(self.config.db_path) as conn:
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
        if self.vector_store.vector_index is not None and entry.embedding is not None:
            self.vector_store.add_embedding(entry.id, entry.embedding)

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
        with sqlite3.connect(self.config.db_path) as conn:
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

    # Delegate to query engine
    def query(self, **kwargs) -> list[KnowledgeEntry]:
        """Query knowledge base with structured filters."""
        return self.query_engine.query(**kwargs)

    def query_conditionals(
        self, query: ConditionalQuery, limit: int = 100
    ) -> list[ConditionalResult]:
        """Return verified positive conditionals matching a P2 read-half query."""
        return self.query_engine.query_conditionals(query, limit)

    def select_flagship(
        self,
        *,
        task: str = "mnist",
        accuracy_gap: float = 0.15,
        substrate: str | None = None,
    ) -> FlagshipDecision:
        """Select the flagship rule by querying the KB (P3a)."""
        return self.query_engine.select_flagship(
            task=task, accuracy_gap=accuracy_gap, substrate=substrate
        )

    def get_by_id(self, entry_id: str) -> KnowledgeEntry | None:
        """Get entry by ID."""
        return self.query_engine.get_by_id(entry_id)

    def get_experiment(self, experiment_id: str) -> dict[str, object] | None:
        """Get experiment by ID."""
        return self.query_engine.get_experiment(experiment_id)

    def list_experiments(
        self,
        model_family: str | None = None,
        task: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List experiments with optional filters."""
        return self.query_engine.list_experiments(model_family, task, limit)

    # Delegate to vector store
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
        results = self.vector_store.search(query, k, min_similarity, filters)

        # Convert entry IDs to KnowledgeEntry objects
        enriched_results = []
        for entry_id, score in results:
            entry = self.get_by_id(entry_id)
            if entry:
                # Apply filters if needed
                if filters and not self._matches_filters(entry, filters):
                    continue
                enriched_results.append((entry, score))

        return enriched_results

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
            if key == "min_confidence" and float(entry.confidence) < float(value):
                return False
        return True

    # Delegate to surrogate manager
    def get_surrogate(self, name: str) -> dict[str, object] | None:
        """Get surrogate model by name."""
        return self.surrogate_manager.get_surrogate(name)

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
        return self.surrogate_manager.register_surrogate(
            name, model_type, target_metric, features, performance, model_path
        )

    def list_surrogates(self) -> list[dict[str, object]]:
        """List all registered surrogate models."""
        return self.surrogate_manager.list_surrogates()

    def train_surrogate(
        self,
        target_metric: str = "val_accuracy",
        model_type: str = "rf",
        experiment_ids: list[str] | None = None,
    ) -> str | None:
        """Train a surrogate model to predict experiment outcomes."""
        return self.surrogate_manager.train_surrogate(
            target_metric, model_type, experiment_ids
        )

    def predict_outcome(
        self,
        config: dict[str, object],
        target_metric: str = "val_accuracy",
    ) -> float:
        """Predict experiment outcome using trained surrogate."""
        return self.surrogate_manager.predict_outcome(config, target_metric)

    # Delegate to causal analyzer
    def run_causal_analysis(
        self,
        outcome: str = "val_accuracy",
    ) -> dict[str, object]:
        """Run causal discovery analysis on experiment data."""
        return self.causal_analyzer.run_causal_analysis(outcome)

    def meta_fit_scaling_laws(
        self,
        model_families: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Aggregate Chinchilla-style scaling law fits across all runs."""
        return self.causal_analyzer.meta_fit_scaling_laws(model_families, tasks)

    def compute_algorithm_fingerprints(
        self,
        model_families: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute algorithm fingerprints: hyperparameter sensitivity embeddings."""
        return self.causal_analyzer.compute_algorithm_fingerprints(model_families)

    def map_failure_manifold(
        self,
        min_samples: int = 5,
    ) -> dict[str, dict[str, object]]:
        """Cluster failed runs by error mode to identify failure manifolds."""
        return self.causal_analyzer.map_failure_manifold(min_samples)

    def generate_algorithm_phylogeny(
        self,
        method: str = "ward",
    ) -> dict[str, object]:
        """Generate phylogenetic tree of algorithms based on fingerprints."""
        return self.causal_analyzer.generate_algorithm_phylogeny(method)

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

    def get_stats(self) -> dict[str, object]:
        """Get knowledge base statistics."""
        with sqlite3.connect(self.config.db_path) as conn:
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
            "vector_index_size": len(self.vector_store.vector_ids)
            if self.vector_store.vector_index
            else 0,
            "has_embeddings": self.vector_store.embedding_model is not None,
        }

    def export_json(self, path: str) -> None:
        """Export knowledge base to JSON."""
        entries = self.query(limit=10000)
        with pathlib.Path(path).open("w") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)

    def close(self) -> None:
        """Close connections."""
        self.vector_store.persist()

    # ------------------------------------------------------------------
    # Metamodel / Surrogate integration (legacy compatibility)
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
            from computronium.knowledge.metamodel import KnowledgebaseMetamodel

            mm = KnowledgebaseMetamodel()
            mm.fit(self.config.db_path)
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
            from computronium.knowledge.metamodel import KnowledgebaseMetamodel

            mm = KnowledgebaseMetamodel()
            mm.fit(self.config.db_path)
            sim_df = mm.compute_algorithm_similarity()
            if sim_df.empty:
                return {}
            return sim_df.to_dict()
        except (sqlite3.Error, KeyError, ValueError) as e:
            logger.exception("Algorithm similarity failed")
            raise KnowledgeBaseError("Algorithm similarity computation failed") from e


# Factory function
def create_knowledge_base(
    db_path: str = db_path("computronium_kb.db"), **kwargs
) -> KnowledgeBase:
    """Create a KnowledgeBase instance."""
    config = KnowledgeBaseConfig(db_path=db_path, **kwargs)
    return KnowledgeBase(config=config)


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
    "KnowledgeBaseConfig",
    "KnowledgeEntry",
    "create_knowledge_base",
]
