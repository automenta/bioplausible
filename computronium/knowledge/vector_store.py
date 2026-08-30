"""
Vector store and embedding utilities for the knowledge base.

Provides FAISS-based semantic search and embedding generation.
"""

import json
import pathlib
import sqlite3
from dataclasses import dataclass

import numpy as np

from computronium.core._paths import db_path
from computronium.core.logging import get_logger

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
class VectorStoreConfig:
    """Configuration for the vector store."""

    vector_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    auto_embed: bool = True


class VectorStore:
    """
    FAISS-based vector store for semantic similarity search.

    Integrates with SQLite for metadata storage and FAISS for
    high-performance vector similarity search.
    """

    def __init__(
        self,
        db_path: str = db_path("computronium_kb.db"),
        config: VectorStoreConfig | None = None,
    ):
        self.db_path = db_path
        self.config = config or VectorStoreConfig()

        # Initialize vector index
        self._init_vector_index()

        # Initialize embedding model
        self.embedding_model = None
        if self.config.auto_embed and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info("Loaded embedding model: %s", self.config.embedding_model)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("Failed to load embedding model: %s", e)

    def _init_vector_index(self) -> None:
        """Initialize FAISS vector index."""
        if HAS_FAISS:
            self.vector_index = faiss.IndexFlatIP(self.config.vector_dim)
            self.vector_ids = []  # Maps index position to knowledge entry ID
        else:
            self.vector_index = None
            self.vector_ids = []
            logger.warning(
                "FAISS not available. Vector search disabled. "
                "Install with: pip install faiss-cpu"
            )

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

    def add_embedding(self, entry_id: str, embedding: list[float] | np.ndarray) -> None:
        """Add an embedding to the vector index."""
        if self.vector_index is None:
            return
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self.vector_index.add(emb)
        self.vector_ids.append(entry_id)

    def search(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.5,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Semantic search using vector embeddings.

        Returns list of (entry_id, similarity_score) tuples.
        """
        if self.vector_index is None or self.embedding_model is None:
            logger.warning(
                "Vector search not available. Falling back to keyword search."
            )
            return self._keyword_search(query, k, min_similarity)

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
                    results.append((entry_id, float(score)))
                    if len(results) >= k:
                        break

        return results

    def _keyword_search(
        self, query: str, k: int, min_similarity: float
    ) -> list[tuple[str, float]]:
        """Term-overlap fallback over the knowledge table when FAISS/embeddings
        are unavailable. Score is the fraction of query terms present in the
        entry's searchable text."""
        terms = query.lower().split()
        if not terms:
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT rowid, id, topic, model_family, finding, details, tags"
                    " FROM knowledge"
                ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("Keyword search failed: %s", e)
            return []
        results: list[tuple[int, str, float]] = []
        for rowid, entry_id, *text_fields in rows:
            text = " ".join(str(field) for field in text_fields if field).lower()
            score = sum(term in text for term in terms) / len(terms)
            if score >= min_similarity:
                results.append((rowid, entry_id, score))
        return [
            (entry_id, score)
            for _, entry_id, score in sorted(results, key=lambda r: (-r[2], -r[0]))
        ][:k][:k]

    def get_stats(self) -> dict[str, object]:
        """Get vector store statistics."""
        return {
            "vector_index_size": len(self.vector_ids) if self.vector_index else 0,
            "has_embeddings": self.embedding_model is not None,
            "embedding_model": self.config.embedding_model,
            "vector_dim": self.config.vector_dim,
        }

    def persist(self) -> None:
        """Persist vector index to disk."""
        if self.vector_index is not None:
            index_path = pathlib.Path(self.db_path).with_suffix(".faiss")
            faiss.write_index(self.vector_index, str(index_path))
            # Save vector_ids
            ids_path = pathlib.Path(self.db_path).with_suffix(".faiss_ids.json")
            with ids_path.open("w") as f:
                json.dump(self.vector_ids, f)

    def load_persisted(self) -> bool:
        """Load persisted vector index from disk."""
        index_path = pathlib.Path(self.db_path).with_suffix(".faiss")
        ids_path = pathlib.Path(self.db_path).with_suffix(".faiss_ids.json")

        if index_path.exists() and ids_path.exists() and HAS_FAISS:
            try:
                self.vector_index = faiss.read_index(str(index_path))
                with ids_path.open() as f:
                    self.vector_ids = json.load(f)
                logger.info(
                    "Loaded persisted vector index with %d vectors",
                    len(self.vector_ids),
                )
                return True
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("Failed to load persisted vector index: %s", e)
        return False


__all__ = [
    "HAS_FAISS",
    "HAS_SENTENCE_TRANSFORMERS",
    "VectorStore",
    "VectorStoreConfig",
]
