"""
Knowledge Package

Upgraded KnowledgeBase with SQLite + vector store (FAISS) for hybrid
structured + embedding search. Integrates surrogate models, symbolic
regression, and causal discovery.
"""

from computronium.knowledge.causal import CausalAnalyzer, CausalConfig
from computronium.knowledge.entries import (
    ConditionalQuery,
    ConditionalResult,
    FlagshipCandidate,
    FlagshipDecision,
    KnowledgeEntry,
)
from computronium.knowledge.kb import (
    KnowledgeBase,
    KnowledgeBaseConfig,
    create_knowledge_base,
)
from computronium.knowledge.query import QueryConfig, QueryEngine
from computronium.knowledge.seed import KNOWLEDGE_BASE_SEED
from computronium.knowledge.surrogate import SurrogateConfig, SurrogateManager
from computronium.knowledge.vector_store import VectorStore, VectorStoreConfig


def __getattr__(name: str) -> object:
    """Lazy-access DEFAULT_KB to avoid SQLite at import time."""
    if name == "DEFAULT_KB":
        from computronium.knowledge.kb import _get_default_kb

        return _get_default_kb()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CausalAnalyzer",
    "CausalConfig",
    "ConditionalQuery",
    "ConditionalResult",
    "DEFAULT_KB",  # ruff: ignore[undefined-export] -- provided by __getattr__
    "FlagshipCandidate",
    "FlagshipDecision",
    "KNOWLEDGE_BASE_SEED",
    "KnowledgeBase",
    "KnowledgeBaseConfig",
    "KnowledgeEntry",
    "QueryConfig",
    "QueryEngine",
    "SurrogateConfig",
    "SurrogateManager",
    "VectorStore",
    "VectorStoreConfig",
    "create_knowledge_base",
]
