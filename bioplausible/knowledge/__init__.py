"""
Knowledge Package

Upgraded KnowledgeBase with SQLite + vector store (FAISS) for hybrid
structured + embedding search. Integrates surrogate models, symbolic
regression, and causal discovery.
"""

from bioplausible.knowledge.kb import (
    KnowledgeBase,
    KnowledgeEntry,
    create_knowledge_base,
)
from bioplausible.knowledge.seed import KNOWLEDGE_BASE_SEED


def __getattr__(name: str) -> object:
    """Lazy-access DEFAULT_KB to avoid SQLite at import time."""
    if name == "DEFAULT_KB":
        from bioplausible.knowledge.kb import _get_default_kb

        return _get_default_kb()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_KB",  # ruff: ignore[undefined-export] -- provided by __getattr__
    "KNOWLEDGE_BASE_SEED",
    "KnowledgeBase",
    "KnowledgeEntry",
    "create_knowledge_base",
]
