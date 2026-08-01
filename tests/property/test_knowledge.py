"""Hypothesis property-based tests for KnowledgeEntry serialization
(Sprint 5.5).

Laws:
  - from_dict(to_dict(entry)) == entry (round-trip, embedding excluded).
  - to_dict() never includes the embedding vector.
  - to_dict() is deterministic (same entry -> same dict, including a stable
    embedding when present).
"""

from hypothesis import given
from hypothesis import strategies as st

from bioplausible.knowledge.kb import KnowledgeEntry

text_strat = st.text(min_size=1, max_size=40, alphabet="abcdefghijklmnopqrstuvwxyz ")
source_strat = st.sampled_from(["manual", "experiment", "surrogate", "causal"])
id_strat = st.one_of(st.none(), st.text(min_size=1, max_size=16, alphabet="ABC123"))


@st.composite
def entry_strat(draw):
    """Generate a random KnowledgeEntry without an embedding."""
    return KnowledgeEntry(
        id=draw(id_strat) or "KB-x",
        topic=draw(text_strat),
        model_family=draw(text_strat),
        finding=draw(text_strat),
        details=draw(text_strat),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        tags=draw(
            st.lists(st.text(min_size=1, max_size=8, alphabet="abc"), max_size=4)
        ),
        timestamp=draw(st.floats(min_value=0.0, max_value=1e9, allow_nan=False)),
        source=draw(source_strat),
        experiment_id=draw(id_strat),
        metrics=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=8, alphabet="abc"),
                st.floats(allow_nan=False, allow_infinity=False),
                max_size=4,
            )
        ),
        hyperparameters=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=8, alphabet="abc"),
                st.one_of(st.integers(), st.floats(allow_nan=False), st.booleans()),
                max_size=4,
            )
        ),
        extra=draw(
            st.dictionaries(
                st.text(min_size=1, max_size=8, alphabet="abc"),
                st.one_of(st.text(max_size=8), st.integers()),
                max_size=4,
            )
        ),
    )


@given(entry=entry_strat())
def test_round_trip_preserves_entry(entry):
    """from_dict(to_dict(entry)) equals the original entry."""
    restored = KnowledgeEntry.from_dict(entry.to_dict())
    assert restored == entry


@given(entry=entry_strat())
def test_to_dict_never_contains_embedding(entry):
    """Embedding is always excluded from to_dict output."""
    emitted = KnowledgeEntry(
        id=entry.id,
        topic=entry.topic,
        model_family=entry.model_family,
        finding=entry.finding,
        details=entry.details,
        confidence=entry.confidence,
        tags=entry.tags,
        timestamp=entry.timestamp,
        source=entry.source,
        experiment_id=entry.experiment_id,
        metrics=entry.metrics,
        hyperparameters=entry.hyperparameters,
        embedding=[0.1, 0.2, 0.3],
        extra=entry.extra,
    )
    assert "embedding" not in emitted.to_dict()


@given(entry=entry_strat())
def test_to_dict_deterministic_and_embedding_stable(entry):
    """to_dict is deterministic, and a fixed embedding reproduces exactly."""
    emitted = KnowledgeEntry(
        id=entry.id,
        topic=entry.topic,
        model_family=entry.model_family,
        finding=entry.finding,
        details=entry.details,
        confidence=entry.confidence,
        tags=entry.tags,
        timestamp=entry.timestamp,
        source=entry.source,
        experiment_id=entry.experiment_id,
        metrics=entry.metrics,
        hyperparameters=entry.hyperparameters,
        embedding=[0.1, 0.2, 0.3],
        extra=entry.extra,
    )
    d1 = emitted.to_dict()
    d2 = emitted.to_dict()
    assert d1 == d2
    assert emitted.embedding == [0.1, 0.2, 0.3]
