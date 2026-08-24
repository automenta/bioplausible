"""Tests for the KnowledgeBase system."""

import os
import pathlib
import tempfile

import pytest

from computronium.knowledge import KnowledgeBase, KnowledgeEntry, create_knowledge_base


@pytest.fixture
def tmp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kb.db")
        yield db_path


def test_knowledge_base_creation(tmp_db_path):
    """Test creating a KnowledgeBase."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    assert kb is not None
    assert pathlib.Path(tmp_db_path).exists()


def test_add_entry(tmp_db_path):
    """Test adding a knowledge entry."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    entry = KnowledgeEntry(
        id="TEST-001",
        topic="Test",
        model_family="test_model",
        finding="Test finding",
        details="Test details",
        confidence=0.9,
        tags=["test", "pytest"],
    )
    entry_id = kb.add_entry(entry)
    assert entry_id == "TEST-001"


def test_query_by_id(tmp_db_path):
    """Test querying by entry ID."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry = KnowledgeEntry(
        id="TEST-002",
        topic="Test",
        model_family="test_model",
        finding="Test finding 2",
        details="Test details 2",
        confidence=0.8,
    )
    kb.add_entry(entry)

    retrieved = kb.get_by_id("TEST-002")
    assert retrieved is not None
    assert retrieved.finding == "Test finding 2"


def test_add_entry_auto_embed_true(tmp_db_path):
    """add_entry with auto_embed=True should not crash even without sentence-transformers."""
    kb = KnowledgeBase(db_path=tmp_db_path, auto_embed=True)
    entry = KnowledgeEntry(
        id="TEST-EMBED",
        topic="Test",
        model_family="test_model",
        finding="Auto-embed test",
        details="Testing auto-embed path",
        confidence=0.5,
    )
    entry_id = kb.add_entry(entry)
    assert entry_id == "TEST-EMBED"
    retrieved = kb.get_by_id("TEST-EMBED")
    assert retrieved is not None
    # Embedding should be None if no sentence-transformers available
    assert retrieved.embedding is None


def test_query_by_model_family(tmp_db_path):
    """Test querying by model family."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    for i in range(3):
        entry = KnowledgeEntry(
            id=f"MOD-{i:03d}",
            topic="Test",
            model_family=f"family_{i}",
            finding=f"Finding {i}",
            details=f"Details {i}",
            confidence=0.5 + i * 0.2,
        )
        kb.add_entry(entry)

    results = kb.query(model_family="family_1")
    assert len(results) == 1
    assert results[0].id == "MOD-001"


def test_query_by_tag(tmp_db_path):
    """Test querying by tag."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry1 = KnowledgeEntry(
        id="TAG-001",
        topic="A",
        model_family="m1",
        finding="f1",
        details="d1",
        confidence=0.5,
        tags=["alpha", "beta"],
    )
    entry2 = KnowledgeEntry(
        id="TAG-002",
        topic="B",
        model_family="m2",
        finding="f2",
        details="d2",
        confidence=0.5,
        tags=["alpha", "gamma"],
    )
    kb.add_entry(entry1)
    kb.add_entry(entry2)

    results = kb.query(tag="alpha")
    assert len(results) == 2

    results = kb.query(tag="beta")
    assert len(results) == 1
    assert results[0].id == "TAG-001"


def test_query_by_confidence(tmp_db_path):
    """Test querying by minimum confidence."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    for i in range(5):
        entry = KnowledgeEntry(
            id=f"CONF-{i:03d}",
            topic="Test",
            model_family="test_confidence",
            finding=f"f{i}",
            details="d",
            confidence=i * 0.25,
        )
        kb.add_entry(entry)

    results = kb.query(min_confidence=0.6, model_family="test_confidence")
    assert len(results) == 2  # 0.75 and 1.0


def test_add_experiment(tmp_db_path):
    """Test adding an experiment."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    exp_id = kb.add_experiment(
        name="test_exp",
        model_family="eqprop",
        task="mnist",
        config={"lr": 0.01, "epochs": 10},
        metrics={"accuracy": 0.95, "loss": 0.1},
    )

    assert exp_id is not None

    # Check it was added to experiments table
    exp = kb.get_experiment(exp_id)
    assert exp is not None
    assert exp["model_family"] == "eqprop"
    assert exp["task"] == "mnist"


def test_list_experiments(tmp_db_path):
    """Test listing experiments with filters."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    kb.add_experiment("exp1", "model_a", "mnist", {}, {"acc": 0.9})
    kb.add_experiment("exp2", "model_a", "cifar10", {}, {"acc": 0.8})
    kb.add_experiment("exp3", "model_b", "mnist", {}, {"acc": 0.7})

    results = kb.list_experiments(model_family="model_a")
    assert len(results) == 2

    results = kb.list_experiments(task="mnist")
    assert len(results) == 2


def test_get_stats(tmp_db_path):
    """Test getting knowledge base stats."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    stats = kb.get_stats()

    assert "total_entries" in stats
    assert "by_source" in stats
    assert "by_model_family" in stats
    assert "by_topic" in stats
    assert "total_experiments" in stats


def test_natural_language_query(tmp_db_path):
    """Test natural language query."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry = KnowledgeEntry(
        id="NL-001",
        topic="Test",
        model_family="test_model",
        finding="The quick brown fox jumps over the lazy dog",
        details="This is a test sentence for natural language queries.",
        confidence=0.95,
        tags=["test", "nlq"],
    )
    kb.add_entry(entry)

    answer = kb.natural_language_query("What is the fox doing?")
    assert "brown fox" in answer or "test_model" in answer


def test_export_json(tmp_db_path):
    """Test exporting knowledge base to JSON."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry = KnowledgeEntry(
        id="EXP-001",
        topic="Test",
        model_family="m",
        finding="test",
        details="test",
        confidence=0.5,
    )
    kb.add_entry(entry)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "export.json")
        kb.export_json(json_path)
        assert pathlib.Path(json_path).exists()

        import json

        with pathlib.Path(json_path).open() as f:
            data = json.load(f)
        assert len(data) >= 1


def test_create_knowledge_base_factory(tmp_db_path):
    """Test the factory function."""
    kb = create_knowledge_base(db_path=tmp_db_path)
    assert isinstance(kb, KnowledgeBase)


def test_seed_data_loading(tmp_db_path):
    """Test that seed data is loaded automatically."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    stats = kb.get_stats()
    assert stats["total_entries"] >= 3  # Seed data


def test_close(tmp_db_path):
    """Test close method."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.close()  # Should not raise


# --- KnowledgeEntry utility methods ---


def test_knowledge_entry_to_dict(tmp_db_path):
    """to_dict excludes embedding but includes all other fields."""
    entry = KnowledgeEntry(
        id="UTIL-001",
        topic="Test",
        model_family="m",
        finding="f",
        details="d",
        confidence=0.5,
        tags=["a", "b"],
        source="manual",
        metrics={"acc": 0.9},
        hyperparameters={"lr": 0.01},
    )
    d = entry.to_dict()
    assert "embedding" not in d
    assert d["id"] == "UTIL-001"
    assert d["finding"] == "f"
    assert d["metrics"] == {"acc": 0.9}
    assert d["hyperparameters"] == {"lr": 0.01}
    assert d["tags"] == ["a", "b"]


def test_knowledge_entry_from_dict():
    """from_dict reconstructs a KnowledgeEntry from a dict."""
    d = {
        "id": "UTIL-002",
        "topic": "Test",
        "model_family": "m",
        "finding": "f",
        "details": "d",
        "confidence": 0.75,
        "tags": ["tag1"],
        "source": "manual",
        "experiment_id": None,
        "metrics": {},
        "hyperparameters": {},
        "extra": {},
    }
    entry = KnowledgeEntry.from_dict(d)
    assert entry.id == "UTIL-002"
    assert entry.confidence == pytest.approx(0.75)
    assert entry.tags == ["tag1"]


def test_knowledge_entry_from_dict_ignores_extra_keys():
    """from_dict silently ignores keys not in KnowledgeEntry fields."""
    d = {
        "id": "UTIL-003",
        "topic": "T",
        "model_family": "m",
        "finding": "f",
        "details": "d",
        "confidence": 0.5,
        "nonexistent_field": "should be ignored",
    }
    entry = KnowledgeEntry.from_dict(d)
    assert entry.id == "UTIL-003"


# --- Surrogate model operations ---


def test_register_and_get_surrogate(tmp_db_path):
    """Register a surrogate model and retrieve it."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    sid = kb.register_surrogate(
        name="test_surrogate",
        model_type="rf",
        target_metric="val_accuracy",
        features=["lr", "batch_size"],
        performance={"r2": 0.85, "n_samples": 50},
        model_path="/tmp/test_model.pkl",
    )
    assert sid is not None
    assert len(sid) == 8

    retrieved = kb.get_surrogate("test_surrogate")
    assert retrieved is not None
    assert retrieved["name"] == "test_surrogate"
    assert retrieved["model_type"] == "rf"
    assert retrieved["target_metric"] == "val_accuracy"


def test_get_surrogate_nonexistent(tmp_db_path):
    """get_surrogate returns None for unknown name."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    assert kb.get_surrogate("nonexistent") is None


def test_list_surrogates(tmp_db_path):
    """list_surrogates returns all registered surrogates."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    kb.register_surrogate("s1", "rf", "acc", ["lr"], {"r2": 0.9})
    kb.register_surrogate("s2", "gp", "loss", ["lr", "bs"], {"r2": 0.8})

    surrogates = kb.list_surrogates()
    assert len(surrogates) == 2
    names = [s["name"] for s in surrogates]
    assert "s1" in names
    assert "s2" in names


# --- Keyword search (FAISS fallback) ---


def test_keyword_search_fallback(tmp_db_path):
    """search falls back to keyword when FAISS is unavailable."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    entry = KnowledgeEntry(
        id="SRCH-001",
        topic="Optimization",
        model_family="eqprop",
        finding="Equilibrium propagation achieves O(1) memory scaling",
        details="EP requires constant memory regardless of trajectory length",
        confidence=0.95,
        tags=["memory", "eqprop"],
    )
    kb.add_entry(entry)

    # Vector_index is None when FAISS is not installed, should use keyword fallback
    results = kb.search("memory scaling", k=5)
    assert len(results) >= 1
    entry_result, score = results[0]
    assert entry_result.id == "SRCH-001"
    assert score > 0


def test_keyword_search_with_filters(tmp_db_path):
    """search with filters restricts results."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.add_entry(
        KnowledgeEntry(
            id="SRCH-F1",
            topic="A",
            model_family="m1",
            finding="alpha beta gamma",
            details="",
            confidence=0.5,
        )
    )
    kb.add_entry(
        KnowledgeEntry(
            id="SRCH-F2",
            topic="B",
            model_family="m2",
            finding="alpha beta delta",
            details="",
            confidence=0.5,
        )
    )

    # Filter by model_family
    results = kb.search("alpha beta", k=10, filters={"model_family": "m1"})
    assert len(results) == 1
    assert results[0][0].id == "SRCH-F1"


def test_keyword_search_empty_query(tmp_db_path):
    """search with empty query returns empty list."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.add_entry(
        KnowledgeEntry(
            id="SRCH-E",
            topic="T",
            model_family="m",
            finding="test",
            details="",
            confidence=0.5,
        )
    )
    results = kb.search("", k=5)
    assert len(results) == 0


def test_keyword_search_no_match(tmp_db_path):
    """search with query matching nothing returns empty list."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.add_entry(
        KnowledgeEntry(
            id="SRCH-NO",
            topic="T",
            model_family="m",
            finding="unique finding text",
            details="",
            confidence=0.5,
        )
    )
    results = kb.search("zzzzzzyxwvutsrqponmlkjihgfedcba", k=5)
    assert len(results) == 0


# --- Predict outcome (stub/implicit tests) ---


def test_predict_outcome_no_surrogate(tmp_db_path):
    """predict_outcome returns 0.0 when no surrogate is trained."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    result = kb.predict_outcome({"lr": 0.01})
    assert result == pytest.approx(0.0)


# --- Causal analysis edge cases ---


def test_causal_analysis_no_data(tmp_db_path):
    """run_causal_analysis returns error when no experiments exist."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    result = kb.run_causal_analysis(outcome="val_accuracy")
    assert "error" in result
    assert "Not enough data" in result["error"]


def test_causal_analysis_with_data(tmp_db_path):
    """run_causal_analysis with enough experiments returns correlations."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    for i in range(12):
        kb.add_experiment(
            name=f"exp_{i}",
            model_family="test_model",
            task="mnist",
            config={"lr": 0.001 * (i + 1), "hidden_dim": 128, "batch_size": 64},
            metrics={"val_accuracy": 0.5 + i * 0.04},
        )

    result = kb.run_causal_analysis(outcome="val_accuracy")
    assert "error" not in result
    assert result["outcome"] == "val_accuracy"
    assert "correlations" in result
    assert "ranked_factors" in result
    assert result["n_samples"] >= 10


# --- Edge cases ---


def test_duplicate_id_overwrites(tmp_db_path):
    """Adding an entry with duplicate ID overwrites via INSERT OR REPLACE."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    entry1 = KnowledgeEntry(
        id="DUP-001",
        topic="A",
        model_family="m",
        finding="original",
        details="",
        confidence=0.5,
    )
    entry2 = KnowledgeEntry(
        id="DUP-001",
        topic="B",
        model_family="m",
        finding="replacement",
        details="",
        confidence=0.9,
    )
    kb.add_entry(entry1)
    kb.add_entry(entry2)

    retrieved = kb.get_by_id("DUP-001")
    assert retrieved is not None
    assert retrieved.finding == "replacement"
    assert retrieved.confidence == pytest.approx(0.9)


def test_get_by_id_nonexistent(tmp_db_path):
    """get_by_id returns None for unknown ID."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    assert kb.get_by_id("NONEXISTENT") is None


def test_get_experiment_nonexistent(tmp_db_path):
    """get_experiment returns None for unknown ID."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    assert kb.get_experiment("nonexistent-id") is None


def test_query_with_multiple_filters(tmp_db_path):
    """query with combined model_family + min_confidence + topic."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    entries = [
        KnowledgeEntry(
            id=f"MF-{i:02d}",
            topic="A",
            model_family="m1",
            finding=f"f{i}",
            details="",
            confidence=0.5 + i * 0.2,
        )
        for i in range(5)
    ]
    for e in entries:
        kb.add_entry(e)

    results = kb.query(model_family="m1", min_confidence=0.8, topic="A")
    assert len(results) == 3  # confidence 0.9, 1.1, 1.3


def test_query_by_source(tmp_db_path):
    """query filters by source field."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.add_entry(
        KnowledgeEntry(
            id="SRC-1",
            topic="T",
            model_family="m",
            finding="from experiment",
            details="",
            confidence=0.5,
            source="experiment",
        )
    )
    kb.add_entry(
        KnowledgeEntry(
            id="SRC-2",
            topic="T",
            model_family="m",
            finding="from literature",
            details="",
            confidence=0.5,
            source="literature",
        )
    )
    results = kb.query(source="experiment")
    assert len(results) == 1
    assert results[0].id == "SRC-1"


def test_query_by_experiment_id(tmp_db_path):
    """query filters by experiment_id."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.add_entry(
        KnowledgeEntry(
            id="EXP-Q1",
            topic="T",
            model_family="m",
            finding="exp result",
            details="",
            confidence=0.5,
            experiment_id="exp-123",
        )
    )
    kb.add_entry(
        KnowledgeEntry(
            id="EXP-Q2",
            topic="T",
            model_family="m",
            finding="not linked",
            details="",
            confidence=0.5,
        )
    )
    results = kb.query(experiment_id="exp-123")
    assert len(results) == 1
    assert results[0].id == "EXP-Q1"


def test_query_limit(tmp_db_path):
    """query respects the limit parameter."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    for i in range(20):
        kb.add_entry(
            KnowledgeEntry(
                id=f"LIM-{i:02d}",
                topic="T",
                model_family="m",
                finding=f"entry {i}",
                details="",
                confidence=0.5,
            )
        )
    results = kb.query(limit=5)
    assert len(results) == 5


def test_empty_kb_query(tmp_db_path):
    """query on empty KB returns empty list."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    results = kb.query(model_family="anything")
    assert len(results) == 0


def test_add_experiment_with_explicit_id(tmp_db_path):
    """add_experiment with explicit experiment_id uses it."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    exp_id = kb.add_experiment(
        name="exp_with_id",
        model_family="m",
        task="t",
        config={},
        metrics={"acc": 0.9},
        experiment_id="my-custom-id",
    )
    assert exp_id == "my-custom-id"
    exp = kb.get_experiment("my-custom-id")
    assert exp is not None
    assert exp["name"] == "exp_with_id"


def test_add_entry_with_embedding_preserved(tmp_db_path):
    """add_entry preserves a pre-set embedding without auto_embed overriding it."""
    kb = KnowledgeBase(db_path=tmp_db_path, auto_embed=False)
    entry = KnowledgeEntry(
        id="EMB-PRE",
        topic="T",
        model_family="m",
        finding="pre-embedded",
        details="",
        confidence=0.5,
        embedding=[0.1] * 384,
    )
    kb.add_entry(entry)
    retrieved = kb.get_by_id("EMB-PRE")
    assert retrieved is not None
    # Embedding not stored in SQLite — only in FAISS index (unavailable here)
    # But verify the entry was stored without errors
    assert retrieved.finding == "pre-embedded"


def test_knowledge_entry_all_fields_populated(tmp_db_path):
    """KnowledgeEntry with all optional fields set stores and retrieves correctly."""
    import time

    now = time.time()
    entry = KnowledgeEntry(
        id="ALL-FIELDS",
        topic="Comprehensive",
        model_family="test_model",
        finding="All fields test",
        details="Testing all optional fields",
        confidence=0.99,
        tags=["tag1", "tag2"],
        timestamp=now,
        source="experiment",
        experiment_id="exp-all",
        metrics={"accuracy": 0.99, "loss": 0.01},
        hyperparameters={"lr": 0.001, "epochs": 100},
        embedding=[0.5] * 384,
        extra={"notes": "test all fields"},
    )
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.add_entry(entry)

    retrieved = kb.get_by_id("ALL-FIELDS")
    assert retrieved is not None
    assert retrieved.topic == "Comprehensive"
    assert retrieved.tags == ["tag1", "tag2"]
    assert abs(retrieved.timestamp - now) < 1.0
    assert retrieved.source == "experiment"
    assert retrieved.experiment_id == "exp-all"
    assert retrieved.metrics == {"accuracy": 0.99, "loss": 0.01}
    assert retrieved.hyperparameters == {"lr": 0.001, "epochs": 100}
    # Embedding is stored in FAISS index, not SQLite — not retrievable via get_by_id
    assert retrieved.embedding is None
    assert retrieved.extra == {"notes": "test all fields"}


# --- Surrogate training stub ---


def test_train_surrogate_no_experiments(tmp_db_path):
    """train_surrogate returns None when no experiments exist."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    result = kb.train_surrogate(target_metric="val_accuracy")
    assert result is None


def test_extract_symbolic_rules_empty(tmp_db_path):
    """extract_symbolic_rules returns error message when no data."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    rules = kb.extract_symbolic_rules()
    assert isinstance(rules, list)
    assert len(rules) >= 1
    assert "no data" in rules[0].lower()


def test_compute_algorithm_similarity_empty(tmp_db_path):
    """compute_algorithm_similarity returns empty dict when no data."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    result = kb.compute_algorithm_similarity()
    assert result == {}
