"""Snapshot tests for _QueryFilter predicate dispatch table (task 1.6).

Verifies that:
  - Predicates are deterministically selected based on filter fields.
  - Each predicate performs correct axis matching.
  - matches() short-circuits correctly.
  - Empty filter matches everything.
"""

from computronium.core.registry import (
    ComponentMetadata,
    ComputeProfile,
    LocalityLevel,
    _ComputeIs,
    _CreditTypeIs,
    _FamilyIs,
    _LocalityIs,
    _MaxBioScore,
    _MinBioScore,
    _QueryFilter,
    _RequiresBackwardIs,
    _TagsAll,
)

# ---- Baseline metadata ----
_META = ComponentMetadata(
    name="test_comp",
    category="model",
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.CPU,
    requires_backward=False,
    bio_plausibility_score=0.8,
    credit_assignment_type="hebbian",
    tags=["local", "online"],
    family="tile",
)

# ---- Predicate dispatch table (__post_init__) ->


def test_predicate_dispatch_empty() -> None:
    """No constraints -> empty predicate list."""
    q = _QueryFilter()
    assert q._predicates == ()


def test_predicate_dispatch_single() -> None:
    """Each non-None field adds exactly one predicate."""
    q = _QueryFilter(locality=LocalityLevel.LOCAL)
    assert len(q._predicates) == 1
    assert isinstance(q._predicates[0], _LocalityIs)

    q = _QueryFilter(compute=ComputeProfile.CPU)
    assert len(q._predicates) == 1
    assert isinstance(q._predicates[0], _ComputeIs)


def test_predicate_dispatch_all_fields() -> None:
    """All fields specified -> 8 predicates in deterministic order."""
    q = _QueryFilter(
        locality=LocalityLevel.LOCAL,
        compute=ComputeProfile.CPU,
        requires_backward=False,
        min_bio_score=0.3,
        max_bio_score=0.9,
        credit_type="hebbian",
        tags=["local"],
        family="tile",
    )
    assert len(q._predicates) == 8
    expected_types = [
        _LocalityIs,
        _ComputeIs,
        _RequiresBackwardIs,
        _MinBioScore,
        _MaxBioScore,
        _CreditTypeIs,
        _TagsAll,
        _FamilyIs,
    ]
    for pred, expected in zip(q._predicates, expected_types, strict=True):
        assert isinstance(pred, expected), f"Expected {expected}, got {type(pred)}"


# ---- Individual predicate correctness ----


def test_locality_predicate() -> None:
    """_LocalityIs: exact match on locality."""
    p = _LocalityIs(LocalityLevel.LOCAL)
    assert p(_META)
    assert not _LocalityIs(LocalityLevel.GLOBAL)(_META)


def test_compute_predicate() -> None:
    """_ComputeIs: exact match on compute profile."""
    p = _ComputeIs(ComputeProfile.CPU)
    assert p(_META)
    assert not _ComputeIs(ComputeProfile.GPU)(_META)


def test_requires_backward_predicate() -> None:
    """_RequiresBackwardIs: exact match on bool."""
    assert _RequiresBackwardIs(False)(_META)
    assert not _RequiresBackwardIs(True)(_META)


def test_min_bio_score_predicate() -> None:
    """_MinBioScore: True iff score >= min."""
    assert _MinBioScore(0.8)(_META)
    assert _MinBioScore(0.5)(_META)
    assert not _MinBioScore(0.9)(_META)


def test_max_bio_score_predicate() -> None:
    """_MaxBioScore: True iff score <= max."""
    assert _MaxBioScore(0.8)(_META)
    assert _MaxBioScore(1.0)(_META)
    assert not _MaxBioScore(0.7)(_META)


def test_credit_type_predicate() -> None:
    """_CreditTypeIs: exact match on credit assignment string."""
    assert _CreditTypeIs("hebbian")(_META)
    assert not _CreditTypeIs("gradient")(_META)


def test_tags_all_predicate() -> None:
    """_TagsAll: True iff meta has ALL required tags."""
    assert _TagsAll(frozenset({"local", "online"}))(_META)
    assert _TagsAll(frozenset({"local"}))(_META)
    assert not _TagsAll(frozenset({"local", "spiking"}))(_META)


def test_family_predicate() -> None:
    """_FamilyIs: exact match on family string."""
    assert _FamilyIs("tile")(_META)
    assert not _FamilyIs("backprop")(_META)


# ---- matches() integration ----


def test_matches_passes_all_constraints() -> None:
    """matches() returns True when all predicates pass."""
    q = _QueryFilter(
        locality=LocalityLevel.LOCAL,
        min_bio_score=0.5,
        tags=["local"],
    )
    assert q.matches(_META)


def test_matches_fails_on_mismatch() -> None:
    """matches() returns False when any predicate fails."""
    q = _QueryFilter(locality=LocalityLevel.GLOBAL)
    assert not q.matches(_META)


def test_matches_empty_filter() -> None:
    """Empty filter matches everything (zero predicates = all pass)."""
    assert _QueryFilter().matches(_META)
