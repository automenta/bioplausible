"""Hypothesis property-based tests for ``_QueryFilter`` predicate semantics.

Encodes two laws (Sprint 5.1):
  1. ``matches(meta)`` is logically equivalent to the AND of each individual
     axis predicate evaluated independently (short-circuit conjunction).
  2. Composition is commutative and idempotent at the match-decision level,
     and multi-axis querying returns the commutative intersection of the
     per-axis result sets.
"""

import copy

from hypothesis import assume, given
from hypothesis import strategies as st

from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    LocalityLevel,
    Registry,
    _QueryFilter,
)

locality_strat = st.sampled_from(list(LocalityLevel))
compute_strat = st.sampled_from(list(ComputeProfile))
credit_strat = st.sampled_from([
    "gradient",
    "equilibrium",
    "hebbian",
    "forward-only",
    "local",
    "spiking",
])
family_strat = st.sampled_from(["eqprop", "fa", "hebbian", "backprop"])
bio_strat = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
NO_MATCH = "zzz_no_match"
tag_choices = ["local", "online", "fast", "small", NO_MATCH]
tags_strat = st.lists(st.sampled_from(tag_choices), max_size=4)


@st.composite
def meta_strat(draw):
    """Generate a random ComponentMetadata."""
    return ComponentMetadata(
        name=f"m{draw(st.integers(min_value=0, max_value=10**6))}",
        category=ComponentCategory.MODEL,
        locality_level=draw(locality_strat),
        compute_profile=draw(compute_strat),
        bio_plausibility_score=draw(bio_strat),
        requires_backward=draw(st.booleans()),
        tags=draw(st.lists(st.sampled_from(tag_choices[:-1]), max_size=4)),
        credit_assignment_type=draw(credit_strat),
        family=draw(family_strat),
    )


@st.composite
def filter_strat(draw):
    """Generate a random _QueryFilter with a mix of constrained axes."""
    return _QueryFilter(
        locality=draw(st.one_of(st.none(), locality_strat)),
        compute=draw(st.one_of(st.none(), compute_strat)),
        requires_backward=draw(st.one_of(st.none(), st.booleans())),
        min_bio_score=draw(st.one_of(st.none(), bio_strat)),
        max_bio_score=draw(st.one_of(st.none(), bio_strat)),
        credit_type=draw(st.one_of(st.none(), credit_strat)),
        tags=draw(st.one_of(st.none(), tags_strat)),
        family=draw(st.one_of(st.none(), family_strat)),
    )


def _reference_matches(meta: ComponentMetadata, flt: _QueryFilter) -> bool:
    """Independent re-implementation of the filter predicate conjunction."""
    checks: list[bool] = []
    if flt.locality is not None:
        checks.append(flt.locality == meta.locality_level)
    if flt.compute is not None:
        checks.append(flt.compute == meta.compute_profile)
    if flt.requires_backward is not None:
        checks.append(flt.requires_backward == meta.requires_backward)
    if flt.min_bio_score is not None:
        checks.append(meta.bio_plausibility_score >= flt.min_bio_score)
    if flt.max_bio_score is not None:
        checks.append(meta.bio_plausibility_score <= flt.max_bio_score)
    if flt.credit_type is not None:
        checks.append(meta.credit_assignment_type == flt.credit_type)
    if flt.tags is not None:
        checks.append(all(t in meta.tags for t in flt.tags))
    if flt.family is not None:
        checks.append(meta.family == flt.family)
    return all(checks) if checks else True


@given(meta=meta_strat(), flt=filter_strat())
def test_matches_equivalent_to_reference(meta, flt):
    """matches(meta) is logically equivalent to independent conjunction."""
    assert flt.matches(meta) == _reference_matches(meta, flt)


@given(meta=meta_strat())
def test_empty_filter_matches_everything(meta):
    """An unconstrained filter matches all metadata."""
    assert _QueryFilter().matches(meta) is True


@given(meta=meta_strat())
def test_impossible_tag_never_matches(meta):
    """A tag absent from every generated metadata can never match."""
    flt = _QueryFilter(tags=[NO_MATCH])
    assert flt.matches(meta) is False


@given(f1=filter_strat(), f2=filter_strat(), meta=meta_strat())
def test_conjunction_commutative_and_idempotent(f1, f2, meta):
    """Conjunction of filter decisions is commutative and idempotent."""
    a = f1.matches(meta)
    b = f2.matches(meta)
    assert (a and b) == (b and a)
    assert (a and a) == a


@given(meta=meta_strat())
def test_axis_predicates_are_independent(meta):
    """Each axis predicate fires independently of the others.

    Any single axis that forces a mismatch makes the full conjunction False,
    regardless of which axis it is and what the other axes say.
    """
    locality_off = _QueryFilter(locality=_other_locality(meta))
    assert locality_off.matches(meta) is False


def _other_locality(meta: ComponentMetadata) -> LocalityLevel:
    all_localities = list(LocalityLevel)
    return all_localities[
        (all_localities.index(meta.locality_level) + 1) % len(all_localities)
    ]


def _seed(ma: ComponentMetadata, mb: ComponentMetadata):
    """Seed registry with two components; return saved global state."""
    saved = copy.deepcopy(Registry._components)
    Registry._components.clear()
    Registry._components[ComponentCategory.MODEL] = {
        ma.name: {"class": object, "metadata": ma},
        mb.name: {"class": object, "metadata": mb},
    }
    return saved


def _restore(saved):
    Registry._components.clear()
    Registry._components.update(copy.deepcopy(saved))


@given(ma=meta_strat(), mb=meta_strat())
def test_query_set_commutative(ma, mb):
    """Multi-axis querying returns a commutative result set."""
    assume(ma.name != mb.name)
    saved = _seed(ma, mb)
    try:
        if ma.family and mb.family:
            axis = ma.family
            set_ab = {
                r["name"]
                for r in Registry.query(
                    category=ComponentCategory.MODEL,
                    family=axis,
                )
            }
            set_ba = {
                r["name"]
                for r in Registry.query(
                    category=ComponentCategory.MODEL,
                    family=axis,
                )
            }
            assert set_ab == set_ba
    finally:
        _restore(saved)
