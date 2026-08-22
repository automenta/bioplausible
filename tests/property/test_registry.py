"""Hypothesis property-based tests for Registry query monotonicity."""

import copy

from hypothesis import assume, given
from hypothesis import strategies as st

from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    LocalityLevel,
    Registry,
)

# ---------- strategies for metadata fields ----------
locality_strat = st.sampled_from(list(LocalityLevel))
compute_strat = st.sampled_from(list(ComputeProfile))
bio_score_strat = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
backward_strat = st.booleans()
tags_strat = st.lists(st.text(min_size=1, max_size=10, alphabet="abcdef"), max_size=5)
credit_strat = st.sampled_from(["gradient", "equilibrium", "hebbian", "forward-only"])
family_strat = st.sampled_from([
    "eqprop",
    "fa",
    "hebbian",
    "backprop",
    "predictive_coding",
])


@st.composite
def metadata_strat(draw):
    """Generate a random ComponentMetadata."""
    return ComponentMetadata(
        name=f"test_{draw(st.integers(min_value=0, max_value=10000))}",
        category=ComponentCategory.MODEL,
        locality_level=draw(locality_strat),
        compute_profile=draw(compute_strat),
        bio_plausibility_score=draw(bio_score_strat),
        requires_backward=draw(backward_strat),
        tags=draw(tags_strat),
        credit_assignment_type=draw(credit_strat),
        family=draw(family_strat),
        description="property test metadata",
    )


# ---------- fixtures ----------
def _setup():
    """Setup fresh registry state, return saved state for restore."""
    saved = copy.deepcopy(Registry._components)
    Registry._components.clear()
    return saved


def _restore(saved):
    Registry._components.clear()
    Registry._components.update(copy.deepcopy(saved))


@given(meta=metadata_strat())
def test_query_matches_itself(meta):
    """A metadata matches a filter built from its own fields."""
    saved = _setup()
    try:
        Registry._components[ComponentCategory.MODEL] = {
            meta.name: {"class": object, "metadata": meta}
        }
        results = Registry.query(
            category=ComponentCategory.MODEL,
            locality=meta.locality_level,
            compute=meta.compute_profile,
            requires_backward=meta.requires_backward,
            family=meta.family,
        )
        names = {r["name"] for r in results}
        assert meta.name in names
    finally:
        _restore(saved)


@given(
    meta1=metadata_strat(),
    meta2=metadata_strat(),
)
def test_query_monotonic_constraining_family(meta1, meta2):
    """Adding a family constraint never adds results."""
    assume(meta1.name != meta2.name)
    saved = _setup()
    try:
        Registry._components[ComponentCategory.MODEL] = {
            meta1.name: {"class": object, "metadata": meta1},
            meta2.name: {"class": object, "metadata": meta2},
        }
        all_results = Registry.query(category=ComponentCategory.MODEL)
        # Constrain by a single family
        if meta1.family:
            f = meta1.family
            constrained = Registry.query(category=ComponentCategory.MODEL, family=f)
            assert len(constrained) <= len(all_results)
    finally:
        _restore(saved)


@given(
    meta=metadata_strat(),
)
def test_query_monotonic_constraining_bio_score(meta):
    """Adding a min_bio_score constraint never adds results."""
    saved = _setup()
    try:
        Registry._components[ComponentCategory.MODEL] = {
            meta.name: {"class": object, "metadata": meta}
        }
        all_results = Registry.query(category=ComponentCategory.MODEL)
        constrained = Registry.query(
            category=ComponentCategory.MODEL,
            min_bio_score=meta.bio_plausibility_score,
        )
        assert len(constrained) <= len(all_results)
    finally:
        _restore(saved)


@given(
    meta=metadata_strat(),
)
def test_query_empty_for_exclusive_constraint(meta):
    """A filter with no matching family returns empty."""
    saved = _setup()
    try:
        Registry._components[ComponentCategory.MODEL] = {
            meta.name: {"class": object, "metadata": meta}
        }
        # Use a family guaranteed not to match
        if meta.family:
            other_families = ["eqprop", "fa", "hebbian", "backprop", "predictive_coding", "mep", "tile"]
            other = [f for f in other_families if f != meta.family]
            if other:
                results = Registry.query(
                    category=ComponentCategory.MODEL, family=other[0]
                )
                assert len(results) == 0
    finally:
        _restore(saved)
