"""Hypothesis property-based tests for Registry register/get round-trip
(Sprint 5.4).

Laws:
  - get(register(x)) == x: registering a component and fetching it returns
    the identical object.
  - get_metadata returns the metadata that was provided at registration
    (name, category, family, locality, bio score preserved).
  - registering a name twice overwrites; get returns the latest component.
"""

import copy

from hypothesis import given
from hypothesis import strategies as st

from computronium.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    LocalityLevel,
    Registry,
)

name_strat = st.integers(min_value=0, max_value=10**6).map(lambda i: f"pcomp_{i}")
family_strat = st.sampled_from(["eqprop", "fa", "hebbian", "backprop"])


class _Dummy:
    """Generic registered component (used to avoid registry pollution by name)."""


def _snapshot():
    saved = copy.deepcopy(Registry._components)
    Registry._components.clear()
    return saved


def _restore(saved):
    Registry._components.clear()
    Registry._components.update(copy.deepcopy(saved))


@given(
    name=name_strat,
    family=family_strat,
    bio=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
)
def test_get_returns_registered_component(name, family, bio):
    """get(register(x)) == x and metadata is preserved."""
    saved = _snapshot()
    try:
        Registry.register(
            ComponentCategory.MODEL,
            name,
            family=family,
            bio_plausibility_score=bio,
            locality_level=LocalityLevel.LOCAL,
        )(_Dummy)
        got = Registry.get(ComponentCategory.MODEL, name)
        assert got is _Dummy
        meta = Registry.get_metadata(ComponentCategory.MODEL, name)
        assert isinstance(meta, ComponentMetadata)
        assert meta.name == name and meta.category == ComponentCategory.MODEL
        assert meta.family == family
        assert meta.bio_plausibility_score == bio
    finally:
        _restore(saved)


@given(name=name_strat)
def test_registration_then_query_finds_it(name):
    """A registered component is discoverable via query and list."""
    saved = _snapshot()
    try:
        Registry.register(ComponentCategory.PARAM_UPDATE, name, family="fa")(_Dummy)
        listed = Registry.list(ComponentCategory.PARAM_UPDATE)
        assert name in listed[ComponentCategory.PARAM_UPDATE.value]
        results = Registry.query(category=ComponentCategory.PARAM_UPDATE, family="fa")
        assert any(r["name"] == name for r in results)
    finally:
        _restore(saved)


@given(name=name_strat)
def test_register_overwrites_returns_latest(name):
    """Re-registering a name overwrites; get returns the latest component."""
    saved = _snapshot()
    try:

        class _First:
            pass

        class _Second:
            pass

        Registry.register(ComponentCategory.MODEL, name)(_First)
        Registry.register(ComponentCategory.MODEL, name, family="backprop")(_Second)
        assert Registry.get(ComponentCategory.MODEL, name) is _Second
        meta = Registry.get_metadata(ComponentCategory.MODEL, name)
        assert meta.family == "backprop"
        assert Registry.get(ComponentCategory.MODEL, name) is not _First
    finally:
        _restore(saved)


@given(name=name_strat)
def test_round_trip_identity(name):
    """Register/fetch preserves object identity for arbitrary callables."""
    saved = _snapshot()
    try:

        def factory():
            return 42

        Registry.register(ComponentCategory.CREDIT_ASSIGNMENT, name)(factory)
        assert Registry.get(ComponentCategory.CREDIT_ASSIGNMENT, name) is factory
    finally:
        _restore(saved)
