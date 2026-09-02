"""Lock on the package root's export mechanism: ``_LAZY`` map ≡ ``__all__``.

Both are hand-maintained and can drift apart silently. Any name in one but
not the other fails here, so a root export can never resolve to an
unadvertised symbol (or advertise one that raises AttributeError).
"""

import computronium


def test_lazy_map_and_all_are_in_lockstep() -> None:
    lazy = set(computronium._LAZY)
    all_ = set(computronium.__all__) - {"__version__"}
    assert lazy == all_, {
        "in_LAZY_not_in_all": sorted(lazy - all_),
        "in_all_not_in_LAZY": sorted(all_ - lazy),
    }


def test_lazy_entries_resolve() -> None:
    for name, (module, attr) in sorted(computronium._LAZY.items()):
        imported = __import__(module, fromlist=[attr] if attr else ["*"])
        if attr is not None:
            assert hasattr(imported, attr), f"{module}.{attr} missing for root {name!r}"
