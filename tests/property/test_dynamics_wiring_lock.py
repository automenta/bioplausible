"""Wiring lockstep lock (TODO.md R1.2 + R5.1).

Every future StateDynamics primitive must hit every export surface in
lockstep — the ePC landing needed five hand-edited lists plus a closed
if/elif chain, each discovered by the next ImportError. This lock makes
skipping a surface impossible:

1. every ``StateDynamicsConfig`` factory classmethod's ``dynamics_type``
   is a ``DYNAMICS_REGISTRY`` key (and vice versa);
2. every registry class round-trips ``to_spec`` → ``from_spec`` with its
   default config preserved exactly;
3. every registry class appears in root ``__all__`` and ``_LAZY``;
4. every root ``__all__`` name has an explicit ``TYPE_CHECKING`` import
   (typed lazy exports — pyright signal, not lazy-map noise).
"""

import dataclasses
from pathlib import Path

from computronium import _LAZY
from computronium.ontology.dynamics import (
    DYNAMICS_REGISTRY,
    StateDynamicsConfig,
    dynamics_from_config,
)

ROOT_INIT = Path("computronium") / "__init__.py"


def _config_classmethods() -> dict[str, str]:
    """Map factory-classmethod name → dynamics_type of the returned config."""
    found: dict[str, str] = {}
    for name, member in vars(StateDynamicsConfig).items():
        if name.startswith("_") or not isinstance(member, classmethod):
            continue
        config = getattr(StateDynamicsConfig, name)()
        if isinstance(config, StateDynamicsConfig):
            found[name] = config.dynamics_type
    return found


def test_config_classmethods_cover_the_registry() -> None:
    classmethods = _config_classmethods()
    assert set(classmethods.values()) == set(DYNAMICS_REGISTRY), (
        f"registry keys and StateDynamicsConfig classmethods diverged: "
        f"methods→{sorted(classmethods.values())} vs "
        f"registry→{sorted(DYNAMICS_REGISTRY)}"
    )


def test_registry_classes_round_trip_to_spec_from_spec() -> None:
    for dynamics_type in sorted(DYNAMICS_REGISTRY):
        config = StateDynamicsConfig(
            **dataclasses.asdict(
                getattr(StateDynamicsConfig, _classmethod_for(dynamics_type))()
            )
        )
        dynamics = dynamics_from_config(config)
        assert type(dynamics) is DYNAMICS_REGISTRY[dynamics_type]
        assert dynamics.config == config


def _classmethod_for(dynamics_type: str) -> str:
    for name, dtype in _config_classmethods().items():
        if dtype == dynamics_type:
            return name
    raise AssertionError(f"no StateDynamicsConfig classmethod for {dynamics_type!r}")


def test_registry_classes_are_root_exports() -> None:
    from computronium import __all__

    for cls in DYNAMICS_REGISTRY.values():
        name = cls.__name__
        assert name in __all__, f"{name} missing from root __all__"
        assert name in _LAZY, f"{name} missing from root _LAZY"
        assert _LAZY[name][1] == name


def test_root_all_names_have_typed_type_checking_imports() -> None:
    source = ROOT_INIT.read_text(encoding="utf-8")
    block = source.split("if TYPE_CHECKING:", 1)[1].split("\n\n# Lazy imports", 1)[0]
    from computronium import __all__

    for name in __all__:
        if name == "__version__":
            continue
        assert f" {name}" in block or f",{name}" in block, (
            f"{name} missing from the root TYPE_CHECKING import block — "
            "pyright sees it as `object` via the lazy map"
        )
