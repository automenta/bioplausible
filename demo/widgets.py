"""Config-driven widget descriptors (Sprint 3.2).

Pure, UI-agnostic layer that turns a Pydantic/dataclass config into a tree of
widget descriptors. The NiceGUI layer renders these; the logic here stays
unit-testable without a browser.

Every descriptor records its source field so the UI can round-trip a changed
value back onto the config object (:func:`apply`). Unsupported types degrade to
a read-only ``ReadOnlyField`` rather than crashing.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, get_args, get_origin

try:
    from pydantic import BaseModel

    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - environment guard
    _HAS_PYDANTIC = False


@dataclass(frozen=True)
class WidgetField:
    """A single editable widget descriptor."""

    name: str
    label: str
    kind: str  # "slider" | "number" | "integer" | "boolean" | "select" | "readonly"
    value: Any
    min: float | None = None
    max: float | None = None
    options: tuple[str, ...] = ()
    tooltip: str = ""
    # Nested location within the config root. Empty means the field is at the
    # top level (``path == (name,)``). Used for expanded ``dict`` knobs like
    # ``optimizer_kwargs["lr"]`` and ``model_kwargs["hidden_dim"]`` (Sprint 3.2).
    path: tuple[str, ...] = ()

    def apply(self, config: Any, value: Any) -> Any:
        """Return the config with this field set to ``value`` (in place for
        mutable parents).

        Supports mutable dataclasses (setattr), frozen dataclasses / Pydantic
        (rebuild on the leaf), and nested ``dict`` parents (the common demo
        case: ``optimizer_kwargs`` / ``model_kwargs``).
        """
        path = self.path or (self.name,)
        if len(path) == 1:
            return _set_leaf(config, path[0], value, name=self.name)
        parent = config
        for key in path[:-1]:
            parent = _read_child(parent, key)
        _set_child(parent, path[-1], value)
        return config


def _read_child(parent: Any, key: str) -> Any:
    if isinstance(parent, dict):
        return parent[key]
    return getattr(parent, key)


def _set_child(parent: Any, key: str, value: Any) -> None:
    if isinstance(parent, dict):
        parent[key] = value
    else:
        setattr(parent, key, value)


def _set_leaf(config: Any, key: str, value: Any, name: str) -> Any:
    if dataclasses.is_dataclass(config):
        if config.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            return dataclasses.replace(config, **{name: value})
        setattr(config, name, value)
        return config
    if _HAS_PYDANTIC and isinstance(config, BaseModel):
        return config.model_copy(update={name: value})
    if isinstance(config, dict):
        config = dict(config)
        config[name] = value
        return config
    setattr(config, name, value)
    return config


@dataclass(frozen=True)
class WidgetGroup:
    """A named group of fields (renders as an accordion/card in the UI)."""

    label: str
    fields: list[WidgetField]
    groups: list[WidgetGroup] = dataclasses.field(default_factory=list)


def _tooltip(annotated: Any) -> str:
    """Derive a tooltip from a field annotation or its docstring snippet."""
    return getattr(annotated, "__doc__", None) or ""


def _kind_for_annotation(annotation: Any, value: Any) -> tuple[str, tuple[str, ...]]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is bool or annotation is bool:
        return "boolean", ()
    if origin in (list, tuple, set, frozenset):
        return "readonly", ()
    if origin is type(None) or annotation is type(None):
        return "readonly", ()
    if isinstance(annotation, type) and issubclass(annotation, str):
        return "text", ()
    if origin is getattr(typing, "Literal", None):
        return "select", tuple(str(a) for a in args)

    # Numeric / Optional[T] — None-capable numerics are read-only (no slider).
    if annotation in (float, int) or origin in (float, int):
        has_none = len(args) == 2 and args[1] is type(None)
        if has_none:
            return "readonly", ()
        base = "integer" if origin is int or annotation is int else "number"
        return base, ()

    return _kind_from_value(value)


def _kind_from_value(value: Any) -> tuple[str, tuple[str, ...]]:
    """Heuristic fallback for unannotated or dynamic fields."""
    if isinstance(value, bool):
        return "boolean", ()
    if isinstance(value, int):
        return "integer", ()
    if isinstance(value, float):
        return "number", ()
    return "readonly", ()


def _is_leaf_dataclass(obj: Any) -> bool:
    if dataclasses.is_dataclass(obj):
        return True
    if _HAS_PYDANTIC and isinstance(obj, BaseModel):
        return True
    return False


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _iter_fields(obj: Any):
    """Yield (name, value, annotation) pairs for a dataclass/model object."""
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield f.name, getattr(obj, f.name), f.type
    elif _HAS_PYDANTIC and isinstance(obj, BaseModel):
        for name, field in obj.model_fields.items():
            yield name, getattr(obj, name), field.annotation
    elif isinstance(obj, dict):
        for key, val in obj.items():
            yield str(key), val, type(val)
    else:
        # Generic attribute introspection (objects without dataclass decorator).
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                yield name, getattr(obj, name), type(getattr(obj, name))
            except Exception:
                continue


def build_widget_tree(config: Any, root_label: str = "Config") -> WidgetGroup:
    """Build a nested :class:`WidgetGroup` tree for an arbitrary config object.

    Nested dataclasses/Pydantic models recurse into child groups (e.g.
    ``EquiTileConfig.tile.sparsity`` → nested accordion) per Sprint 3.2.
    Unsupported leaves degrade to read-only.
    """
    fields: list[WidgetField] = []
    groups: list[WidgetGroup] = []

    for name, value, annotation in _iter_fields(config):
        label = name.replace("_", " ").title()
        # Nested composite → recurse
        if _is_leaf_dataclass(value):
            groups.append(build_widget_tree(value, root_label=label))
            continue

        if isinstance(value, (list, tuple)) and value and _is_leaf_dataclass(value[0]):
            groups.append(build_widget_tree(value[0], root_label=f"{label}[0]"))
            continue

        # Dict of scalar knobs (e.g. optimizer_kwargs["lr"], model_kwargs
        # ["hidden_dim"]) → group of live controls instead of read-only JSON.
        if (
            isinstance(value, dict)
            and value
            and all(_is_scalar(v) for v in value.values())
        ):
            knob_fields = [
                WidgetField(
                    name=str(k),
                    label=str(k).replace("_", " ").title(),
                    kind=_kind_from_value(v)[0],
                    value=v,
                    options=_kind_from_value(v)[1],
                    path=(name, str(k)),
                )
                for k, v in value.items()
            ]
            groups.append(WidgetGroup(label=label, fields=knob_fields))
            continue

        kind, options = _kind_for_annotation(annotation, value)
        fields.append(
            WidgetField(
                name=name,
                label=label,
                kind=kind,
                value=value,
                options=options,
                tooltip=_tooltip(annotation),
            )
        )

    return WidgetGroup(label=root_label, fields=fields, groups=groups)
