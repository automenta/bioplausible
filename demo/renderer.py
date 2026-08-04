"""Widget renderer (Sprint 3.2) — WidgetField/Group -> NiceGUI controls.

Two layers:

1. ``control_spec`` — a **pure**, browser-free transform that maps a
   :class:`WidgetField` to the (component + keyword args) needed to build a
   NiceGUI control. Unit-testable without importing NiceGUI.

2. ``render_group`` — a thin NiceGUI adapter that instantiates the actual
   ``ui.*`` widgets from a :class:`WidgetGroup` and binds change handlers that
   apply values back onto the config via :meth:`WidgetField.apply`.

Keeping the spec layer pure means the mapping logic (kind -> component, default
min/max for sliders, select/boolean mapping) is testable in CI without a
browser, matching the "UI stays a thin consumer" architecture rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from widgets import WidgetField, WidgetGroup

# NiceGUI component names we emit (strings so this file stays UI-agnostic).
SLIDER = "slider"
NUMBER = "number"
TEXT = "text"
SELECT = "select"
SWITCH = "switch"
READONLY = "readonly"


@dataclass(frozen=True)
class ControlSpec:
    """The (component, kwargs) needed to build one NiceGUI control."""

    component: str
    kwargs: dict[str, Any]


def control_spec(field: WidgetField) -> ControlSpec:
    """Map a :class:`WidgetField` to a writeable control spec.

    Sliders get a sane default ``[0, 1]`` range unless a wider one is needed
    (integers default to ``[0, 100]``). Selects pass through their options.
    Unsupported kinds degrade to a read-only ``ui.label``-style spec.
    """
    kind = field.kind
    if kind in ("slider", "number"):
        lo, hi = _slider_range(field)
        return ControlSpec(SLIDER, {"min": lo, "max": hi, "step": _step(field)})
    if kind == "integer":
        return ControlSpec(NUMBER, {"precision": 0})
    if kind == "text":
        return ControlSpec(TEXT, {})
    if kind == "boolean":
        return ControlSpec(SWITCH, {"value": bool(field.value)})
    if kind == "select":
        return ControlSpec(SELECT, {"options": list(field.options)})
    return ControlSpec(READONLY, {"value": _coerce_text(field.value)})


def _coerce_text(value: Any) -> str:
    return str(value) if value is not None else "—"


def _slider_range(field: WidgetField) -> tuple[float, float]:
    if field.min is not None and field.max is not None:
        return float(field.min), float(field.max)
    if field.kind == "integer":
        return 0.0, 100.0
    lo = 0.0 if isinstance(field.value, (int, float)) and field.value >= 0 else -1.0
    if isinstance(field.value, (int, float)):
        hi = max(float(field.value) * 2, 1.0)
    else:
        hi = 1.0
    return lo, hi


def _step(field: WidgetField) -> float:
    if field.kind == "integer":
        return 1.0
    return 0.05 if isinstance(field.value, float) else 1.0


def render_group(
    group: WidgetGroup,
    config: Any,
    on_change: Callable[[Any], None],
    container: Any,
) -> str:
    """Render a widget tree into ``container`` (a NiceGUI context).

    Creates a card titled ``group.label``, populates its fields with controls
    (each bound to ``on_change(fresh_config)`` via ``WidgetField.apply``) and
    recurses into child groups. Returns a short summary of created controls
    (useful for smoke assertions without a browser).

    ``import nicegui.ui as ui`` is done lazily so the pure spec layer and this
    module import cleanly in headless tests that never call ``render_group``.
    """
    from nicegui import ui

    created: list[str] = []
    with container, ui.card():
        ui.label(group.label).classes("text-bold")
        for field in group.fields:
            _render_field(field, config, on_change, created)
        for child in group.groups:
            created.append(render_group(child, config, on_change, container))
    return f"{group.label}: {len(group.fields)} fields, {len(group.groups)} groups"


def _render_field(
    field: WidgetField,
    config: Any,
    on_change: Callable[[Any], None],
    created: list[str],
) -> None:
    """Build one control bound to update the shared config object."""
    from nicegui import ui

    spec = control_spec(field)

    def commit(raw_value: Any) -> None:
        new_cfg = field.apply(config, raw_value)
        on_change(new_cfg)

    if spec.component == SLIDER:
        ui.slider(
            min=spec.kwargs["min"],
            max=spec.kwargs["max"],
            step=spec.kwargs["step"],
            value=float(field.value),
            on_change=lambda e: commit(e.value),
        )
    elif spec.component == NUMBER:
        ui.number(
            value=float(field.value),
            label=field.label,
            on_change=lambda e: commit(e.value),
            **spec.kwargs,
        )
    elif spec.component == TEXT:
        ui.input(
            value=str(field.value),
            label=field.label,
            on_change=lambda e: commit(e.value),
        )
    elif spec.component == SWITCH:
        ui.switch(
            field.label, value=bool(field.value), on_change=lambda e: commit(e.value)
        )
    elif spec.component == SELECT:
        ui.select(
            spec.kwargs["options"],
            value=str(field.value),
            label=field.label,
            on_change=lambda e: commit(e.value),
        )
    else:  # READONLY
        ui.label(f"{field.label}: {spec.kwargs['value']}").classes("text-grey")

    created.append(field.label)
