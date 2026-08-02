"""Widget renderer spec-layer tests (Sprint 3.2) — browser-free."""

from dataclasses import dataclass, field

from renderer import (
    NUMBER,
    READONLY,
    SELECT,
    SLIDER,
    SWITCH,
    control_spec,
)
from widgets import WidgetField, build_widget_tree


@dataclass
class Nested:
    sparsity: float = 0.5


@dataclass
class Outer:
    lr: float = 0.001
    epochs: int = 5
    mode: str = "pc"
    train: bool = True
    nested: Nested = field(default_factory=Nested)


class TestControlSpec:
    def _field(self, name: str):
        return next(f for f in build_widget_tree(Outer()).fields if f.name == name)

    def test_float_becomes_slider_with_range(self):
        spec = control_spec(self._field("lr"))
        assert spec.component == SLIDER
        assert spec.kwargs["min"] <= 0.001 <= spec.kwargs["max"]

    def test_int_becomes_integer_number(self):
        spec = control_spec(self._field("epochs"))
        assert spec.component == NUMBER
        assert spec.kwargs["precision"] == 0

    def test_bool_becomes_switch(self):
        spec = control_spec(self._field("train"))
        assert spec.component == SWITCH
        assert spec.kwargs["value"] is True

    def test_sequence_becomes_readonly(self):
        spec = control_spec(WidgetField("tags", "Tags", "readonly", [1, 2]))
        assert spec.component == READONLY

    def test_none_value_readonly_text(self):
        spec = control_spec(WidgetField("x", "X", "readonly", None))
        assert spec.kwargs["value"] == "—"

    def test_select_passes_options(self):
        field_ = WidgetField("mode", "Mode", "select", "pc", options=("pc", "ep"))
        spec = control_spec(field_)
        assert spec.component == SELECT
        assert tuple(spec.kwargs["options"]) == ("pc", "ep")
