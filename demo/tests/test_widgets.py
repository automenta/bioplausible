"""Widget-tree descriptor tests (Sprint 3.2)."""

from dataclasses import dataclass, field

from bioplausible.core.trainer import TrainerConfig
from widgets import build_widget_tree


@dataclass
class Nested:
    sparsity: float = 0.5


@dataclass
class Outer:
    lr: float = 0.001
    epochs: int = 5
    nested: Nested = field(default_factory=Nested)


class TestWidgetTree:
    def test_builds_field_descriptors(self):
        group = build_widget_tree(Outer())
        names = {f.name for f in group.fields}
        assert names == {"lr", "epochs"}

    def test_nested_dataclass_recurse(self):
        group = build_widget_tree(Outer())
        assert len(group.groups) == 1
        assert group.groups[0].label == "Nested"
        assert group.groups[0].fields[0].name == "sparsity"

    def test_trainer_config_tree(self):
        cfg = TrainerConfig(model="backprop_mlp", task="mnist", epochs=3)
        group = build_widget_tree(cfg)
        names = {f.name for f in group.fields}
        assert "epochs" in names
        assert "task" in names

    def test_apply_to_frozen_dataclass(self):
        outer = Outer()
        group = build_widget_tree(outer)
        lr_field = next(f for f in group.fields if f.name == "lr")
        updated = lr_field.apply(outer, 0.01)
        assert updated.lr == 0.01
